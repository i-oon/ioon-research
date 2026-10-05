"""Does a frame-reading Froude head memorise pose-in-room in the training rooms? (F313 follow-up, 2026-10-05)

Hypothesis (user): each training room holds one source clip and all its branches start at the same place, so after a
switch the frame shows where the robot has got to in a room seen many times -- pose (especially heading) in a known room
gives away what it has been doing, without z. That works only in training rooms.

Measure, per checkpoint (arm B: head(z); arm F: head(e_t, z)), the Froude-head error on branch pairs t = b + j (j = 0..10,
b = the branch frame) from TRAINING rooms (`*_branches_train`, a fixed subset of groups) and HELDOUT rooms
(`*_branches_heldout`), with the true z = ITM(e_t, e_t+5) and with z shuffled across the whole set (other groups / commands).
Target = the training label (mean 1 s CoM Froude over [t, t+5)), standardised with the checkpoint's body_stats; error = MSE
per channel. Leak signature: arm F much better than arm B in training rooms at j >= 1 but not at j = 0 (the switch frame
cannot show the new command yet) and not in heldout rooms; and a bigger train-vs-heldout gap for F than for B.

    .venv/bin/python3 scripts/diagnostics/forward_model/frame_head_leak_check.py --embodiment hexapod --body c10
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from counterfactual_readout import groups_of, tokens  # noqa: E402
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402

N_TOK = 16          # frames b .. b+15, the read-out cache's range (P_max 11 + K 5)
J = 11              # pairs j = 0..10


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", default="hexapod", choices=("hexapod", "b1"))
    ap.add_argument("--body", default="c10")
    ap.add_argument("--groups", type=int, default=16, help="groups per room set (24 commands each)")
    ap.add_argument("--ckpt", nargs="+", default=["B=wm/runs/round1_branches_s0/best.pt",
                                                   "F=wm/runs/round2_framehead_s0/best.pt"])
    ap.add_argument("--out", default="results/check/frame_head_leak")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    adim = 18 if args.embodiment == "hexapod" else 12

    sets = {}
    for split in ("train", "heldout"):
        G = groups_of(f"data/counterfactual_walks/{args.body}_branches_{split}")
        keys = sorted(G)
        rng = np.random.default_rng(0)
        keys = [keys[i] for i in sorted(rng.choice(len(keys), min(args.groups, len(keys)), replace=False))]
        sets[split] = [p for k in keys for p in G[k]]
        print(f"{split}: {len(keys)} groups, {len(sets[split])} branch files")

    held = {}

    def encoder():
        if "enc" not in held:
            from vjepa2_encoder import VJEPA2FrameEncoder
            held["enc"] = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
        return held["enc"]

    models = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), args.embodiment, adim, dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [torch.as_tensor(np.asarray(x).ravel()[:3], dtype=torch.float32, device=dev) for x in ck["body_stats"]]
        frame = bool(getattr(m.md, "body_sees_frame", False))
        off = None if m.offset is None else m.offset.float().to(dev)
        models.append(dict(name=name, m=m, mean=mean, std=std, frame=frame, off=off))
        print(f"{name}: body_sees_frame={frame}, stride {m.stride}")
        del ck

    lines = [f"{args.body} ({args.embodiment}): Froude-head MSE (standardised, mean of fwd/lat/yaw; per channel in brackets)"]
    for split, files in sets.items():
        # per model: lists of (j, pred_true, pred_shuf_input) collected after a first pass gathers all z for shuffling
        rec = {r["name"]: collections.defaultdict(list) for r in models}
        allz = {r["name"]: [] for r in models}
        cache = []
        for p in files:
            c = load(p, REGISTRY[args.embodiment], lazy_frames=True)
            b = int(c["first_pair"])
            e = tokens(p, c, b, N_TOK, encoder).float().to(dev)
            bm = np.asarray(c["body_motion"])[:, :3]
            K = models[0]["m"].stride
            tgt = torch.as_tensor(np.stack([bm[b + j:b + j + K].mean(0) for j in range(J)]), dtype=torch.float32, device=dev)
            cache.append((e.half().cpu(), tgt))
            for r in models:
                ee = e if r["off"] is None else e - r["off"].reshape(e.shape[1:])
                z = r["m"].itm(ee[:J], ee[K:K + J])
                allz[r["name"]].append(z)
        for r in models:
            Z = torch.cat(allz[r["name"]])
            perm = torch.randperm(len(Z), generator=torch.Generator().manual_seed(0)).to(dev)
            Zs = Z[perm]
            i = 0
            for e, tgt in cache:
                e = e.float().to(dev)
                ee = e if r["off"] is None else e - r["off"].reshape(e.shape[1:])
                x = ee[:J] if r["frame"] else None
                y = (tgt - r["mean"]) / r["std"]
                for kind, zz in (("true", Z[i:i + J]), ("shuf", Zs[i:i + J])):
                    pred = r["m"].md.body(x, zz).float()
                    se = (pred - y) ** 2
                    for j in range(J):
                        rec[r["name"]][(kind, j)].append(se[j].cpu().numpy())
                i += J
        for r in models:
            R = rec[r["name"]]
            mse = {k: np.mean(np.stack(v), 0) for k, v in R.items()}
            j0, jl = mse[("true", 0)], np.mean([mse[("true", j)] for j in range(1, J)], 0)
            s0 = np.mean([mse[("shuf", j)] for j in range(J)], 0)
            t_all = np.mean([mse[("true", j)] for j in range(J)], 0)
            f = lambda v: f"{v.mean():.3f} ({' / '.join(f'{x:.2f}' for x in v)})"  # noqa: E731
            lines.append(f"  {split:<8} {r['name']}: true z all {f(t_all)} | pair 0 {f(j0)} | pairs 1-10 {f(jl)} | "
                         f"z shuffled {f(s0)} | shuf/true {s0.mean() / max(t_all.mean(), 1e-9):.2f}x")
        del cache
    held.clear()
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, f"{args.body}.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
