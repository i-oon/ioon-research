"""Counterfactual read-out on the v4 physics branches (doc/DATA_PLAN.md section 3): no kinematic posing, no simulator.

Each group = one (source clip, branch frame b) with 24 branches, one per command, simulated in physics from the
same state (B1: exact restore; hexapod: replay, a few mm apart -- F302 / DATA_PLAN). Per group, Pearson r across
the 24 branches between a read and the branch's true Froude (CoM labels from the loader, `segment`-aware), per
channel fwd / lat / yaw, then the mean over groups:

  real cf   Froude head(ITM(e_b, e_{b+K}))                  read of the real future of each command
  FTM cf    Froude head(ITM(e_b, FTM(e_b, proj(chunk))))     read of the predicted future (rollout, one step)
  direct    Froude head(proj(chunk))                         chunk = the branch's K commands at b + lag

Truth = mean label over [b, b+K) (the training target for the pair starting at b). With --pairs P > 1 every read
is averaged over the P consecutive pairs b .. b+P-1 and the truth over [b, b+P-1+K) (a windowed read; P <= 11
fits the 20 frames after the branch). Processed one group at a time (bounded RAM); tokens cached per file on disk (results/wm/cache/cf_tokens).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_readout.py --embodiment b1 \\
        --cf_dir data/counterfactual_walks/b1_branches_heldout --ckpt NAME=results/eval/NAME/ckpt/b1.pt --pairs 1 11
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at  # noqa: E402

CH = ("forward", "lateral", "yaw")


def groups_of(cf_dir):
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, cf_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            g[(str(d["cf_source"]), int(d["cf_t"]))].append(p)
    return g


CACHE = os.path.join(ROOT, "results/wm/cache/cf_tokens")


def tokens(path, clip, bp, n, encoder):
    """Frozen-encoder patch tokens of frames bp .. bp+n-1 of one branch, from a per-file disk cache.

    The encoder is frozen, so these are the same for every model: encoded once (float32 encoder, stored fp16 like every
    other embedding cache) and reused by every evaluation and diagnosis. One file per branch, loaded group by group, so RAM
    stays bounded. Stamped with the source file's size + mtime and the frame range; a mismatch re-encodes."""
    rel = os.path.relpath(path, ROOT).replace(os.sep, "__")
    f = os.path.join(CACHE, rel + ".pt")
    st = os.stat(path)
    stamp = (st.st_size, st.st_mtime_ns, bp, n)
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu")
        if tuple(d["stamp"]) == stamp:
            return d["e"]
    e = encode_clip(encoder(), np.asarray(clip["frames"][bp:bp + n]), 8).half().cpu()
    os.makedirs(CACHE, exist_ok=True)
    tmp = f + ".tmp"
    torch.save({"e": e, "stamp": stamp}, tmp)
    os.replace(tmp, f)
    return e


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", required=True, choices=("hexapod", "b1"))
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--pairs", type=int, nargs="+", default=[1, 11])
    ap.add_argument("--cache", default="", help="ignored (kept so old command lines still parse)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, emb_name = args.device, args.embodiment
    G = groups_of(args.cf_dir)
    if not G:
        raise SystemExit(f"no branch files in {args.cf_dir}")
    sizes = collections.Counter(len(v) for v in G.values())
    print(f"{len(G)} groups; branches per group: {dict(sizes)}")

    # **One group at a time, nothing kept.** Holding every branch's frames and patch-token embeddings
    # (1,728 files: ~10 GB frames + ~26 GB fp16 tokens) exhausted the 31 GB machine on 2026-10-03 and the OS
    # killed VS Code with it. Frames are read lazily, each group's 24 branches are encoded, read by every
    # checkpoint at every P, and dropped. Tokens are cached per branch file on disk (~11.5 MB each, ~20 GB per body).
    P_max = max(args.pairs)
    held = {}

    def encoder():                     # loaded only if some branch has no current cached tokens
        if "enc" not in held:
            from vjepa2_encoder import VJEPA2FrameEncoder
            held["enc"] = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
        return held["enc"]

    n_frames = None
    runs = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), emb_name, 18 if emb_name == "hexapod" else 12, dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        del ck
        off = None if m.offset is None else m.offset.float().to(dev)
        runs.append(dict(name=name, m=m, mean=mean, std=std, off=off,
                         R={P: {k: [] for k in ("real cf", "FTM cf", "direct")} for P in args.pairs}))
    K_max = max(r["m"].stride for r in runs)

    for gi, (key, paths) in enumerate(sorted(G.items())):
        clips = [load(p, REGISTRY[emb_name], lazy_frames=True) for p in paths]
        bs = [int(c["first_pair"]) for c in clips]
        # only the frames any read uses: b .. b + P_max - 1 + K_max
        n_frames = P_max + K_max
        E = [tokens(p, c, bp, n_frames, encoder) for p, c, bp in zip(paths, clips, bs)]
        for r in runs:
            m, K = r["m"], r["m"].stride
            rd = lambda z, r=r: r["m"].md.body(None, z).float().cpu().numpy() * r["std"] + r["mean"]  # noqa: E731
            fix = (lambda e: e) if r["off"] is None else (lambda e, o=r["off"]: e - o.reshape(e.shape[1:]))  # noqa: E731
            for P in args.pairs:
                truth, reads = [], {k: [] for k in r["R"][P]}
                for c, bp, e in zip(clips, bs, E):
                    e = e.float().to(dev)
                    if P - 1 + K >= len(e) or bp + P - 1 + K > len(c["body_motion"]):
                        raise SystemExit(f"--pairs {P} does not fit the frames after the branch")
                    truth.append(np.asarray(c["body_motion"])[bp:bp + P - 1 + K, :3].mean(0))
                    e0, e1 = fix(e[:P]), fix(e[K:K + P])
                    chunk = np.stack([action_chunk_at(c["actions"], bp + j + m.action_lag, K) for j in range(P)])
                    z = m.proj(torch.as_tensor(chunk, device=dev), emb_name)
                    reads["real cf"].append(rd(m.itm(e0, e1)).mean(0))
                    reads["FTM cf"].append(rd(m.itm(e0, m.ftm_step(e0, z))).mean(0))
                    reads["direct"].append(rd(z).mean(0))
                truth = np.stack(truth)
                for k in reads:
                    f = np.stack(reads[k])
                    r["R"][P][k].append([corr(f[:, j], truth[:, j]) for j in range(3)])
        del clips, E
        if (gi + 1) % 12 == 0:
            print(f"  {gi + 1}/{len(G)} groups", flush=True)
    held.clear()
    torch.cuda.empty_cache()

    for r in runs:
        for P in args.pairs:
            K = r["m"].stride
            print(f"\n=== {r['name']}: {len(G)} groups, pairs averaged {P} (truth over {P - 1 + K} frames); "
                  f"r fwd / lat / yaw", flush=True)
            for k, v in r["R"][P].items():
                print(f"  {k:<8}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)), flush=True)

if __name__ == "__main__":
    main()
