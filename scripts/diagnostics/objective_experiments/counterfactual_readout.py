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
fits the 20 frames after the branch). Embeddings are cached per file.

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
from wm.data.emb_cache import load_cache, note, save_cache  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at  # noqa: E402

CH = ("forward", "lateral", "yaw")


def groups_of(cf_dir):
    g = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, cf_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            g[(str(d["cf_source"]), int(d["cf_t"]))].append(p)
    return g


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", required=True, choices=("hexapod", "b1"))
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--pairs", type=int, nargs="+", default=[1, 11])
    ap.add_argument("--cache", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, emb_name = args.device, args.embodiment
    G = groups_of(args.cf_dir)
    if not G:
        raise SystemExit(f"no branch files in {args.cf_dir}")
    sizes = collections.Counter(len(v) for v in G.values())
    print(f"{len(G)} groups; branches per group: {dict(sizes)}")

    files = sorted(p for v in G.values() for p in v)
    clips = {p: load(p, REGISTRY[emb_name]) for p in files}
    b = {p: int(clips[p]["first_pair"]) for p in files}
    K = None
    cache = os.path.join(ROOT, args.cache or f"results/wm/cache/cf_readout_v4_{emb_name}_"
                         f"{os.path.basename(os.path.normpath(args.cf_dir))}.pt")
    E = load_cache(cache)
    todo = [p for p in files if p not in E]
    if todo:
        from vjepa2_encoder import VJEPA2FrameEncoder
        enc = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
        for i, p in enumerate(todo):
            fr = clips[p]["frames"][b[p]:]                 # from the branch frame to the end
            E[p] = encode_clip(enc, fr, 8).cpu().half()
            note(E, p)
            if (i + 1) % 200 == 0:
                print(f"  encoded {i + 1}/{len(todo)}", flush=True)
                save_cache(E, cache)
        save_cache(E, cache)
        del enc
        torch.cuda.empty_cache()

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), emb_name, 18 if emb_name == "hexapod" else 12, dev)
        K = m.stride
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        rd = lambda z: m.md.body(None, z).float().cpu().numpy() * std + mean  # noqa: E731
        off = None if m.offset is None else m.offset.float().to(dev)
        fix = (lambda e: e) if off is None else (lambda e: e - off.reshape(e.shape[1:]))  # noqa: E731
        for P in args.pairs:
            R = {k: [] for k in ("real cf", "FTM cf", "direct")}
            for key, paths in sorted(G.items()):
                truth, reads = [], {k: [] for k in R}
                for p in paths:
                    c, e = clips[p], E[p].float().to(dev)
                    if P - 1 + K >= len(e):
                        raise SystemExit(f"--pairs {P} does not fit {len(e)} frames after the branch")
                    bp = b[p]
                    truth.append(np.asarray(c["body_motion"])[bp:bp + P - 1 + K, :3].mean(0))
                    e0, e1 = fix(e[:P]), fix(e[K:K + P])
                    chunk = np.stack([action_chunk_at(c["actions"], bp + j + m.action_lag, K) for j in range(P)])
                    z = m.proj(torch.as_tensor(chunk, device=dev), emb_name)
                    reads["real cf"].append(rd(m.itm(e0, e1)).mean(0))
                    reads["FTM cf"].append(rd(m.itm(e0, m.ftm_step(e0, z))).mean(0))
                    reads["direct"].append(rd(z).mean(0))
                truth = np.stack(truth)
                for k in R:
                    f = np.stack(reads[k])
                    R[k].append([corr(f[:, j], truth[:, j]) for j in range(3)])
            print(f"\n=== {name}: {len(G)} groups, pairs averaged {P} (truth over {P - 1 + K} frames); "
                  f"r fwd / lat / yaw", flush=True)
            for k, v in R.items():
                print(f"  {k:<8}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)), flush=True)


if __name__ == "__main__":
    main()
