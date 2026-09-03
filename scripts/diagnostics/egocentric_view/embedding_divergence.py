"""Does the counterfactual divergence survive the encoder the world model actually sees through?

    .venv/bin/python3 scripts/diagnostics/egocentric_view/embedding_divergence.py --branch 33 \\
        --noise data/allocentric/cf_confirm/insect_forward.npz data/allocentric/cf_confirm/insect_forward_repeat.npz \\
        --pair forward=... turn=... side=...

**The last de-risk gate, and the one every earlier number is upstream of.** `branch_divergence.py`
measures millimetres and degrees; **the world model is never shown millimetres**. F158 and F159
established that these embeddings suppress information that is physically present -- the action is
readable from a pose and contributes 3% of prediction -- so a counterfactual that is 13 degrees apart
in the world may be a rounding error in `e`.

**Same discipline as the physical measurement: displacement since the last shared frame.** The two
futures share a prefix, so `e_A[t] - e_A[ref]` against `e_B[t] - e_B[ref]` asks how differently they
*moved* after the split rather than how far apart they happen to sit, and it cancels whatever
offset the prefix had already accumulated.

**The noise floor is two runs of identical commands through the same encoder**, so encoder jitter,
render noise and gait-phase drift are all inside it. Turning is the case to watch: it is weakest
throughout this project (F136) and its physical divergence lives in heading, which a camera sees
only as a change of shape.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402


def embed(path, encoder, chunk, cache):
    if path not in cache:
        with np.load(os.path.join(ROOT, path), allow_pickle=True) as z:
            frames = np.asarray(z["frames"])
        cache[path] = encode_clip(encoder, frames, chunk).float().cpu()
    return cache[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", type=int, required=True)
    ap.add_argument("--noise", nargs=2, required=True, metavar=("RUN", "REPEAT"))
    ap.add_argument("--pair", nargs="+", required=True, metavar="NAME=CLIP")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15, 25])
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--floor", type=float, default=3.0)
    args = ap.parse_args()

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    cache = {}
    ref = max(args.branch - 1, 0)
    E = {p: embed(p, encoder, args.chunk, cache) for p in
         list(args.noise) + [s.split("=", 1)[1] for s in args.pair]}
    arms = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.pair]
    base_name, base_path = arms[0]

    def moved(path, t):
        e = E[path]
        return (e[t] - e[ref]).flatten()

    def gap(pa, pb, t):
        return float(torch.norm(moved(pa, t) - moved(pb, t)))

    scale = float(torch.norm(moved(base_path, min(len(E[base_path]) - 1, ref + args.horizons[-1]))))
    print(f"embedding divergence, reference = last shared frame {ref}, "
          f"arm '{base_name}'\n")
    print(f"  for scale: the reference arm itself moves {scale:.1f} in the embedding "
          f"over {args.horizons[-1]} steps\n")
    print(f"  {'arm':>10}{'h':>5}{'signal':>11}{'noise':>10}{'x':>8}"
          f"{'signal/own motion':>20}   verdict")
    failed = []
    for name, path in arms[1:]:
        for h in args.horizons:
            t = ref + h
            if t >= min(len(E[path]), len(E[base_path]), *(len(E[p]) for p in args.noise)):
                continue
            s = gap(base_path, path, t)
            n = gap(args.noise[0], args.noise[1], t)
            own = float(torch.norm(moved(base_path, t)))
            r = float("inf") if n == 0 else s / n
            ok = r >= args.floor
            if not ok:
                failed.append((name, h, round(r, 2)))
            print(f"  {name:>10}{h:>5}{s:>11.2f}{n:>10.2f}"
                  + ("  exact" if r == float("inf") else f"{r:>7.1f}x")
                  + f"{s / max(own, 1e-9):>20.2f}   "
                  + ("ok" if ok else f"**BELOW {args.floor}x**"))

    print()
    if failed:
        print(f"**{len(failed)} cells below {args.floor}x**: "
              + ", ".join(f"{n} h={h} ({r}x)" for n, h, r in failed))
        print("**The physical divergence does not reach the encoder at these horizons.** Two futures "
              "that\nthe world separates are, to the model, nearly the same future.")
    else:
        print(f"every arm clears {args.floor}x in the embedding at every horizon: the divergence "
              f"the\nworld model is shown is real, not only the divergence in the world")


if __name__ == "__main__":
    main()
