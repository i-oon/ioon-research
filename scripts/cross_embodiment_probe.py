"""Does the frozen encoder place a hexapod and a quadruped in a shared space?

Stage 2 assumes it does. Nothing has tested it. This tests it without training anything, by
asking whether a readout fitted on one embodiment recovers a physical quantity from the other.

The quantity has to exist for both, so it cannot be a joint command (18 numbers against 12) or a
segment scale (no correspondence). It also cannot be walking speed: within a body the insect's
speed barely varies -- 0.019 m of standard deviation on a 0.573 m mean -- so a speed probe fitted
on insect data would learn "smaller body, slower", which is an appearance cue and would mean
nothing applied to a B1.

**Stance fraction** works: the proportion of feet on the ground, in [0, 1] whether there are six
feet or four, recorded on both sides, and varying richly within every clip as the gait cycles.

Four conditions, so the cross scores can be read against their own ceilings:

    insect -> insect     the ceiling for this quantity on this embodiment
    b1     -> b1         the same
    insect -> b1         does a readout fitted on one transfer to the other
    b1     -> insect     the same question from the other side

Reported as RMSE in stance fraction, against the standard deviation of the target itself, so a
score at or above that spread means the probe has learned nothing usable.

  .venv/bin/python3 scripts/cross_embodiment_probe.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.dataset import CONTACT_THRESHOLD  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

INSECT_BODIES = ["c10f10t10", "c06f10t10", "c10f10t06", "c10f06t06", "c08f09t09"]
INSECT_EPS = [6, 20, 22]


def load_side(encoder, chunk):
    """Embeddings and stance fraction for both embodiments, as parallel arrays."""
    out = {}

    frames, stance = [], []
    for body in INSECT_BODIES:
        for ep in INSECT_EPS:
            clip = np.load(f"{ROOT}/data/ik_walk_8body/{body}_ep{ep}.npz")
            frames.append(encode_clip(encoder, clip["frames"], chunk).mean(1).numpy())
            stance.append((clip["forces"] > CONTACT_THRESHOLD).mean(axis=1))
    out["insect"] = (np.concatenate(frames), np.concatenate(stance))

    frames, stance = [], []
    for path in sorted(glob.glob(f"{ROOT}/data/b1_framed/*.npz")):
        clip = np.load(path, allow_pickle=True)
        frames.append(encode_clip(encoder, clip["frames"], chunk).mean(1).numpy())
        stance.append(clip["foot_contact"].mean(axis=1))
    out["b1"] = (np.concatenate(frames), np.concatenate(stance))
    return out


def halves(x, y, seed=0):
    """Split by clip-length blocks rather than at random, so neighbouring frames of one clip do
    not end up on both sides of the split and inflate the score."""
    n = len(x)
    block = 66
    blocks = np.arange(n) // block
    rng = np.random.default_rng(seed)
    keep = rng.permutation(np.unique(blocks))
    cut = int(0.7 * len(keep))
    tr = np.isin(blocks, keep[:cut])
    return x[tr], y[tr], x[~tr], y[~tr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    data = load_side(encoder, args.chunk)
    del encoder

    for name, (x, y) in data.items():
        print(f"{name:<8} {len(x)} frames, stance fraction mean {y.mean():.3f} "
              f"spread {y.std():.3f}")
    print()

    print(f'{"fitted on":<10}{"tested on":<12}{"RMSE":>8}{"target spread":>15}{"vs spread":>11}')
    for src in ("insect", "b1"):
        xs, ys = data[src]
        xtr, ytr, xte, yte = halves(xs, ys)
        model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xtr, ytr)
        for dst in ("insect", "b1"):
            xd, yd = data[dst]
            # held-out half when testing on itself, everything when testing on the other side
            xq, yq = (xte, yte) if dst == src else (xd, yd)
            err = float(np.sqrt(((model.predict(xq) - yq) ** 2).mean()))
            print(f'{src:<10}{dst:<12}{err:8.3f}{yq.std():15.3f}{err / yq.std():10.2f}x')

    # how separable are the two embodiments at all
    xall = np.concatenate([data["insect"][0], data["b1"][0]])
    lab = np.concatenate([np.zeros(len(data["insect"][0])), np.ones(len(data["b1"][0]))])
    acc = cross_val_score(LogisticRegression(max_iter=3000), xall, lab, cv=5).mean()
    mi, mb = data["insect"][0].mean(0), data["b1"][0].mean(0)
    between = np.linalg.norm(mi - mb)
    within = 0.5 * (np.linalg.norm(data["insect"][0] - mi, axis=1).mean()
                    + np.linalg.norm(data["b1"][0] - mb, axis=1).mean())
    print(f"\nembodiment separable at {acc:.3f} (chance 0.5); distance between the two clusters "
          f"is {between / within:.2f}x the average spread within one")
    print("\nA cross row near its own diagonal means the encoder represents contact the same way "
          "for both.\nA cross row at or above 1.00x of the target spread means it does not.")


if __name__ == "__main__":
    main()
