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


def features(tokens, mode, grid=16):
    """Reduce [T, grid*grid, dim] patch tokens to one vector per frame.

    Averaging every patch is linear, so it keeps whatever is spread across many patches and
    drowns whatever lives in few. Segment scale spans the whole body and survives it (F20);
    which feet are loaded occupies perhaps 6-12 patches of 256 and is diluted 20-40x. The mode
    therefore has to be chosen for the quantity being read, and stated, since a weak result
    under `mean` says as much about the reduction as about the encoder.

      mean   average over all patches. The global-property default.
      bands  average within each of four horizontal bands of the patch grid, concatenated.
             Feet and ground sit in the lower bands, so contact is diluted 4x instead of 256x
             while the feature count stays tractable at 4 x dim.
      max    per-dimension maximum over patches: the strongest local response anywhere in the
             frame, with position discarded.
    """
    if mode == "mean":
        return tokens.mean(1).numpy()
    if mode == "max":
        return tokens.max(1).values.numpy()
    if mode == "bands":
        t = tokens.reshape(len(tokens), grid, grid, -1)
        return t.reshape(len(tokens), 4, grid // 4 * grid, -1).mean(2).flatten(1).numpy()
    raise ValueError(f"unknown feature mode {mode!r}")


def load_side(encoder, chunk, mode, cache_path):
    """Features and stance fraction for both embodiments, as parallel arrays.

    Patch tokens are cached at full width and reduced afterwards, so trying another reduction
    costs no encoder time.
    """
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    fresh = False

    def tokens_for(path, frames):
        nonlocal fresh
        if path not in cache:
            cache[path] = encode_clip(encoder, frames, chunk)
            fresh = True
        return cache[path]

    out = {}
    feats, stance = [], []
    for body in INSECT_BODIES:
        for ep in INSECT_EPS:
            path = f"{ROOT}/data/ik_walk_8body/{body}_ep{ep}.npz"
            clip = np.load(path)
            feats.append(features(tokens_for(path, clip["frames"]), mode))
            stance.append((clip["forces"] > CONTACT_THRESHOLD).mean(axis=1))
    out["insect"] = (np.concatenate(feats), np.concatenate(stance))

    feats, stance = [], []
    for path in sorted(glob.glob(f"{ROOT}/data/b1_framed/*.npz")):
        clip = np.load(path, allow_pickle=True)
        feats.append(features(tokens_for(path, clip["frames"]), mode))
        stance.append(clip["foot_contact"].mean(axis=1))
    out["b1"] = (np.concatenate(feats), np.concatenate(stance))

    if fresh:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return out


def standardise_per_embodiment(data):
    """Centre and scale each embodiment's features by its own statistics.

    The two datasets differ in things that are not behaviour -- the insect renders orange and the
    B1 grey, the B1's apparent size grows several-fold across a clip while the insect's does not,
    the backgrounds differ. Those show up as a constant offset and a scale difference between the
    two clouds, and a readout fitted on one absorbs them, then mis-applies them to the other.

    This is standard unsupervised domain adaptation: it uses only *which dataset* a frame belongs
    to, never the stance fraction being predicted, so it does not leak the target. It is honest to
    report but it is not zero-shot -- it assumes a batch of the new embodiment's frames exists to
    compute statistics from, which is true when adapting to a robot and false when seeing one
    frame of it for the first time.
    """
    return {name: ((x - x.mean(0)) / (x.std(0) + 1e-6), y) for name, (x, y) in data.items()}


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
    ap.add_argument("--features", default="bands", choices=("mean", "bands", "max"))
    ap.add_argument("--normalize", action="store_true",
                    help="centre and scale each embodiment by its own statistics first")
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "wm", "cache",
                                                    "stage2_embeddings.pt"))
    args = ap.parse_args()

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    data = load_side(encoder, args.chunk, args.features, args.cache)
    del encoder
    print(f"features: {args.features}, {data['insect'][0].shape[1]} dimensions per frame"
          f"{', per-embodiment standardised' if args.normalize else ''}\n")
    if args.normalize:
        data = standardise_per_embodiment(data)

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
