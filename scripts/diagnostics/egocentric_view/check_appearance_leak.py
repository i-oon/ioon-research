"""Can heading be read off the room's colour? If so, Q1 measures a leak, not egocentric.

    .venv/bin/python3 scripts/diagnostics/egocentric_view/check_appearance_leak.py \\
        --data data/allocentric/ego_derisk/insect_flat --embodiment hexapod

**A guard, run before Q1 is trusted, not after it is reported.** Egocentric is supposed to break
single-frame pose readability. A room whose walls are permanently coloured hands it straight back --
"see red, facing north" is a landmark, and a landmark is a pose label. Appearance is randomised per
episode to prevent that, **and randomisation that is too coarse does not prevent it**: with few
enough seeds, colour still correlates with heading across the dataset and a probe will find it.

**This measures the leak directly**, on the frames themselves rather than on the intention behind
them. Per frame it takes a cheap colour summary -- mean RGB of the upper half of the image, where
walls are, plus a small hue histogram -- and asks whether that predicts heading on **held-out
clips**. Chance is what it must reach.

**A failure here is fixable and cheap**: add seeds. What is not fixable is discovering afterwards
that Q1's number was a landmark all along.
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import heading  # noqa: E402

QUAT = {"hexapod": "body_quat", "b1": "base_quat"}


def colour_features(frames):
    """Mean RGB of the upper half plus an 8-bin hue histogram -- **appearance only, no geometry.**

    Deliberately weak: if a summary this crude predicts heading, the leak is gross. Using the whole
    frame would fold in the floor and the robot's own limbs, which are not the landmark at issue.
    """
    top = frames[:, : frames.shape[1] // 2].astype(np.float32) / 255.0
    mean = top.reshape(len(top), -1, 3).mean(1)
    mx, mn = top.max(-1), top.min(-1)
    hue = np.where(mx > mn, (top.argmax(-1) + (top.max(-1) - top.min(-1))) % 3.0, 0.0)
    hist = np.stack([((hue >= b / 8.0 * 3) & (hue < (b + 1) / 8.0 * 3)).reshape(len(top), -1)
                     .mean(1) for b in range(8)], 1)
    return np.concatenate([mean, hist], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", choices=("hexapod", "b1"), default="hexapod")
    ap.add_argument("--bins", type=int, default=8, help="heading buckets; chance is 1/bins")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--margin", type=float, default=1.25,
                    help="how far above chance counts as a leak")
    args = ap.parse_args()

    X, Y, C = [], [], []
    for ci, path in enumerate(sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))):
        with np.load(path, allow_pickle=True) as z:
            f = np.asarray(z["frames"])[::args.stride]
            q = np.asarray(z[QUAT[args.embodiment]], float)[::args.stride]
        n = min(len(f), len(q))
        X.append(colour_features(f[:n]))
        Y.append(heading(q[:n], args.embodiment))
        C.append(np.full(n, ci))
    X = np.concatenate(X); Y = np.concatenate(Y); C = np.concatenate(C)
    b = ((Y + np.pi) / (2 * np.pi) * args.bins).astype(int) % args.bins

    clips = sorted(set(C.tolist()))
    test = set(clips[1::2])
    te = np.array([c in test for c in C]); tr = ~te
    if not te.any() or not tr.any():
        raise SystemExit("need at least two clips")

    # nearest-centroid on the colour summary: no capacity to memorise, so what it finds is real
    cents = {}
    for k in range(args.bins):
        m = tr & (b == k)
        if m.any():
            cents[k] = X[m].mean(0)
    keys = sorted(cents)
    M = np.stack([cents[k] for k in keys])
    pred = np.array([keys[int(np.argmin(((M - x) ** 2).sum(1)))] for x in X[te]])
    acc = float((pred == b[te]).mean())
    chance = float(max(collections.Counter(b[te].tolist()).values()) / te.sum())

    print(f"{args.data}\n{len(clips)} clips, {te.sum()} held-out frames, "
          f"{args.bins} heading bins\n")
    print(f"  heading from room colour, held-out clips : {acc:.3f}")
    print(f"  chance (largest bin)                     : {chance:.3f}")
    print(f"  ratio                                    : {acc / max(chance, 1e-9):.2f}x\n")
    if acc >= chance * args.margin:
        print(f"**LEAK: colour predicts heading at {acc / chance:.2f}x chance.** The room is acting "
              f"as a\nlandmark, so Q1 would be measuring that rather than the egocentric view. "
              f"**Add seeds\nand re-run.** Do not read Q1 until this is at chance.")
        raise SystemExit(1)
    print("no leak: room colour does not predict heading above chance, so Q1 is measuring the view")


if __name__ == "__main__":
    main()
