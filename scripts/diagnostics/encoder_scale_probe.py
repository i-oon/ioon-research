"""Can a linear readout of the frozen encoder recover a body's segment scales, and does its error
predict whether a training split will transfer?

Two numbers, both cheap, both needing no trained model:

  **mixture gap** -- pure geometry. The distance from the held-out body's (coxa, femur, tibia) to
  the nearest convex mixture of the training bodies', by non-negative least squares. Zero means
  the held-out body sits inside the hull the training set spans. No encoder, no data, no GPU.

  **probe error** -- fit ridge from mean-pooled frozen-encoder features to the three scales on the
  training bodies, apply to the held-out one, report per-scale absolute error. 1,408 features to
  3 outputs is 4,227 parameters against a 1B-parameter encoder that never saw a robot.

Slide 10's claim is that these predict the outcome of a run before it is trained: a split whose
held-out body cannot be mixed from its training bodies fails, and the probe error separates the
two cases by a wide margin. Running it costs minutes on CPU; finding out by training costs hours
of GPU per body.

**The trap.** Fit the mixture with `scipy.optimize.nnls`, not a hand-rolled projected gradient --
that returned a solution 39x worse than optimal once and briefly inverted a conclusion. And read
the probe error against the *mixture gap*, not on its own: the probe recovering a body well is
only interesting when that body is outside what a mixture could already reach.

  .venv/bin/python3 scripts/diagnostics/encoder_scale_probe.py \\
      --data_dir data/fwd_cov_narrow \\
      --train c10f10t10 c06f10t10 c10f06t06 c08f09t09 --test c10f10t08
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.dataset import load_clip  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402


def scales_of(body):
    """`c08f09t09` -> (0.8, 0.9, 0.9). The body name is the single source of its geometry."""
    if len(body) != 9 or body[0] != "c" or body[3] != "f" or body[6] != "t":
        raise SystemExit(f"body name {body!r} is not in the cXXfYYtZZ form")
    return np.array([int(body[1:3]), int(body[4:6]), int(body[7:9])], dtype=float) / 10.0


def mixture_gap(train_bodies, test_body):
    """Distance from the test body's scales to the nearest convex mixture of the training ones.

    Non-negative least squares with the weights driven to sum to one by an appended row, which is
    the standard way to get a convex fit out of nnls without writing an optimiser.
    """
    A = np.stack([scales_of(b) for b in train_bodies], axis=1)
    b = scales_of(test_body)
    penalty = 100.0                                   # forces the weights to sum to one
    A_aug = np.vstack([A, penalty * np.ones((1, A.shape[1]))])
    b_aug = np.concatenate([b, [penalty]])
    weights, _ = nnls(A_aug, b_aug)
    return float(np.linalg.norm(A @ weights - b)), weights


def features(encoder, data_dir, body, clips, chunk):
    """Mean-pooled patch tokens, one row per frame, for the first `clips` clips of a body."""
    paths = sorted(glob.glob(os.path.join(data_dir, f"{body}_ep*.npz")))[:clips]
    if not paths:
        raise SystemExit(f"no clips for {body} in {data_dir}")
    rows = []
    for path in paths:
        clip = load_clip(path)
        emb = encode_clip(encoder, clip["frames"], chunk)
        rows.append(emb.mean(dim=1).float().cpu().numpy())   # pool over patch tokens
    return np.concatenate(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test_dir", default="",
                    help="directory holding the test body, when it is not stored with the "
                         "training bodies -- the decoupled-ratio bodies live in their own "
                         "directory, so a split can straddle two")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)
    gap, weights = mixture_gap(args.train, args.test)
    print(f"held out {args.test}  scales {scales_of(args.test)}")
    print(f"training set {args.train}")
    print(f"  nearest mixture {' '.join(f'{w:.2f}' for w in weights)}")
    print(f"  **mixture gap {gap:.3f}**  (0 = inside the hull the training bodies span)")

    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    per_body = {b: features(encoder, data_dir, b, args.clips, args.chunk) for b in args.train}
    X = np.concatenate([per_body[b] for b in args.train])
    y = np.concatenate([np.repeat(scales_of(b)[None], len(per_body[b]), 0) for b in args.train])
    test_dir = args.test_dir or args.data_dir
    test_dir = test_dir if os.path.isabs(test_dir) else os.path.join(ROOT, test_dir)
    X_test = features(encoder, test_dir, args.test, args.clips, args.chunk)
    del encoder

    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X, y)

    # The training bodies are reported too, in-sample by construction. They are the check that the
    # readout is well conditioned rather than lucky on one body: a probe that cannot recover the
    # bodies it was fitted on says nothing about the one it was not.
    print(f"\n  {'body':<12}{'':<10}{'coxa':>16}{'femur':>16}{'tibia':>16}")
    for body in args.train:
        pred_b = model.predict(per_body[body]).mean(0)
        true_b = scales_of(body)
        cells = "".join(f"{p:>9.3f} / {t:.2f}" for p, t in zip(pred_b, true_b))
        print(f"  {body:<12}{'train':<10}{cells}")

    pred = model.predict(X_test).mean(0)
    truth = scales_of(args.test)
    cells = "".join(f"{p:>9.3f} / {t:.2f}" for p, t in zip(pred, truth))
    print(f"  {args.test:<12}{'HELD OUT':<10}{cells}")
    print(f"\n  held-out error per scale: "
          f"{'  '.join(f'{abs(p - t):.3f}' for p, t in zip(pred, truth))}")
    print(f"  **probe error {np.abs(pred - truth).mean():.3f}**  "
          f"({X.shape[1]} features to 3 outputs, {X.shape[1] * 3 + 3} parameters)")


if __name__ == "__main__":
    main()
