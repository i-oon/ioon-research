"""Is the B1's joint command as determined by one frame as the insect's is?

F31 measured that for the insect, a single frame predicts the joint command at every
horizon tested, because the data is one gait at one speed and the phase fixes everything
after it. That is why the forward transition model has nothing to do.

The B1 data is not one speed: fourteen clips, two gait policies, seven commanded forward
speeds from 0.30 to 0.50 m/s. If speed is not readable from a still frame, then one frame
should NOT determine the next command, the second frame should earn more than the 1.09x it
earns on the insect, and the forward model has a job in Stage 2 worth keeping.

Everything is measured the same way as the insect so the two are comparable: ridge from a
single mean-pooled frame embedding, RMSE in degrees, reported against the commands' own
spread.

  .venv/bin/python3 scripts/b1_horizon.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.evaluate import encode_clip  # noqa: E402


def rmse(model, features, target):
    return float(np.sqrt(((model.predict(features) - target) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "b1_framed"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no clips in {args.data_dir}")
    # hold out every third clip, so both policies and the whole speed range appear on both
    # sides of the split
    test_index = set(range(2, len(paths), 3))

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    embeddings, actions, speeds = [], [], []
    for path in paths:
        clip = np.load(path, allow_pickle=True)
        embeddings.append(encode_clip(encoder, clip["frames"], args.chunk).mean(1).numpy())
        actions.append(np.degrees(clip["action"]))
        speeds.append(float(clip["command"][0, 0]))

    pooled = np.concatenate(actions)
    steps = np.concatenate([np.diff(a, axis=0) for a in actions])
    print(f"{len(paths)} clips, {len(paths) - len(test_index)} fitted / {len(test_index)} held "
          f"out, commanded speeds {min(speeds):.2f}-{max(speeds):.2f} m/s")
    print(f"command spread {pooled.std(axis=0).mean():.2f} deg per joint | "
          f"step a_t+1 - a_t: std {steps.std():.2f} deg\n")

    def split(build_features, build_target):
        halves = ([], []), ([], [])
        for i, (e, a) in enumerate(zip(embeddings, actions)):
            x, y = build_features(e), build_target(a)
            n = min(len(x), len(y))
            dest = halves[1] if i in test_index else halves[0]
            dest[0].append(x[:n]); dest[1].append(y[:n])
        return [[np.concatenate(v) for v in half] for half in halves]

    print(f'{"predict from one frame":<26} {"RMSE deg":>9}')
    for lag in args.horizons:
        (xtr, ytr), (xte, yte) = split(lambda e: e[:len(e) - lag] if lag else e,
                                       lambda a: a[lag:])
        model = Ridge(alpha=1.0).fit(xtr, ytr)
        print(f'{"a_t" if lag == 0 else f"a_t+{lag}":<26} {rmse(model, xte, yte):9.2f}')

    one = lambda e: e[:-1]
    two = lambda e: np.concatenate([e[:-1], e[1:]], axis=1)
    print(f'\n{"predict":<26} {"1 frame":>9} {"2 frames":>10} {"gain":>7}')
    for name, build in (("a_t+1, whole command", lambda a: a[1:]),
                        ("a_t+1 - a_t, the change", lambda a: np.diff(a, axis=0))):
        scores = []
        for features in (one, two):
            (xtr, ytr), (xte, yte) = split(features, build)
            scores.append(rmse(Ridge(alpha=1.0).fit(xtr, ytr), xte, yte))
        print(f'{name:<26} {scores[0]:9.2f} {scores[1]:10.2f} {scores[0]/scores[1]:6.2f}x')

    # can a single frame even tell how fast the robot was told to walk?
    (xtr, ytr), (xte, yte) = split(lambda e: e, lambda a: a[:, :1] * 0)
    per_clip_x = np.stack([e.mean(0) for i, e in enumerate(embeddings) if i not in test_index])
    per_clip_y = np.array([s for i, s in enumerate(speeds) if i not in test_index])
    held_x = np.stack([e.mean(0) for i, e in enumerate(embeddings) if i in test_index])
    held_y = np.array([s for i, s in enumerate(speeds) if i in test_index])
    speed_model = Ridge(alpha=1.0).fit(per_clip_x, per_clip_y)
    error = float(np.sqrt(((speed_model.predict(held_x) - held_y) ** 2).mean()))
    print(f'\ncommanded speed recovered from a clip of frames: {error:.3f} m/s error, '
          f'against a range of {max(speeds) - min(speeds):.2f} m/s')
    print("\nCompare with the insect (F31): one frame gave 4.61 deg on a_t against an 11.33 deg\n"
          "spread, a_t+32 gave 4.45, and the second frame was worth 1.09x on the change.")


if __name__ == "__main__":
    main()
