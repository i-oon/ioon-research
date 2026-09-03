"""Does the second frame matter more when one frame cannot fix the gait phase?

F31 measured that a single frame predicts the joint command at any horizon, and F29 that
deleting the transition costs the trained model only 1.11-1.19x. The explanation offered is
that six coordinated legs make the phase unambiguous: one leg at mid-stroke could be going
either way, but the other five say which half of the cycle it is in.

That explanation makes a prediction. Hide most of the legs and the ambiguity should come
back, and with it the value of the second frame. If the two-frame gain does not grow as the
view narrows, the explanation is wrong and something else is keeping the transition cheap.

The crop is taken relative to the robot, not the image: the camera is fixed and the robot
walks across the frame, so a fixed image crop would show a different part of the body at
every timestep and would measure that instead. The background is static within a clip, so
the per-pixel median over the clip is the empty scene, and thresholding against it gives the
robot's bounding box. Every variant including the full view goes through the same
crop-and-resize, so the only thing that changes is how much of the body is left.

  .venv/bin/python3 scripts/occlusion_dynamics.py --clips 8
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402


def robot_boxes(frames, threshold=12):
    """Per-frame bounding box of whatever differs from the empty scene."""
    background = np.median(frames, axis=0)
    boxes = []
    for frame in frames:
        mask = np.abs(frame.astype(np.int16) - background).max(axis=2) > threshold
        cols, rows = np.where(mask.any(axis=0))[0], np.where(mask.any(axis=1))[0]
        if len(cols) == 0 or len(rows) == 0:          # nothing found: keep the whole frame
            boxes.append((0, frame.shape[1] - 1, 0, frame.shape[0] - 1))
        else:
            boxes.append((cols[0], cols[-1], rows[0], rows[-1]))
    return boxes


def crop_clip(frames, boxes, keep, size=256, pad=6):
    """Keep the middle `keep` fraction of the body's width, full height, resized back."""
    out = []
    for frame, (x0, x1, y0, y1) in zip(frames, boxes):
        centre = (x0 + x1) / 2
        half = max(4.0, (x1 - x0) * keep / 2)
        left, right = int(round(centre - half)), int(round(centre + half))
        top, bottom = max(0, y0 - pad), min(frame.shape[0], y1 + pad + 1)
        left, right = max(0, left), min(frame.shape[1], right + 1)
        patch = frame[top:bottom, left:right]
        out.append(cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR))
    return np.stack(out)


def fit_ridge(features_train, target_train, features_test, target_test):
    model = Ridge(alpha=1.0).fit(features_train, target_train)
    return float(np.sqrt(((model.predict(features_test) - target_test) ** 2).mean()))


def fit_logistic(features_train, labels_train, features_test, labels_test):
    scaler = StandardScaler().fit(features_train)
    scores = []
    for column in range(labels_train.shape[1]):
        if len(set(labels_train[:, column])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(features_train),
                                                    labels_train[:, column])
        scores.append(clf.score(scaler.transform(features_test), labels_test[:, column]))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", default="c10f10t10")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--train_clips", type=int, default=6)
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5, 0.33, 0.2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--preview", default=os.path.join(ROOT, "results", "wm", "stage2",
                                                      "occlusion_views.png"))
    args = ap.parse_args()

    paths = clip_paths(os.path.join(ROOT, "data", "ik_walk_8body"), (args.morph,))[:args.clips]
    clips = [load_clip(p) for p in paths]
    boxes = [robot_boxes(c["frames"]) for c in clips]

    strip = [crop_clip(clips[0]["frames"][30:31], boxes[0][30:31], k)[0] for k in args.keeps]
    os.makedirs(os.path.dirname(args.preview), exist_ok=True)
    cv2.imwrite(args.preview, cv2.cvtColor(np.concatenate(strip, axis=1), cv2.COLOR_RGB2BGR))
    print(f"what the encoder sees at each setting -> {args.preview}\n")

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    n_train = args.train_clips
    print(f"body '{args.morph}', {len(paths)} clips, fit on {n_train}, test on "
          f"{len(paths) - n_train}")
    steps = np.concatenate([np.diff(np.degrees(c["actions"]), axis=0) for c in clips])
    print(f"the change a_t+1 - a_t has std {steps.std():.2f} deg\n")

    header = (f'{"body visible":<14} {"a_t+1: 1 frame":>15} {"2 frames":>10} {"gain":>7} '
              f'{"change: 1 frame":>16} {"2 frames":>10} {"gain":>7} {"swing: 1":>9} {"2":>7}')
    print(header)
    for keep in args.keeps:
        embeddings = []
        for clip, box in zip(clips, boxes):
            cropped = crop_clip(clip["frames"], box, keep)
            embeddings.append(encode_clip(encoder, cropped, args.chunk).mean(1).numpy())

        def split(build_features, build_target):
            parts = ([], []), ([], [])
            for i, (e, clip) in enumerate(zip(embeddings, clips)):
                x = build_features(e)
                y = build_target(np.degrees(clip["actions"]), clip["forces"])
                n = min(len(x), len(y))
                dest = parts[0] if i < n_train else parts[1]
                dest[0].append(x[:n]); dest[1].append(y[:n])
            return [[np.concatenate(v) for v in half] for half in parts]

        one = lambda e: e[:-1]
        two = lambda e: np.concatenate([e[:-1], e[1:]], axis=1)
        absolute = lambda a, f: a[1:]
        change = lambda a, f: np.diff(a, axis=0)
        swing = lambda a, f: (f[1:] <= 0.5).astype(int)

        row = []
        for target in (absolute, change):
            scores = []
            for features in (one, two):
                (xtr, ytr), (xte, yte) = split(features, target)
                scores.append(fit_ridge(xtr, ytr, xte, yte))
            row += [scores[0], scores[1], scores[0] / scores[1]]
        accuracies = []
        for features in (one, two):
            (xtr, ytr), (xte, yte) = split(features, swing)
            accuracies.append(fit_logistic(xtr, ytr, xte, yte))
        print(f'{keep:>12.0%}   {row[0]:15.2f} {row[1]:10.2f} {row[2]:6.2f}x '
              f'{row[3]:16.2f} {row[4]:10.2f} {row[5]:6.2f}x '
              f'{accuracies[0]:9.3f} {accuracies[1]:7.3f}')

    print("\nIf hiding the legs is what restores the value of the transition, both gain columns "
          "rise\nas the visible fraction falls. If they stay flat, the explanation in F31 is "
          "wrong.")


if __name__ == "__main__":
    main()
