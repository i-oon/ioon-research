"""Is body speed readable from one frame because of physics, or because we built the data badly?

The frozen encoder reads forward Froude from a single frame at R^2 0.676, and that number is why the
shared head has to be blind (F64). But it mixes two things that call for opposite responses:

    between-clip   the frame reveals *which clip* this is, and every clip has one constant speed,
                   so recognising the condition is enough -- **an artefact of how we collected**
    within-clip    the frame reveals the robot's instantaneous speed from its posture, stride
                   length and body pitch -- **real physics, not removable**

If the readability is nearly all between-clip, the fix is data: let the speed change inside a clip at
unpredictable moments, so that recognising the clip stops answering the question. If a large part is
within-clip, no amount of collection removes it, blinding the head is the only route, and that is a
genuine domain difference between locomotion and manipulation worth writing down.

The decomposition is the standard one. Each frame's target is split into its clip's mean and the
deviation from it, and a readout is fitted to each part separately from the same features. The
variance shares are reported alongside, because a part that holds 2 percent of the variance cannot
matter however well it is predicted.

  .venv/bin/python3 scripts/diagnostics/speed_variance_split.py --device cuda
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from wm.data.embodiment import body_motion  # noqa: E402

from diagnostics.body_motion_probe import DT, bands, insect_paths  # noqa: E402


def fit(x, y, clip, seed):
    """Ridge from frame features to `y`, held out by clip."""
    if y.std() < 1e-9:
        return float("nan")
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-9)
    rng = np.random.default_rng(seed)
    ids = np.unique(clip)
    rng.shuffle(ids)
    train = np.isin(clip, ids[:int(0.7 * len(ids))])
    if y[~train].std() < 1e-9:
        return float("nan")
    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x[train], y[train])
    return float(r2_score(y[~train], model.predict(x[~train])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--insect_dir", default="data/fwd_hex7speed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trim", type=int, default=0,
                    help="drop this many frames from each end before measuring. **Not cosmetic.** "
                         "`body_motion` smooths with `np.convolve(mode='same')`, which does not "
                         "renormalise at the edges, so the first and last half-window of every "
                         "clip is biased toward zero -- 28 percent of a 72-frame clip at the "
                         "one-second window. The robot also walks across the frame, so its "
                         "position tells a probe how near the edge it is, which makes the "
                         "artefact easy to predict and inflates the within-clip score. Pass "
                         "half the window (10) to measure what is left.")
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, "results", "wm", "cache",
                              f"probe_{os.path.basename(args.insect_dir)}.pt")
    cache = torch.load(cache_path, map_location="cpu")

    groups = {"insect": insect_paths(os.path.join(ROOT, args.insect_dir)),
              "b1": sorted(glob.glob(f"{ROOT}/data/fwd_b1_50hz/*.npz"))}

    print("Forward Froude read from ONE frame, split into the part that varies between clips")
    print("and the part that varies inside a clip.\n")
    print(f"{'set':<22}{'clips':>7}{'var between':>13}{'R2 between':>12}{'R2 within':>11}")

    for name, paths in groups.items():
        rows = {"all": []}
        for i, path in enumerate(paths):
            if path not in cache:
                continue
            with np.load(path, allow_pickle=True) as clip:
                position = clip["head"] if "head" in clip.files else clip["base_pos"]
                # the insect set records the speed condition it was retimed to; a ramp has a
                # different start and end, and its within-clip variation is real rather than gait
                # oscillation, so the two kinds are reported apart
                s = float(clip["speed"]) if "speed" in clip.files else 0.0
                e = float(clip["speed_end"]) if "speed_end" in clip.files else s
            y = body_motion(position.astype(np.float64), DT[name])[:, 0]
            x = bands(cache[path].float())[:len(y)]
            y = y[:len(x)]
            if args.trim:
                if len(y) <= 2 * args.trim + 4:
                    continue
                x, y = x[args.trim:-args.trim], y[args.trim:-args.trim]
            item = (x, y, i)
            rows["all"].append(item)
            if "speed" in np.load(path, allow_pickle=True).files:
                rows.setdefault("constant speed" if s == e else "ramped", []).append(item)

        for label, items in rows.items():
            if not items:
                continue
            x = np.concatenate([a for a, _, _ in items])
            y = np.concatenate([b for _, b, _ in items])
            c = np.concatenate([np.full(len(b), i) for _, b, i in items])
            means = np.array([y[c == i].mean() for i in c])
            between, within = means, y - means
            share = between.var() / (y.var() + 1e-12)
            tag = f"{name} — {label}" if label != "all" else f"{name} — all"
            print(f"{tag:<22}{len(items):>7}{share:>12.1%}"
                  f"{fit(x, between, c, args.seed):>12.3f}{fit(x, within, c, args.seed):>11.3f}")

    print("\nRead `var between` first: it is the ceiling on how much the within part could ever")
    print("matter. High share with a high between-R2 and a low within-R2 means the encoder is")
    print("recognising the condition, not seeing the motion -- which is a data-collection fix.")


if __name__ == "__main__":
    main()
