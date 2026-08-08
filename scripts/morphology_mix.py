"""What mixture of the training bodies does the model's answer look like?

The one-dimensional version of this question (scripts/morphology_axis.py) asks where on the
line between two training bodies a prediction lands. With several training bodies spanning a
volume of segment scales there is no single line, so the same question becomes: which convex
mixture of the training bodies' joint commands best reproduces the prediction?

The mixture separates two behaviours that a single error number cannot. A model that has
learned to read morphology off the frame should produce a mixture close to the one that
actually reconstructs the held-out body. A model that has memorised the training bodies and
picks the nearest should put almost all its weight on one of them. Averaged segment scales make
this readable in metres rather than weights: the mixture implies a body, and that body can be
compared against the one the model was actually shown.

Weights are non-negative and sum to one, fitted by projected gradient over the whole command
sequence rather than its mean: the mean alone is 18 numbers and many different mixtures
reproduce it, so fitting the mean gives weights that do not reproduce the trajectory. The fit
uses ground-truth commands from the training bodies only; the held-out body's commands are
never used except to report the target.

Run from the repository root:
  .venv/bin/python3 scripts/morphology_mix.py --pred results/wm/predictions/<name>.npz
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOINT_TYPES = ("TC", "CF", "FT")


def scales_from_name(body):
    """c08f09t09 -> (0.8, 0.9, 0.9)."""
    return tuple(int(body[i + 1:i + 3]) / 10 for i in (0, 3, 6))


def mean_actions(data_dir, body, episodes, frames):
    paths = [os.path.join(data_dir, f"{body}_ep{e}.npz") for e in episodes]
    return np.concatenate([np.degrees(np.load(p)["actions"][:frames]) for p in paths])


def fit_mixture(target, sources):
    """Non-negative weights summing to one that minimise ||sum(w_i * source_i) - target||.

    Solved exactly with non-negative least squares. The sum-to-one constraint is imposed by
    appending a heavily weighted row of ones, which is standard and avoids the projected
    gradient this used to run: with weights bounded in [0, 1] that iteration oscillated and
    returned a solution 39x worse than the optimum.
    """
    matrix = np.stack(sources).T                   # (dim, n_bodies)
    penalty = 1e3 * np.abs(matrix).max()
    augmented = np.vstack([matrix, penalty * np.ones((1, len(sources)))])
    rhs = np.append(target, penalty)
    weights, _ = nnls(augmented, rhs)
    total = weights.sum()
    return weights / total if total > 1e-9 else np.full(len(sources), 1.0 / len(sources))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="npz from wm.predict_actions")
    ap.add_argument("--data_dir", default="data/ik_walk_8body")
    ap.add_argument("--train_morphs", nargs="+", default=None,
                    help="defaults to the train_morphs recorded in the prediction file")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = np.load(args.pred, allow_pickle=True)
    held = str(data["morph"])
    train = list(args.train_morphs or [str(x) for x in data["train_morphs"]])
    data_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(ROOT, args.data_dir)

    episodes = [int(os.path.basename(str(c)).split("_ep")[1].split(".")[0]) for c in data["clips"]]
    frames = int(data["lengths"][0])
    target_actions = mean_actions(data_dir, held, episodes, frames)
    sources = {b: mean_actions(data_dir, b, episodes, frames) for b in train}

    pred = np.degrees(data["pred"])
    stack = [sources[b].ravel() for b in train]
    model_weights = fit_mixture(pred.ravel(), stack)
    true_weights = fit_mixture(target_actions.ravel(), stack)

    scales = np.array([scales_from_name(b) for b in train])
    implied = model_weights @ scales
    reachable = true_weights @ scales
    actual = np.array(scales_from_name(held))

    print(f"held out '{held}'  scales coxa {actual[0]:.2f} femur {actual[1]:.2f} tibia {actual[2]:.2f}")
    print(f"\n{'training body':<14}{'model implies':>15}{'best possible':>15}")
    for i, b in enumerate(train):
        print(f"{b:<14}{model_weights[i]:>15.3f}{true_weights[i]:>15.3f}")
    print(f"{'concentration':<14}{model_weights.max():>15.3f}{true_weights.max():>15.3f}"
          "   (1.0 = copied a single body)")

    print(f"\n{'segment scale':<14}{'model implies':>15}{'best possible':>15}{'actual':>10}")
    for i, name in enumerate(("coxa", "femur", "tibia")):
        print(f"{name:<14}{implied[i]:>15.3f}{reachable[i]:>15.3f}{actual[i]:>10.2f}")

    nearest = min(train, key=lambda b: np.sqrt(((sources[b] - target_actions) ** 2).mean()))
    rmse = lambda p: float(np.sqrt(((p - target_actions) ** 2).mean()))
    print(f"\n{'predictor':<34}{'RMSE deg':>10}")
    print(f"{'model':<34}{rmse(pred):>10.2f}")
    print(f"{'model mixture, applied to GT':<34}"
          f"{rmse(sum(w * sources[b] for w, b in zip(model_weights, train))):>10.2f}")
    print(f"{'best possible mixture':<34}"
          f"{rmse(sum(w * sources[b] for w, b in zip(true_weights, train))):>10.2f}")
    print(f"{'copy nearest (' + nearest + ')':<34}{rmse(sources[nearest]):>10.2f}")

    results = {
        "held_out": held, "train_morphs": train, "epoch": int(data["epoch"]),
        "weights_model": dict(zip(train, model_weights.round(4).tolist())),
        "weights_best_possible": dict(zip(train, true_weights.round(4).tolist())),
        "scales_implied": implied.round(4).tolist(),
        "scales_best_possible": reachable.round(4).tolist(),
        "scales_actual": list(actual),
        "concentration_model": float(model_weights.max()),
        "rmse_deg_model": rmse(pred),
    }
    out = args.out or os.path.join(
        ROOT, "results", "wm", f"mix_{os.path.splitext(os.path.basename(args.pred))[0]}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
