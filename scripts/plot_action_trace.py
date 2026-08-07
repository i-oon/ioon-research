"""Predicted vs ground-truth joint commands, per joint, for a held-out body.

The cheap check before spending simulator time: if the traces track, an open-loop replay
has a chance of walking; if they drift, the video will show why. Per-joint R^2 says how
much of the real command's variation the model captured, and RMSE says how wrong it is in
units a roboticist reads (degrees).

Run from the repository root:
  .venv/bin/python3 scripts/plot_action_trace.py \
      --pred results/wm/predictions/<run>_epoch020_medium.npz
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = ["TC", "CF", "FT"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    data = np.load(args.pred, allow_pickle=True)
    pred, gt = np.degrees(data["pred"]), np.degrees(data["gt"])
    joints = list(data["joints"])
    morph, epoch = str(data["morph"]), int(data["epoch"])
    bounds = np.cumsum(data["lengths"])[:-1]

    residual = ((gt - pred) ** 2).sum(axis=0)
    total = ((gt - gt.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-9)
    rmse = np.sqrt(((gt - pred) ** 2).mean(axis=0))
    amplitude = gt.std(axis=0)

    print(f"body '{morph}', epoch {epoch}, {len(pred)} transitions "
          f"({'held out' if bool(data['held_out']) else 'seen in training'})")
    print(f"{'joint':<8} {'R2':>7} {'RMSE deg':>9} {'GT std deg':>11} {'RMSE/std':>9}")
    for i, name in enumerate(joints):
        print(f"{name:<8} {r2[i]:>7.3f} {rmse[i]:>9.2f} {amplitude[i]:>11.2f} {rmse[i]/amplitude[i]:>9.2f}")
    print(f"{'MEAN':<8} {r2.mean():>7.3f} {rmse.mean():>9.2f} {amplitude.mean():>11.2f} "
          f"{(rmse/amplitude).mean():>9.2f}")

    fig, axes = plt.subplots(6, 3, figsize=(14, 13), sharex=True)
    x = np.arange(len(pred))
    for i, name in enumerate(joints):
        ax = axes[i // 3, i % 3]
        ax.plot(x, gt[:, i], color="k", lw=1.6, label="IK ground truth")
        ax.plot(x, pred[:, i], color="tab:red", lw=1.4, ls="--", label="world model")
        for b in bounds:
            ax.axvline(b, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{name}   R2 {r2[i]:.2f}   RMSE {rmse[i]:.1f}deg", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("timestep (dotted line = clip boundary)", fontsize=8)
    for row in axes:
        row[0].set_ylabel("joint angle (deg)", fontsize=9)

    fig.suptitle(
        f"Action reconstruction on the held-out body '{morph}' -- epoch {epoch}\n"
        f"mean R2 {r2.mean():.2f}, mean RMSE {rmse.mean():.1f} deg against a signal of "
        f"{amplitude.mean():.1f} deg std. Ground-truth frames are the input; this is not closed-loop control.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    out = args.out or os.path.join(
        ROOT, "results", "wm", f"action_trace_{os.path.splitext(os.path.basename(args.pred))[0]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
