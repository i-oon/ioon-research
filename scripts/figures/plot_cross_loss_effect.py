"""Slide 5's figure: what the cross-body loss does to the decoder's two inputs.

The claim the figure has to carry is a division of labour, not an improvement in error. With the
cross-body term the decoder reads *which body* from the frame and *what movement* from the latent;
without it, the frame is barely read at all and the latent does both jobs.

Left  -- held-out error per epoch, the two runs against each other. Read the tail, not any single
         epoch: held-out error swings within a run, which is why the shaded band is the last five
         epochs' spread rather than a point.
Right -- the cost of deleting each input, per run. The bars are what the claim rests on: without
         the term, deleting the frame costs almost nothing.

Reads the training runs' own tfevents, so the figure cannot drift from the logs.

  .venv/bin/python3 scripts/figures/plot_cross_loss_effect.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

CROSS, CONTROL = "#2a7fb8", "#c0392b"


def series(run, tag):
    ea = EventAccumulator(os.path.join(ROOT, "wm", "runs", run, "summary"))
    ea.Reload()
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cross", default="stage1_m3d_cross")
    ap.add_argument("--control", default="stage1_m3d_bracketed")
    ap.add_argument("--out", default="results/wm/stage1_correct/figures/cross_loss_effect.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for run, colour, label in ((args.control, CONTROL, "without the cross-body loss"),
                               (args.cross, CROSS, "with the cross-body loss")):
        steps, values = series(run, "heldout/motion")
        left.plot(steps, values, color=colour, lw=1.9, label=label)
        tail = values[-5:]
        left.axhspan(tail.min(), tail.max(), color=colour, alpha=0.13, lw=0)
    left.set_xlabel("epoch")
    left.set_ylabel("held-out error, standardised")
    left.set_title("Held-out body through training\nband = spread of the last five epochs",
                   fontsize=11)
    left.legend(fontsize=9, frameon=False)
    left.spines[["top", "right"]].set_visible(False)

    # Right: cost of deleting each input, at the final epoch.
    names, gaps = [], []
    for run, label in ((args.control, "without"), (args.cross, "with")):
        _, base = series(run, "heldout/motion")
        _, no_z = series(run, "heldout/motion_zero_z")
        _, no_x = series(run, "heldout/motion_zero_x")
        names.append(label)
        gaps.append((no_x[-1] / base[-1], no_z[-1] / base[-1]))
    gaps = np.array(gaps)

    x = np.arange(2)
    width = 0.36
    right.bar(x - width / 2, gaps[:, 0], width, color="#2a7fb8", label="delete the frame")
    right.bar(x + width / 2, gaps[:, 1], width, color="#7f8c8d", label="delete the latent")
    for i in range(2):
        right.text(i - width / 2, gaps[i, 0], f"{gaps[i, 0]:.1f}x", ha="center", va="bottom",
                   fontsize=11, fontweight="bold")
        right.text(i + width / 2, gaps[i, 1], f"{gaps[i, 1]:.1f}x", ha="center", va="bottom",
                   fontsize=11)
    right.axhline(1.0, color="0.4", ls=":", lw=1)
    right.set_xticks(x)
    right.set_xticklabels([f"{n}\nthe cross-body loss" for n in names], fontsize=10)
    right.set_ylabel("error multiplier when the input is deleted")
    right.set_title("Which input the decoder actually uses\n"
                    "1.0x means deleting it costs nothing", fontsize=11)
    right.legend(fontsize=9, frameon=False, loc="upper left")
    right.spines[["top", "right"]].set_visible(False)
    right.set_ylim(0, max(gaps.max() * 1.25, 2))

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    print(f"-> {out}")
    print(f"   delete-frame  {gaps[0,0]:.2f}x without, {gaps[1,0]:.2f}x with")
    print(f"   delete-latent {gaps[0,1]:.2f}x without, {gaps[1,1]:.2f}x with")


if __name__ == "__main__":
    main()
