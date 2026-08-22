"""Contact raster for a few clips side by side: which feet are down, when.

Duty factors and tripod-agreement numbers say a gait is or is not grouped, but not *how* it is
grouped, and the difference matters -- an oscillator driven with a tripod sign pattern scored 0.6
on within-group agreement against a recorded wave gait's 0.5, which is either a meaningful change
or none at all depending on what the raster looks like.

Legs are ordered so the two tripods sit together: **FL HL MR** above the line, **ML FR HR** below.
A clean alternating tripod is two solid blocks in antiphase. A wave is a staircase.

  .venv/bin/python3 scripts/dataset/plot_gait_compare.py /tmp/rec2 /tmp/lift0.20 --labels wave cpg
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
ORDER = ["FL", "HL", "MR", "ML", "FR", "HR"]     # tripod A on top, tripod B below
THRESHOLD = 0.27                                  # N, validated in plot_gait_quality.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="results/wm/dataset/gait_compare.png")
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.dirs]
    fig, axes = plt.subplots(len(args.dirs), 1, figsize=(11, 2.0 * len(args.dirs)), squeeze=False)

    for row, (d, label) in enumerate(zip(args.dirs, labels)):
        paths = sorted(p for p in glob.glob(os.path.join(d, "*.npz")) if "manifest" not in p)
        ax = axes[row][0]
        if not paths:
            ax.set_axis_off()
            continue
        with np.load(paths[0], allow_pickle=True) as clip:
            contact = clip["forces"] > THRESHOLD
        for r, leg in enumerate(ORDER):
            on = contact[:, LEGS.index(leg)]
            ax.broken_barh([(i, 1) for i, v in enumerate(on) if v], (r + 0.1, 0.8),
                           facecolors="#2a9d8f" if leg in ORDER[:3] else "#e76f51")
        ax.axhline(3.0, color="#22333b", lw=1.0)
        ax.set_yticks(np.arange(6) + 0.5)
        ax.set_yticklabels(ORDER, fontsize=8)
        ax.set_ylim(0, 6)
        ax.set_xlim(0, len(contact))
        ax.invert_yaxis()
        ax.set_title(f"{label}   —   {contact.sum(1).mean():.2f} feet down on average",
                     fontsize=9, loc="left")
        if row == len(args.dirs) - 1:
            ax.set_xlabel("frame")

    fig.suptitle("Foot contact. Teal = tripod A (FL HL MR), orange = tripod B (ML FR HR).  "
                 "A clean tripod is two solid blocks in antiphase.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
