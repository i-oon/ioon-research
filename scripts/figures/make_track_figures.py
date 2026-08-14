"""The three coverage/latent-share results, each as its own slide-sized figure.

These began as one three-panel strip, which made every panel too small to read projected and forced
three unrelated results onto whichever slide the strip landed on. They answer different questions
and belong on different slides, so each is written separately:

  coverage_experiment.png     does filling the gap the diagnosis named actually close it
  latent_variance_share.png   how much of the latent is "which body/robot is this", Stage 1 vs 2
  encoder_probe_matrix.png    does the frozen encoder hand over a shared space, before training

Numbers are results, not measurements made here; each is sourced in the comment beside it so a
figure can never drift from the finding it draws. Regenerate after any of those change.

  .venv/bin/python3 scripts/make_track_figures.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "results", "wm", "figures")

BLUE, RED, GREY, GREEN = "#2471a3", "#c0392b", "#c8ccce", "#1e8449"

# F35: volume-matched retrain, four bodies -> seven, 7,540 -> 7,735 training pairs
BEFORE, AFTER = 27.68, 16.10
CONSTANT_POSE, COPY_NEAREST = 16.01, 10.63

# F19 / F38: two-way variance decomposition of the latent
SHARES = [
    ('Stage 2\nhexapod against B1\nboth trained on', 39.6, 33.0, 27.4),
    ('Stage 1\nacross insect bodies\nwith the cross-body loss', 88.7, 1.2, 10.1),
]

# F37 / F41: stance-fraction readout on the frozen encoder, RMSE over the target's own spread.
#
# Band-pooled patch tokens with each embodiment standardised by its own statistics -- the setting
# that controls the most and transfers best, and therefore the fair test of whether the behaviour
# is readable across bodies at all.
#
# Two nuisances had to be removed to get here. The reduction: mean-pooling over all 256 patches
# buries a quantity living in the 6-12 patches near the feet, and preserves a large constant
# offset between the two embodiments' frames which a fitted readout absorbs and then mis-applies.
# The appearance: the insect renders orange and small in frame, the B1 grey and large, neither of
# which is behaviour. Per-embodiment standardisation removes both, using only which dataset a
# frame came from and never the stance fraction being predicted.
#
# Un-normalised, the cross cells swing from 1.06x to 4.72x depending purely on the reduction.
# Normalised they sit in 1.02x-1.57x whichever reduction is used, which is why this is the number
# reported: it is the one that is a property of the encoder rather than of the pooling.
PROBE = np.array([[0.82, 1.16],      # fitted on insect -> insect, B1
                  [1.04, 0.89]])     # fitted on B1     -> insect, B1
PROBE_RANGE = ("band-pooled, each embodiment standardised by its own statistics\n"
               "without that control the cross cells range 1.06x to 4.72x by reduction alone")


def save(fig, name, dpi):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {path}")


def coverage(dpi):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.bar([0, 1], [BEFORE, AFTER], width=0.5, color=[RED, BLUE])
    for x, v in ((0, BEFORE), (1, AFTER)):
        ax.text(x, v + 0.7, f"{v:.2f}", ha="center", fontsize=15, fontweight="bold")

    # the two no-learning references the result has to be read against: beating a constant pose is
    # the minimum bar, copying the nearest training body is what coverage alone would buy.
    # Labelled in the right margin because the constant-pose line lands within 0.1 deg of the
    # "after" bar's top, so anything placed over the axes collides with the bar's own value.
    for y, colour, text in ((CONSTANT_POSE, "0.35", f"predict a constant pose, {CONSTANT_POSE:.2f}"),
                            (COPY_NEAREST, GREEN, f"copy the nearest body, {COPY_NEAREST:.2f}")):
        ax.axhline(y, ls="--", lw=1.4, color=colour)
        ax.text(1.34, y, text, fontsize=11, color=colour, va="center", ha="left",
                bbox=dict(fc="white", ec=colour, alpha=.9, boxstyle="round,pad=0.28"))

    ax.annotate("", xy=(0.72, AFTER + 3.0), xytext=(0.14, BEFORE - 1.2),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="0.25"))
    ax.text(0.88, BEFORE - 3.8, f"{BEFORE / AFTER:.1f}x", fontsize=16, fontweight="bold",
            color="0.25", ha="center")

    ax.set_xlim(-0.5, 2.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before\n4 bodies", "after\n7 bodies"], fontsize=12)
    ax.set_ylabel("held-out error, degrees per joint", fontsize=12)
    ax.set_ylim(0, BEFORE * 1.18)
    ax.set_title("Filling the gap the diagnosis named\n"
                 "same data volume: 7,735 training pairs against 7,540", fontsize=13)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    save(fig, "coverage_experiment.png", dpi)


def variance_share(dpi):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = [s[0] for s in SHARES]
    y = np.arange(len(SHARES))[::-1]
    left = np.zeros(len(SHARES))
    for col, (colour, name) in enumerate(((BLUE, "gait phase"),
                                          (RED, "which robot / body"),
                                          (GREY, "interaction"))):
        width = np.array([s[col + 1] for s in SHARES])
        ax.barh(y, width, left=left, height=0.42, color=colour, label=name)
        for yi, (w, l) in zip(y, zip(width, left)):
            # 1.2% is too narrow to hold its own label, so it is called out from outside the bar
            if w < 4:
                ax.annotate(f"{w:.1f}%", xy=(l + w / 2, yi - 0.21),
                            xytext=(l + w / 2 + 6, yi - 0.46), fontsize=12,
                            fontweight="bold", color=RED, ha="center",
                            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
            else:
                ax.text(l + w / 2, yi, f"{w:.1f}%", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if colour != GREY else "0.2")
        left = left + width

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("share of the latent's variance", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.85, len(SHARES) - 0.4)
    ax.set_title('A third of the latent is "which robot"\n'
                 "even on embodiments it trained on", fontsize=13)
    ax.legend(fontsize=10, loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.02))
    save(fig, "latent_variance_share.png", dpi)


def probe_matrix(dpi):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    # green below 1.0 (the readout beats guessing), amber to red above it. The boundary is the
    # whole point of the figure, so the thresholds sit just above 1.00 rather than at round
    # numbers chosen for looks.
    colours = [["#27ae60" if v < 1 else ("#e8b33c" if v < 1.5 else RED) for v in row]
               for row in PROBE]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, fc=colours[i][j], ec="white", lw=2.5))
            ax.text(j + .5, 1.5 - i, f"{PROBE[i, j]:.2f}x", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="white" if PROBE[i, j] > 3.5 else "0.15")

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([.5, 1.5]); ax.set_yticks([.5, 1.5])
    ax.set_xticklabels(["tested on\ninsect", "tested on\nB1"], fontsize=12)
    ax.set_yticklabels(["fitted on\nB1", "fitted on\ninsect"], fontsize=12)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Frozen encoder, before any training\n"
                 "stance fraction: error over the target's own spread", fontsize=13)
    ax.set_xlabel("above 1.00x means no better than predicting the average\n"
                  + PROBE_RANGE, fontsize=10, labelpad=10)
    save(fig, "encoder_probe_matrix.png", dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    coverage(args.dpi)
    variance_share(args.dpi)
    probe_matrix(args.dpi)


if __name__ == "__main__":
    main()
