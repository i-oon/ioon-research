"""Slide 19's figure: what we ran, and what the source method actually does.

Three panels, left to right, each a data-flow sketch of the motion decoder.

    what we had     per-embodiment heads only. 18-D and 12-D joint commands live in different
                    spaces, so nothing requires one `z` to mean the same thing on both robots.
    what works      the same, plus one shared head reading `z` alone. One mapping for both robots,
                    so `z` has to use one code. +0.544 / +0.435.
    LAC-WM          **no joint-angle head at all.** One decoder, one target, and that target is a
                    position in a physical space both embodiments share.

**A frame-conditioned version of our shared head was also run and scored -10.5**, worse than no
term at all, which is where F64 comes from. It is not drawn: it differs from LAC-WM's decoder in
*two* ways at once -- the frame conditioning and a one-dimensional state target sitting beside
joint heads that carry ten times its weight -- so the panel invited the reading that we had tested
the published design and refuted it. We had not. The number belongs in a sentence, not a diagram.

The third panel carries no score because it has not been run. It is drawn to show that all our
variants share a shape the source method does not: joint angles as the main target, with alignment
bolted on beside them.

  .venv/bin/python3 scripts/figures/plot_body_head_design.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAL, ORANGE, GREY, INK = "#2a9d8f", "#e76f51", "#adb5bd", "#22333b"


def box(ax, x, y, w, h, label, face="white", edge=INK, fs=8.5, lw=1.2, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, zorder=3, color=INK)


def arrow(ax, p, q, colour=INK, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9,
                                 color=colour, linewidth=lw, zorder=1,
                                 shrinkA=1, shrinkB=1))


def panel(ax, title, verdict, verdict_colour, shared_head=None, source=False):
    """shared_head: None or 'z'. source=True draws LAC-WM's shape instead of ours."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)

    box(ax, 0.2, 7.6, 1.5, 0.9, "$e_t$")
    box(ax, 0.2, 5.9, 1.5, 0.9, "$z$")
    box(ax, 2.6, 6.6, 2.1, 1.9, "trunk\ncross-attn", face="#f1f3f5")
    arrow(ax, (1.7, 8.05), (2.6, 7.9))
    arrow(ax, (1.7, 6.35), (2.6, 7.2))

    if source:
        # one MLP, one target -- "the resulting cross-attended features are fed into an MLP"
        box(ax, 5.6, 7.0, 4.1, 1.1, "one MLP  →  positions in a\nshared physical space",
            face="#eef2ff", edge="#4c6ef5", lw=1.6)
        arrow(ax, (4.7, 7.55), (5.6, 7.55), colour="#4c6ef5", lw=1.6)
        ax.text(7.65, 6.3, "no joint-angle head anywhere",
                fontsize=7.8, color="#4c6ef5", ha="center")
        ax.text(7.65, 5.8, "→ one target, shared by construction\n→ nothing to align alongside it",
                fontsize=7.8, color="#4c6ef5", ha="center", va="top")
        ax.text(5.0, 1.4, verdict, fontsize=10.5, color=verdict_colour, linespacing=1.6,
                ha="center", va="center", weight="bold")
        return

    box(ax, 5.6, 7.9, 3.9, 0.85, "head [hexapod]  →  18", face="white")
    box(ax, 5.6, 6.7, 3.9, 0.85, "head [b1]  →  12", face="white")
    arrow(ax, (4.7, 7.9), (5.6, 8.3))
    arrow(ax, (4.7, 7.2), (5.6, 7.1))
    ax.text(9.7, 7.5, "per\nembodiment", fontsize=7.5, color=GREY, va="center", ha="left")

    if shared_head == "frame":
        box(ax, 5.6, 4.9, 3.9, 0.85, "body_head  →  speed", face="#ffe8e0", edge=ORANGE, lw=1.6)
        arrow(ax, (4.7, 6.7), (5.6, 5.5), colour=ORANGE, lw=1.6)
        ax.text(7.55, 4.35, "sees the frame through the trunk",
                fontsize=7.8, color=ORANGE, ha="center")
        ax.text(7.55, 3.85, "→ the frame says which robot\n→ one mapping per robot",
                fontsize=7.8, color=ORANGE, ha="center", va="top")
    elif shared_head == "z":
        box(ax, 5.6, 4.9, 3.9, 0.85, "body_head  →  speed", face="#dff3f0", edge=TEAL, lw=1.6)
        arrow(ax, (0.95, 5.9), (0.95, 5.32), colour=TEAL, lw=1.6, style="-")
        arrow(ax, (0.95, 5.32), (5.6, 5.32), colour=TEAL, lw=1.6)
        ax.text(7.55, 4.35, "$z$ only — no route to the frame",
                fontsize=7.8, color=TEAL, ha="center")
        ax.text(7.55, 3.85, "→ cannot tell the robots apart\n→ one mapping, so one code",
                fontsize=7.8, color=TEAL, ha="center", va="top")
    else:
        ax.text(7.55, 4.6, "nothing asks $z$ to mean\nthe same thing twice",
                fontsize=7.8, color=GREY, ha="center", va="top")

    ax.text(5.0, 1.4, verdict, fontsize=10.5, color=verdict_colour, linespacing=1.6,
            ha="center", va="center", weight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wm/stage2/figures/body_head_design.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    # F83, on the matched behaviour data and held out by condition. The earlier -7.08 / +0.54 was
    # measured on forward-walking-only clips at the wrong B1 frame rate and is not comparable
    # (F74, F84), so it is not shown even as a second row.
    panel(axes[0], "Joint targets alone — what we had",
          "own robot  0.3517\ncross-robot  −28.9", GREY, shared_head=None)
    panel(axes[1], "Ours: one shared head, reading $z$ alone",
          "own robot  0.2183   (−38%)\ncross-robot  +0.61", TEAL, shared_head="z")
    panel(axes[2], "LAC-WM: one decoder, one shared target",
          "no joint head to fix\nnot run", "#4c6ef5", source=True)
    fig.suptitle("A joint-space target works within one robot on its own. It crosses robots only "
                 "with the body term —\nand the body term also cuts joint error by 38%.",
                 fontsize=12, color=INK, y=0.995)
    # The two rows of every verdict are different quantities, and a reader who assumes one scale
    # will read -28.9 as catastrophically bad joint error rather than as a readout correlation.
    fig.text(0.5, 0.035,
             "own robot = validation joint-command MSE, standardised (1.0 = predicting the mean)"
             "        cross-robot = body-speed readout fitted on one robot, applied to the other "
             "(0 = no better than a constant)",
             ha="center", fontsize=8.2, color=GREY)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
