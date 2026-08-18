"""Slide 19's figure: why the shared head has to be blind to the frame.

Three panels, left to right, each a data-flow sketch of the motion decoder.

    what we had     per-embodiment heads. 18-D and 12-D joint commands are different spaces, so
                    nothing requires one `z` to mean the same thing on both robots.
    what failed     one shared head, but conditioned on the frame like LAC-WM's motion decoder.
                    The frame identifies the robot, so the head learns one mapping per robot and
                    `z` is free to stay robot-specific -- the per-embodiment problem re-entering
                    through the image. Measured: -10.5 / -57.2, worse than no term at all.
    what works      one shared head reading `z` alone. One mapping, so `z` must use one code.
                    +0.544 / +0.435.

Drawn rather than described because the distinction is three qualifications in a sentence -- one
head, no embodiment key, blind to the frame -- and one picture of the wiring settles all three.

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


def panel(ax, title, verdict, verdict_colour, shared_head=None):
    """shared_head: None, 'frame' or 'z'."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=INK, pad=8)

    box(ax, 0.2, 7.6, 1.5, 0.9, "$e_t$")
    box(ax, 0.2, 5.9, 1.5, 0.9, "$z$")
    box(ax, 2.6, 6.6, 2.1, 1.9, "trunk\ncross-attn", face="#f1f3f5")
    arrow(ax, (1.7, 8.05), (2.6, 7.9))
    arrow(ax, (1.7, 6.35), (2.6, 7.2))

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

    ax.text(5.0, 1.5, verdict, fontsize=10, color=verdict_colour,
            ha="center", va="center", weight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wm/stage2/figures/body_head_design.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    panel(axes[0], "What we had", "insect→b1   −7.08", GREY, shared_head=None)
    panel(axes[1], "One shared head, conditioned on the frame",
          "insect→b1   −10.48", ORANGE, shared_head="frame")
    panel(axes[2], "One shared head, reading $z$ alone",
          "insect→b1   +0.54", TEAL, shared_head="z")
    fig.suptitle("The shared head only constrains $z$ if it cannot tell the robots apart",
                 fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
