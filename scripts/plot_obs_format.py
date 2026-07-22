"""Secondary figure for "Why Test Vision When Proprioception Is Available?".

Schematic, not data. Descriptive contrast only: proprioceptive vectors differ in
dimension and ordering between robot classes, while RGB observations share a
tensor shape. This figure DESCRIBES the difference; it does not by itself argue
that vision is necessary. If asked "why not just remap the indices?", the answer
is that remapping requires documentation of both bodies, which is exactly what
the access argument (Slide 9) assumes is unavailable. Keep the two separate.

Dimensions are symbolic (N, M). Real sensor layouts for published robots are
deliberately not invented here.

Usage:
  .venv/bin/python scripts/plot_obs_format.py
"""
import matplotlib.patches as mp
import matplotlib.pyplot as plt

OUT = "report/fig_obs_format.png"
MONO = {"family": "monospace", "ha": "center"}


def robot(ax, x, y, n_legs, colour, name="", sub=""):
    """Schematic top-down body with n_legs legs, half per side."""
    ax.add_patch(mp.FancyBboxPatch((x - 0.34, y - 0.13), 0.68, 0.26,
                                   boxstyle="round,pad=0.02", fc=colour, ec="none", alpha=0.85))
    per = n_legs // 2
    for i in range(per):
        bx = x - 0.26 + i * (0.52 / max(per - 1, 1))
        for s in (1, -1):
            ax.plot([bx, bx + s * 0.055], [y + s * 0.13, y + s * 0.30], color=colour, lw=2.1,
                    solid_capstyle="round")
            ax.plot([bx + s * 0.055, bx + s * 0.15], [y + s * 0.30, y + s * 0.40], color=colour,
                    lw=2.1, solid_capstyle="round")
    if name:
        ax.text(x, y + 0.62, name, ha="center", fontsize=12, fontweight="bold")
        ax.text(x, y + 0.48, sub, ha="center", fontsize=9, color="0.35")


def proprio_block(ax, x, y, vector, lines, dim):
    """Plain-text proprioception vector: contents, index meanings, dimension."""
    ax.text(x, y, vector, fontsize=12.5, **MONO)
    for i, line in enumerate(lines):
        ax.text(x, y - 0.34 - i * 0.24, line, fontsize=9.6, color="0.3", **MONO)
    ax.text(x, y - 0.34 - len(lines) * 0.24 - 0.06, dim, fontsize=10.2,
            color="0.15", fontweight="bold", **MONO)


fig, ax = plt.subplots(figsize=(11.5, 8.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.35)
ax.axis("off")

LX, RX = 2.75, 7.25

ax.text(5, 7.05, "Scope: this thesis tests three hexapods with $N = M = 18$ and identical ordering, "
                 "the case most favourable to proprioception.",
        ha="center", fontsize=8.6, color="0.42", style="italic")

# --- robots ---------------------------------------------------------------
robot(ax, LX, 5.80, 6, "#c1121f", "BODY A", "hexapod")
robot(ax, RX, 5.80, 4, "#0466c8", "BODY B", "quadruped")

for x in (LX, RX):
    ax.annotate("", xy=(x, 5.02), xytext=(x, 5.32),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.4))

# --- proprioception, as plain text ---------------------------------------
# Ordering differs on purpose: index 7 is a femur on A and a knee on B.
proprio_block(ax, LX, 4.76, "[p1 p2 p3 ... pN]",
              ["joint 1 = front coxa", "joint 7 = middle femur"], "dimension N")
proprio_block(ax, RX, 4.76, "[p1 p2 ... pM]",
              ["joint 1 = front-left hip", "joint 7 = rear-right knee"], "dimension M")

ax.text(5, 3.44, "DIFFERENT INTERNAL INTERFACES", ha="center", fontsize=12.5,
        fontweight="bold", color="#9d0208")
ax.text(5, 3.16, "different dimension, and a different semantic ordering of the same quantities",
        ha="center", fontsize=9, color="0.35")

ax.plot([0.6, 9.4], [2.86, 2.86], color="0.7", lw=1.1, ls="--")

# --- RGB observations -----------------------------------------------------
for x, colour, n in ((LX, "#c1121f", 6), (RX, "#0466c8", 4)):
    ax.add_patch(mp.Rectangle((x - 0.72, 1.58), 1.44, 1.08, fc="0.96", ec="0.45", lw=1.5))
    ax.add_patch(mp.Rectangle((x - 0.72, 2.66), 1.44, 0.16, fc="0.45", ec="none"))
    robot(ax, x, 2.01, n, colour)
    ax.text(x, 1.30, r"$H \times W \times 3$", ha="center", fontsize=11)

ax.text(5, 0.84, "COMMON EXTERNAL FORMAT", ha="center", fontsize=12.5,
        fontweight="bold", color="#1b4332")
ax.text(5, 0.56, "same tensor shape on any body, obtainable without access to the body's interior",
        ha="center", fontsize=9, color="0.35")
ax.text(5, 0.22, "Common format is a precondition, not the result: morphology still decodes from raw "
                 "features at 99.9%.\nRemoving that is what the inverse transition model is for.",
        ha="center", fontsize=8.5, color="#9d0208", style="italic", linespacing=1.6)

fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"saved: {OUT}")
