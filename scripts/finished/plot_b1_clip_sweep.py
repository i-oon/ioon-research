"""B1 clips used for stage 4 (body-head fit) vs. held-out readout -- Section 10's clip-count table as a plot.

    .venv/bin/python3 scripts/figures/plot_b1_clip_sweep.py

Numbers are the F238 sweep (12 held-out B1 clips, same head recipe, only the stage-4 clip count changes).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = [3, 6, 12, 24]
RATIO = [1.530, 0.865, 0.736, 0.694]
RHO = {"forward": [0.169, 0.218, 0.326, 0.428],
       "lateral": [0.191, 0.512, 0.551, 0.537],
       "yaw":     [0.359, 0.391, 0.466, 0.466]}
MEDIAN = [0.191, 0.391, 0.466, 0.466]
COL = {"forward": "#2a78d6", "lateral": "#eb6834", "yaw": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
x = range(len(N))
for ax in (a, b):
    ax.set_xticks(list(x)); ax.set_xticklabels([str(n) if n != 24 else "24 (full)" for n in N])
    ax.set_xlabel("B1 clips used to fit stage 4", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)

a.plot(x, RATIO, color=COL["forward"], lw=2, marker="o", ms=7, mfc="white", mew=2)
a.axhline(1.0, color=MUTED, lw=1, ls="--")
a.text(3.05, 1.0, "1.0 = no better\nthan the mean", va="bottom", ha="right", fontsize=8, color=MUTED)
for i, v in zip(x, RATIO):
    a.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
a.set_ylim(0.5, 1.7)
a.set_ylabel("held-out ratio (lower is better)", color=MUTED, fontsize=9)
a.set_title("Held-out ratio", loc="left", color=INK, fontsize=11, fontweight="bold")

for k, v in RHO.items():
    b.plot(x, v, color=COL[k], lw=2, marker="o", ms=6, mfc="white", mew=2)
    b.annotate(k, (3, v[-1]), textcoords="offset points", xytext=(8, 0), va="center", fontsize=9, color=COL[k])
b.plot(x, MEDIAN, color=INK, lw=1.5, ls="--", marker="s", ms=5)
b.text(0.02, 0.97, "dashed black = median of the three channels\n(equals yaw at 12 and 24 clips)", transform=b.transAxes, va="top", fontsize=8.5, color=INK)
b.set_xlim(-0.3, 3.7); b.set_ylim(0.1, 0.65)
b.set_ylabel("held-out ρ with Froude", color=MUTED, fontsize=9)
b.set_title("Per-channel readout", loc="left", color=INK, fontsize=11, fontweight="bold")

fig.suptitle("How many B1 clips does stage 4 need?", x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(ROOT, "results/deck/b1_clip_count_sweep.png")
plt.savefig(out, dpi=140)
print(out)
