"""Stage 2 (cross-embodiment) Training/Testing diagram, matching the visual style of the
Stage 1 diagram screenshot (yellow=simulator, gray=processing block, pink=Motion Decoder,
light blue=loss). Two real differences from Stage 1's medium test, drawn explicitly:
  1. Motion Decoder needs a PER-EMBODIMENT output head (hexapod 18-D, B1 12-D are disjoint
     spaces -> no single head can emit both).
  2. The held-out 4-leg body can't be zero-shot like Stage 1's medium -- its action space
     is new (not 18-D, not 12-D), so a small NEW head must be calibrated on a little real
     4-leg (frame, action) data (few-shot), not full retraining, and ITM/FTM/backbone stay frozen.

  .venv/bin/python3 scripts/make_stage2_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "pipeline_diagram_stage2_cross_embodiment.png")

COL = dict(
    sim="#fdf1a8",       # yellow -- simulator
    proc="#e6e6e6",      # gray -- generic processing block (ITM/FTM/camera/logger)
    md="#f9c8d9",        # pink -- Motion Decoder
    loss="#cfe3fb",      # light blue -- loss
    frame="#dddddd",     # frame thumbnail placeholder
    gt="#e63946",        # red -- ground-truth / held-out annotations
    new="#ffb703",       # amber outline -- newly-calibrated head
)


def box(ax, xy, w, h, text, fc, ec="#555555", lw=1.3, fontsize=9, style="round,pad=0.02,rounding_size=0.08"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                        edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, b1, b2, side1="right", side2="left", label=None, dashed=True, color="#666666", label_dy=0.12):
    def pt(b, side):
        x, y, w, h = b
        return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
                "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]
    p1, p2 = pt(b1, side1), pt(b2, side2)
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, linewidth=1.1,
                        color=color, linestyle=(0, (4, 2)) if dashed else "solid", zorder=2)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.3, color="#333333",
                path_effects=[withStroke(linewidth=2.2, foreground="white")], zorder=5)


fig, ax = plt.subplots(figsize=(26, 9))
ax.set_xlim(0, 26); ax.set_ylim(0, 9); ax.axis("off")

# ============================= TRAINING (left) =============================
ax.text(0.3, 8.5, "Training  —  Stage 2 (hexapod + B1, disjoint action spaces)",
        fontsize=13, fontweight="bold", style="italic")

TRAIN_BG_W = 12.6
bg = FancyBboxPatch((0.15, 0.2), TRAIN_BG_W, 8.05, boxstyle="round,pad=0.02,rounding_size=0.1",
                    linewidth=0, facecolor="#f3f1ee", zorder=1)
ax.add_patch(bg)

sim = box(ax, (0.5, 6.9), 1.9, 0.7, "Simulator\n(CoppeliaSim / MuJoCo→replay)", COL["sim"])
jlog = box(ax, (3.0, 6.9), 1.9, 0.7, "Joint logger\n(per-embodiment)", COL["proc"])
cam = box(ax, (0.5, 5.7), 1.9, 0.6, "Camera sensor", COL["proc"])
arrow(ax, sim, jlog, "right", "left", "joint commands\neach t")
arrow(ax, sim, cam, "bottom", "top", dashed=False)

hexf = box(ax, (0.4, 4.1), 1.5, 1.1, "hexapod\n(18-D)", COL["frame"], fontsize=8.5)
b1f = box(ax, (2.1, 4.1), 1.5, 1.1, "B1\n(12-D)", COL["frame"], fontsize=8.5)
arrow(ax, cam, hexf, "left", "top", dashed=False, color="#888888")
arrow(ax, cam, b1f, "left", "top", dashed=False, color="#888888")

itm = box(ax, (4.35, 5.15), 2.0, 0.85, "Inverse Transition\nModel (ITM)", COL["proc"], fontsize=8.5)
ftm = box(ax, (4.35, 3.0), 2.0, 0.85, "Forward Transition\nModel (FTM)", COL["proc"], fontsize=8.5)
arrow(ax, hexf, itm, "right", "left", "$e_t^1,e_{t+1}^1$", label_dy=0.15)
arrow(ax, b1f, ftm, "right", "left", "$e_t^2,e_{t+1}^2$", label_dy=-0.28)

zt = box(ax, (6.85, 5.3), 0.95, 0.6, "$z_t$", "#ffe8cf", fontsize=10)
arrow(ax, itm, zt, "right", "left", dashed=False)

md = box(ax, (8.2, 5.05), 1.55, 2.0, "Motion\nDecoder\n(shared\nbackbone)", COL["md"], fontsize=8.7)
arrow(ax, zt, md, "right", "left", dashed=False)
arrow(ax, hexf, md, "top", "bottom", "$x_t$ (visual context)", label_dy=0.1)

hxh = box(ax, (10.15, 6.15), 2.35, 0.55, "hexapod head   $\\hat a_t \\in \\mathbb{R}^{18}$", "#fde2ea", fontsize=7.8)
b1h = box(ax, (10.15, 5.3), 2.35, 0.55, "B1 head   $\\hat a_t \\in \\mathbb{R}^{12}$", "#fde2ea", fontsize=7.8)
arrow(ax, md, hxh, "right", "left", dashed=False, color="#c0507a")
arrow(ax, md, b1h, "right", "left", dashed=False, color="#c0507a")

lmot = box(ax, (9.95, 3.55), 2.6, 1.0, "$\\mathcal{L}_{motion}$\n$\\|\\hat a_t - a_t\\|^2$\n(per embodiment)", COL["loss"], fontsize=8)
arrow(ax, hxh, lmot, "bottom", "top", dashed=False, color="#4477aa")
arrow(ax, b1h, lmot, "bottom", "top", dashed=False, color="#4477aa")
arrow(ax, jlog, lmot, "right", "top", "ground-truth $a_t$", dashed=True, color="#888888", label_dy=0.1)

ehat = box(ax, (6.85, 3.15), 1.6, 0.6, "$\\hat e_{t+1}$ predicted", "#eef0f5", fontsize=7.8)
arrow(ax, ftm, ehat, "right", "left", dashed=False)
lrec = box(ax, (8.7, 3.15), 1.75, 0.6, "$\\mathcal{L}_{recon}$\n$\\|\\hat e_{t+1}-e_{t+1}\\|^2$", COL["loss"], fontsize=7.8)
arrow(ax, ehat, lrec, "right", "left", dashed=False, color="#4477aa")

ax.text(6.45, 1.0,
        "One shared ITM/FTM backbone trained jointly on BOTH bodies (vision is embodiment-agnostic).\n"
        "Motion Decoder splits into a per-embodiment head at the very end — 18-D and 12-D are disjoint,\n"
        "so no single head can emit both.", fontsize=8.3, ha="center", style="italic", color="#444444")

# ============================= TESTING (right) =============================
TX = 13.6   # x-offset for the whole testing panel, clear of the training background (right edge 12.75)
ax.text(TX, 8.5, "Testing  —  held-out 4-leg stick insect (“middle-loss”)",
        fontsize=13, fontweight="bold", style="italic")

sim2 = box(ax, (TX, 6.9), 1.9, 0.7, "Simulator\n(CoppeliaSim)", COL["sim"])
jlog2 = box(ax, (TX + 5.6, 6.9), 2.3, 0.7, "Joint logger", "white", ec=COL["gt"], lw=1.6)
ax.text(TX + 5.6 + 1.15, 7.78, "Ground-truth action $a_t$", ha="center", fontsize=7.8, color=COL["gt"])
cam2 = box(ax, (TX, 5.7), 1.9, 0.6, "Camera sensor", COL["proc"])
arrow(ax, sim2, jlog2, "right", "left", dashed=True, color="#888888")
arrow(ax, sim2, cam2, "bottom", "top", dashed=False)

fourleg = box(ax, (TX - 0.1, 4.0), 1.9, 1.2, "4-leg insect\n(“middle-loss”)\nnew action space", "white", ec=COL["gt"], lw=1.6, fontsize=8)
ax.text(TX + 0.85, 3.82, "Ground-truth features", ha="center", fontsize=7.8, color=COL["gt"])
arrow(ax, cam2, fourleg, "left", "top", dashed=False, color="#888888")

itm2 = box(ax, (TX + 2.3, 5.15), 2.0, 0.85, "ITM (frozen)", COL["proc"], fontsize=8.5)
arrow(ax, fourleg, itm2, "right", "left", "$e_t^1,e_{t+1}^1$", label_dy=0.15)

zt2 = box(ax, (TX + 4.75, 5.3), 0.95, 0.6, "$z_t$", "#ffe8cf", fontsize=10)
arrow(ax, itm2, zt2, "right", "left", dashed=False)

newh = box(ax, (TX + 6.15, 4.95), 2.75, 1.15,
          "NEW small head\n(calibrated on a\nLITTLE real 4-leg data\n— few-shot, not zero-shot)",
          "#fff3d6", ec=COL["new"], lw=2.0, fontsize=7.8)
arrow(ax, zt2, newh, "right", "left", dashed=False)
arrow(ax, fourleg, newh, "top", "bottom", "$x_t$", label_dy=0.05)

lmot2 = box(ax, (TX + 6.3, 3.35), 2.5, 0.85, "$\\mathcal{L}_{motion}$\n$\\|\\hat a_t - a_t\\|^2$", COL["loss"], fontsize=8.5)
arrow(ax, newh, lmot2, "bottom", "top", dashed=False, color="#4477aa")
arrow(ax, jlog2, lmot2, "bottom", "right", dashed=True, color=COL["gt"], label_dy=-0.05)

ax.text(TX + 0.1, 2.9,
        "ITM stays frozen (unchanged from training) — only this small new head is fit,\n"
        "with a handful of real 4-leg episodes. Cannot be pure zero-shot like Stage 1's\n"
        "medium test: 4-leg's action space is genuinely new (not 18-D, not 12-D), so\n"
        "there is no existing head whose output already means anything for it.",
        fontsize=8.2, ha="left", style="italic", color="#444444")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved ->", OUT)
