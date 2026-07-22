"""Why fixing the command and fixing the foot target give different joint angles.

Planar 2-link leg, drawn at 1.0x and 0.5x link length, which are two of the three
morphologies in this study.

  Left  (what we do now): the same joint angles are commanded to both bodies.
                          The angles match; the foot lands somewhere different.
  Right (IK retargeting): the same foot target is given to both bodies.
                          The foot lands in the same place; the angles differ.

Usage:
  .venv/bin/python scripts/plot_ik_intuition.py
"""
import matplotlib.pyplot as plt
import numpy as np

OUT = "report/fig_ik_intuition.png"
LONG, SHORT = 1.0, 0.5
A1, A2 = np.radians(-60), np.radians(-40)   # commanded angles, shared
TARGET = np.array([0.25, -0.90])            # shared foot target for the IK panel


def forward(L, a1, a2):
    """Hip at origin, returns (hip, knee, foot)."""
    knee = L * np.array([np.cos(a1), np.sin(a1)])
    foot = knee + L * np.array([np.cos(a1 + a2), np.sin(a1 + a2)])
    return np.zeros(2), knee, foot


def inverse(L, target):
    """Planar 2R inverse kinematics, elbow-down solution."""
    d = np.linalg.norm(target)
    assert d <= 2 * L, f"target unreachable for L={L}"
    a2 = -np.arccos(np.clip((d ** 2 - 2 * L ** 2) / (2 * L * L), -1, 1))
    a1 = np.arctan2(target[1], target[0]) - np.arctan2(L * np.sin(a2), L + L * np.cos(a2))
    return a1, a2


def draw(ax, L, a1, a2, colour, label):
    hip, knee, foot = forward(L, a1, a2)
    ax.plot(*zip(hip, knee), color=colour, lw=4, solid_capstyle="round")
    ax.plot(*zip(knee, foot), color=colour, lw=4, solid_capstyle="round")
    ax.plot(*knee, "o", color="white", mec=colour, mew=2.5, ms=9, zorder=3)
    ax.plot(*foot, "o", color=colour, ms=11, zorder=3)
    ax.text(foot[0], foot[1] - 0.16, label, ha="center", fontsize=9, color=colour, fontweight="bold")
    return foot


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 6.0))

# ---- left: same commanded angles ----------------------------------------
fl = draw(axL, LONG, A1, A2, "#c1121f", "foot")
fs = draw(axL, SHORT, A1, A2, "#0466c8", "foot")
axL.annotate("", xy=(fl[0], fl[1]), xytext=(fs[0], fs[1]),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.6, ls=":"))
axL.text((fl[0] + fs[0]) / 2 + 0.30, (fl[1] + fs[1]) / 2,
         f"{np.linalg.norm(fl - fs):.2f} apart", fontsize=9.5, color="0.3", va="center")
axL.set_title("What we do now: fix the COMMAND\n"
              f"both bodies get the same angles ({np.degrees(A1):.0f}°, {np.degrees(A2):.0f}°)",
              fontsize=11, pad=10)
axL.text(0.5, 0.02, "angles identical  →  foot positions differ",
         transform=axL.transAxes, ha="center", fontsize=10.5, color="#9d0208", fontweight="bold")

# ---- right: same foot target --------------------------------------------
for L, colour in ((LONG, "#c1121f"), (SHORT, "#0466c8")):
    a1, a2 = inverse(L, TARGET)
    draw(axR, L, a1, a2, colour, "")
    lbl = "long" if L == LONG else "short"
    axR.text(-1.28, -0.30 if L == LONG else -0.48,
             f"{lbl}:  ({np.degrees(a1):6.1f}°, {np.degrees(a2):6.1f}°)",
             fontsize=10, color=colour, family="monospace", fontweight="bold")

axR.plot(*TARGET, "*", color="0.15", ms=20, zorder=4)
axR.text(TARGET[0] + 0.08, TARGET[1] - 0.02, "shared foot target", fontsize=9.5, color="0.15", va="center")
axR.set_title("IK retargeting: fix the FOOT TARGET\n"
              "solve each body for the angles that reach it", fontsize=11, pad=10)
axR.text(0.5, 0.02, "foot positions identical  →  angles differ",
         transform=axR.transAxes, ha="center", fontsize=10.5, color="#1b4332", fontweight="bold")

for ax in (axL, axR):
    ax.plot(0, 0, "s", color="0.3", ms=10)
    ax.text(0.06, 0.06, "hip", fontsize=9, color="0.3")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-2.55, 0.35)
    ax.set_aspect("equal")
    ax.axis("off")

fig.text(0.5, 0.965, "Same angle, different reach  ·  same reach, different angle",
         ha="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.015,
         "Foot position depends on joint angle AND link length. Holding the angles fixed therefore does "
         "not hold the behaviour fixed.\nIK retargeting inverts which one is held fixed, which is what "
         "makes the per-body commands differ while the behaviour stays the same.",
         ha="center", fontsize=9, color="0.25", linespacing=1.7)

fig.subplots_adjust(top=0.86, bottom=0.13, wspace=0.05)
fig.savefig(OUT, dpi=170)
print(f"saved: {OUT}")
