"""Step -1 figure: does a morphology gap exist?

Two panels:
  left  — top-down body trajectories, all episodes, translated to a common origin
          and rotated so each episode's initial heading points along +x. Rotating
          is what makes heading drift readable as deviation from the x-axis.
  right — per-episode path length and net displacement, so the complete
          separation between bodies is visible without a summary statistic.

Usage:
  .venv/bin/python scripts/plot_step_minus1.py
"""
import glob

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "data/step0_v2"
BODIES = [("long", "1.0x", "#c1121f"), ("medium", "0.75x", "#f77f00"), ("short", "0.5x", "#0466c8")]
OUT = "report/fig_step_minus1.png"
HEADING_WINDOW = 10  # steps used to estimate initial heading


def load(body):
    """Return list of (N,2) xy trajectories, one per episode."""
    return [np.load(f)["head"][:, :2].astype(float) for f in sorted(glob.glob(f"{DATA_DIR}/{body}_ep*.npz"))]


def align(h):
    """Translate to origin, then rotate so the initial heading lies along +x."""
    h = h - h[0]
    v = h[HEADING_WINDOW]
    th = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return h @ R.T


def metrics(h):
    path = np.linalg.norm(np.diff(h, axis=0), axis=1).sum()
    net = np.linalg.norm(h[-1] - h[0])
    return path, net


def drift_deg(h):
    """Net heading drift: angle of the end point off the initial heading.

    Comparing *instantaneous* heading at start and end instead of this gives
    numbers 3-5x too large, because each instantaneous estimate is contaminated
    by the gait's side-to-side wobble.
    """
    a = align(h)
    return np.degrees(np.arctan2(a[-1, 1], a[-1, 0]))


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})

# ---- left: trajectories -------------------------------------------------
for body, label, colour in BODIES:
    for i, h in enumerate(load(body)):
        a = align(h)
        ax0.plot(a[:, 0], a[:, 1], color=colour, lw=1.4, alpha=0.75,
                 label=f"{body} ({label})" if i == 0 else None)
        ax0.plot(a[-1, 0], a[-1, 1], "o", color=colour, ms=5)
    d = [abs(drift_deg(h)) for h in load(body)]
    print(f"{body:8s} mean |drift| {np.mean(d):4.1f} deg, max {np.max(d):4.1f} deg")

ax0.axhline(0, color="0.75", lw=0.8, ls="--", zorder=0)
ax0.set_xlabel("distance along initial heading (m)")
ax0.set_ylabel("lateral deviation (m)")
ax0.set_title("Body trajectories, 5 episodes per morphology\n"
              "identical commands; aligned to a common start and heading", fontsize=10)
ax0.legend(frameon=False, fontsize=9, loc="lower left")
# Equal aspect is not cosmetic here. On a stretched y-axis a 0.4 m lateral
# deviation over 4.5 m of travel reads as violent swerving; at true scale the
# walks are visibly near-straight, which is what the drift numbers say.
ax0.set_aspect("equal", adjustable="box")
ax0.set_ylim(-1.6, 1.6)
ax0.grid(alpha=0.25)

# ---- right: per-episode distances, dots + mean + error bars -------------
# Raw episode dots rather than a summary: the separation and the long-body
# spread are both visible without asking the audience to read a table.
rng = np.random.default_rng(0)  # jitter only, never touches the values

for col, (body, label, colour) in enumerate(BODIES):
    P, N = np.array([metrics(h) for h in load(body)]).T
    for series, dx, filled in ((P, -0.16, True), (N, 0.16, False)):
        x = col + dx + rng.uniform(-0.035, 0.035, len(series))
        ax1.scatter(x, series, s=52, color=colour if filled else "none",
                    edgecolors=colour, linewidths=1.5, zorder=3)
        ax1.errorbar(col + dx, series.mean(), yerr=series.std(), fmt="_",
                     color="0.15", ms=26, mew=2.2, elinewidth=1.6, capsize=7, zorder=4)

    ax1.text(col - 0.16, P.max() + 0.22, f"{P.mean():.2f}", ha="center", fontsize=8.5, color=colour)
    ax1.text(col + 0.16, N.max() + 0.22, f"{N.mean():.2f}", ha="center", fontsize=8.5, color=colour)

ax1.set_xticks(range(len(BODIES)))
ax1.set_xticklabels([f"{b}\n{l}" for b, l, _ in BODIES])
ax1.set_ylabel("distance (m)")
ax1.set_title("Every episode plotted, mean ± sd\n"
              "filled = path length, open = net displacement", fontsize=10)
ax1.grid(axis="y", alpha=0.25)
ax1.set_xlim(-0.55, len(BODIES) - 0.45)
ax1.set_ylim(0, 6.1)

fig.suptitle("Step -1: identical joint commands produce different locomotion "
             "(H₀ rejected, p = 0.0079, Cliff's δ = 1.00 for all pairs)", fontsize=11.5)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print(f"saved: {OUT}")
