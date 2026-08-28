"""The adaptation objective decides cross-embodiment selection: MSE lands on chance, InfoNCE does not.

Both arms are `wm/adapt3.py` on the same 24 B1 clips, the same architecture and the same code path;
only `--lambda_nce` differs, and the MSE arm ran 15,000 steps against the contrastive arm's 12,000.
The goal frames come from a **hexapod** clip and the candidates are B1 clips, so only the goal
crosses embodiments (F107).

**Every point is a separate recorded goal clip, not a rerun.** The B1 loop is MuJoCo and repeats
bit for bit (F105), so repeating a configuration returns the identical number and carries no
information; the spread that means something is across the four clips recorded for one condition.

**Chance is 33%, not 1/12.** The candidate library holds twelve conditions in unequal families --
four speed, four turn, two per sideways direction -- so a planner picking uniformly at random lands
on the right *family* far more often than 8% (F98).

  .venv/bin/python3 scripts/figures/plot_adapt_objective.py
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAL, SAND, INK, GREY = "#2a9d8f", "#e9c46a", "#22333b", "#adb5bd"

PANELS = (
    ("forward", "speed_c5.8", "speed",
     ("hexapod_ep0", "hexapod_ep1", "hexapod_ep2", "hexapod_ep3")),
    ("turning", "turn_s0.56", "turn",
     ("hexapod_ep1300", "hexapod_ep1301", "hexapod_ep1302", "hexapod_ep1303")),
)
ARMS = (("MSE", "spread_rung3mse", SAND), ("+ InfoNCE", "spread_rung3nce", TEAL))
CHANCE = 1 / 3


def family(condition):
    return condition.rsplit("_", 1)[0] if "_" in condition else condition


def rate(run_dir, goal, want):
    """Fraction of *planned* steps that chose a candidate from the goal's behaviour family."""
    hits = glob.glob(os.path.join(ROOT, "results/wm/closed_loop", run_dir, f"*__goal_{goal}.npz"))
    if not hits:
        return None
    with np.load(hits[0], allow_pickle=True) as z:
        chosen = np.asarray(z["chosen"], str)
    # warm-start steps replay a recorded clip rather than being planned, and are excluded here for
    # the same reason `score_closed_loop.py` excludes them (F110)
    warm = int(np.sum(np.char.startswith(chosen, "warm:")))
    planned = chosen[warm:]
    return float(np.mean([family(c) == want for c in planned]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wm/closed_loop/figures/adapt_objective.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    rng = np.random.default_rng(0)          # jitter only, never touches a measured value

    for ax, (title, condition, want, goals) in zip(axes, PANELS):
        ax.axhline(CHANCE, color=GREY, lw=1.1, ls=(0, (5, 3)), zorder=1)
        for i, (label, run_dir, colour) in enumerate(ARMS):
            vals = [v for v in (rate(run_dir, g, want) for g in goals) if v is not None]
            if not vals:
                continue
            vals = np.asarray(vals)
            ax.errorbar(i, vals.mean(), yerr=vals.std(), color=INK, lw=1.4,
                        capsize=5, capthick=1.4, zorder=3)
            ax.plot(i, vals.mean(), "o", ms=9, mfc=colour, mec=INK, mew=1.2, zorder=4)
            ax.plot(i + rng.uniform(-.07, .07, len(vals)), vals, "o", ms=4.5,
                    mfc="white", mec=colour, mew=1.3, ls="none", zorder=2)
            # value label above the marker, so it never collides with the chance line's label
            ax.annotate(f"{vals.mean():.0%}", (i, vals.mean() + vals.std()),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        color=INK, fontsize=9.5, fontweight="semibold")
        ax.set_xlim(-.55, len(ARMS) - .45)
        ax.set_xticks(range(len(ARMS)), [a[0] for a in ARMS])
        ax.set_title(title, fontsize=10.5, color=INK, pad=14)
        ax.text(.5, 1.015, condition, transform=ax.transAxes, ha="center",
                fontsize=8.5, color=GREY, family="monospace")
        ax.tick_params(colors=INK, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GREY)

    axes[0].set_ylim(0, 1)
    axes[0].set_yticks(np.arange(0, 1.01, .25), ["0", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("planned steps on the goal's\nbehaviour family", fontsize=9, color=INK)
    axes[0].annotate("chance", (-.5, CHANCE), textcoords="offset points", xytext=(2, 4),
                     ha="left", color=GREY, fontsize=8.5)
    # the turning panel's conclusion is that nothing clears the line, which is easy to miss
    axes[1].annotate("neither clears chance", (.5, .62), ha="center", color=GREY, fontsize=8.5)
    fig.suptitle("Adaptation objective, not adaptation itself, is what crosses embodiments",
                 fontsize=11, color=INK, y=1.0)
    fig.text(.5, -.03, "hexapod goal frames, B1 candidates and body, MuJoCo physics; "
                       "one point per recorded goal clip, bars are $\\pm$1 s.d. (n=4)",
             ha="center", fontsize=8, color=GREY)
    fig.tight_layout()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
