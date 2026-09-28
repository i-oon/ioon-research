"""Summary of the physics closed loops (`physics_loops.sh`, `physics_random.sh`): mean L2 error to the
goal Froude per model x body x selector x execution, per condition, with the random-candidate
reference and fall counts; a bar figure per body.

    .venv/bin/python3 scripts/figures/physics_report.py
"""
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COND = ["turn_s0.29", "turn_s0.56", "side_L_lvl0", "side_R_lvl1", "speed_c7.1", "speed_c8.8"]


def main():
    src = os.path.join(ROOT, "results/wm/closed_loop/physics/summary.txt")
    R, F = defaultdict(dict), defaultdict(int)
    for line in open(src):
        m = re.match(r"(\S+) (\S+) (\S+) (\S+) (\S+) \| .*mean L2 error ([\d.]+)", line)
        if not m:
            continue
        body, model, mech, ex, cond, err = m.groups()
        R[(body, model, mech, ex)][cond] = float(err)
        if "fell" in line:
            F[(body, model, mech, ex)] += 1
    print(f"{'body':<5}{'model':<7}{'selector':<9}{'execution':<14}" + "".join(f"{c:>12}" for c in COND)
          + f"{'mean':>9}{'falls':>7}")
    for k in sorted(R):
        v = R[k]
        print(f"{k[0]:<5}{k[1]:<7}{k[2]:<9}{k[3]:<14}" + "".join(f"{v.get(c, np.nan):>12.3f}" for c in COND)
              + f"{np.nanmean([v.get(c, np.nan) for c in COND]):>9.3f}{F[k]:>7}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ROOT, "results/deck/physics_closed_loop")
    os.makedirs(out, exist_ok=True)
    for body in sorted({k[0] for k in R}):
        keys = sorted(k for k in R if k[0] == body)
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(keys)), 4.5))
        means = [np.nanmean([R[k].get(c, np.nan) for c in COND]) for k in keys]
        cols = ["#9e9e9e" if k[2] == "random" else "#2e7d32" if k[2] == "direct" else "#1565c0" for k in keys]
        ax.bar(range(len(keys)), means, color=cols)
        for i, v in enumerate(means):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{k[1]}\n{k[2]}\n{k[3]}" for k in keys], fontsize=7)
        ax.set_ylabel("mean L2 error to goal Froude (lower is better)")
        ax.set_title(f"Physics closed loop, {body}: green direct (state-blind), blue rollout (state-aware), grey random")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"summary_{body}.png"), dpi=110); plt.close(fig)
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
