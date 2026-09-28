"""Rollout-horizon figure in the thesis style.

The per-horizon ratios were read back from the earlier version of this figure (the measuring run is not
stored), so they are accurate to about +/-0.002.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
report_style.apply()
T = report_style.TOL
k = [1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 50, 60]
r = [0.621, 0.614, 0.633, 0.678, 0.738, 0.804, 0.779, 0.830, 0.837, 0.851, 0.850, 0.874]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.axhline(1.0, color=T["red"], ls="--", lw=1.4, label="copy-forward baseline (ratio 1.0)")
ax.axvspan(15, 25, color=T["green"], alpha=0.15, lw=0, label="usable imagination horizon (ratio 0.78 to 0.83)")
ax.axvline(100, color="#888888", ls=":", lw=1.2)
ax.text(98, 1.03, "discount horizon (about 100 steps)", ha="right", va="bottom", fontsize=8.5, color="#555555")
ax.plot(k, r, color=T["blue"], marker="o", ms=4.5)
ax.set_xlim(0, 105)
ax.set_ylim(0.5, 1.1)
ax.set_xlabel("auto-regressive rollout steps ahead, $k$")
ax.set_ylabel("model MSE / copy-forward MSE")
ax.legend(loc="lower right", bbox_to_anchor=(0.93, 0.0))
out = os.path.join(ROOT, "report/image/rollout_horizon_accuracy.png")
fig.savefig(out)
print("saved", out)
