"""Single-frame command-recoverability error against frame offset, in the thesis style.

The values were read back from the earlier version of this figure (the run that measured them is not
reproducible here), so they are accurate to about +/-0.01 degree.
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
vals = [2.342, 2.25, 2.25, 2.318, 2.367, 2.315, 2.136, 1.906, 1.792, 1.898, 2.102, 2.221, 2.165, 2.056, 2.006, 2.139, 2.36, 2.492, 2.48, 2.338, 2.099, 1.989, 2.082, 2.197, 2.259, 2.17, 1.991, 1.899, 2.01, 2.197, 2.312, 2.273, 2.134, 2.02, 2.125, 2.358, 2.483, 2.45, 2.244, 2.011, 1.991, 2.157, 2.313, 2.361, 2.257, 2.048, 1.838, 1.74, 1.834, 2.008, 2.118, 2.1, 2.023, 1.9, 1.806, 1.829, 1.899, 1.916, 1.831]
h = list(range(len(vals)))
fig, ax = plt.subplots(figsize=(6.6, 3.3))
for m in range(0, len(vals), 19):
    ax.axvline(m, color="#888888", ls="--", lw=0.8)
ax.plot(h, vals, color=T["blue"], marker="o", ms=3)
ax.text(19.6, 1.72, "dashed: multiples of 19 frames", fontsize=8.5, color="#555555")
ax.set_xlabel("frame offset $h$")
ax.set_ylabel("command RMSE (deg)")
ax.set_ylim(1.68, 2.55)
out = os.path.join(ROOT, "report/image/periodicity_curve_fixed_58.png")
fig.savefig(out)
print("saved", out)
