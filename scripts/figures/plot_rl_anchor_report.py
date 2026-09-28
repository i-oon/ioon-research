"""Return curve of the imagination-RL actor on the fixed start and goal, in the thesis style."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
report_style.apply()
d = np.load(os.path.join(ROOT, "results/wm/closed_loop/rl_imagination_isolation_v2_h12_anchor.npz"),
            allow_pickle=True)
it, ret = d["eval_iter"], d["eval_realized"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.axhline(0, color="#888888", lw=0.8)
ax.plot(it, ret, color=report_style.TOL["blue"], marker="o", ms=3.5)
ax.set_xlabel("training iteration")
ax.set_ylabel("return under the model's Froude reading")
ax.annotate(f"{ret[0]:.1f}", (it[0], ret[0]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=9)
ax.annotate(f"{ret[-1]:.1f}", (it[-1], ret[-1]), textcoords="offset points", xytext=(-4, -12), ha="right", fontsize=9)
out = os.path.join(ROOT, "report/image/rl_anchor_convergence.png")
fig.savefig(out)
print("saved", out, ret[0], ret[-1])
