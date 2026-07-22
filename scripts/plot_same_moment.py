"""One physical moment, read two ways.

Unlike fig_obs_format.py this is NOT a schematic. The frame, the 18 joint
targets and the foot forces are all taken from the same recorded timestep, so
the figure shows what each channel actually contained at that instant.

Honesty note. The internal column is q_t, the 18 joint targets, which is what
this pipeline uses as a_t. It is NOT "proprioception in general": a real robot
also carries force and IMU, and those DO report contact and world motion. The
claim here is narrower and still true — from joint angles alone, foot position
needs link lengths to recover, and contact and world motion are not recoverable
at all. The force column is printed alongside precisely so this is not hidden.

Usage:
  .venv/bin/python scripts/plot_same_moment.py
"""
import matplotlib.pyplot as plt
import numpy as np

EP = "data/step0_v2/long_ep0.npz"
STEP = 25          # a verified no-contact -> contact transition for the ML foot
FOOT = 1           # index of _ML in foot_order
THRESH = 3.0       # N, matches CONTACT_THRESH in step0_encode.py
OUT = "report/fig_same_moment.png"

d = np.load(EP)
frame, q, F, order = d["frames"][STEP], d["actions"][STEP], d["forces"][STEP], d["foot_order"]
prev = d["forces"][STEP - 1]
assert prev[FOOT] <= THRESH < F[FOOT], "STEP is not a touchdown for this foot"

fig = plt.figure(figsize=(12.4, 7.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.12,
                      top=0.86, bottom=0.24, left=0.05, right=0.97)

fig.text(0.5, 0.965, "SAME PHYSICAL MOMENT", ha="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.923, f'"Left-middle foot enters stance"   ·   {EP.split("/")[-1]}, step {STEP}',
         ha="center", fontsize=9.5, color="0.35", style="italic")

# ---- left: internal -----------------------------------------------------
axl = fig.add_subplot(gs[0])
axl.axis("off")
axl.set_xlim(0, 1)
axl.set_ylim(0, 1)
axl.text(0.5, 0.93, "INTERNAL OBSERVATION", ha="center", fontsize=11.5,
         fontweight="bold", color="#9d0208")

rows = ["".join(f"{v:>7.2f}" for v in q[i:i + 6]) for i in range(0, 18, 6)]
for i, r in enumerate(rows):
    axl.text(0.5, 0.79 - i * 0.075, r, ha="center", family="monospace", fontsize=11.5)
axl.text(0.5, 0.53, r"$q_t \in \mathbb{R}^{18}$", ha="center", fontsize=12)

axl.text(0.5, 0.41, "directly readable", ha="center", fontsize=9, color="0.45", style="italic")
for i, t in enumerate(["joint configuration", "body-relative coordinates", "known joint ordering"]):
    axl.text(0.5, 0.33 - i * 0.062, t, ha="center", fontsize=10.2, color="0.15")

axl.text(0.5, 0.12, "foot position needs link lengths;\n"
                    "contact and world motion are not recoverable from angles alone",
         ha="center", fontsize=8.7, color="#9d0208", style="italic", linespacing=1.5)

# ---- right: external ----------------------------------------------------
axr = fig.add_subplot(gs[1])
axr.imshow(frame)
axr.set_xticks([])
axr.set_yticks([])
for s in axr.spines.values():
    s.set_edgecolor("0.45")
axr.set_title("EXTERNAL OBSERVATION", fontsize=11.5, fontweight="bold", color="#1b4332", pad=9)
axr.set_xlabel(r"$o_t \in \mathbb{R}^{256 \times 256 \times 3}$", fontsize=11, labelpad=6)
axr.text(0.5, -0.125, "body shape · foot position · contact consequence · motion in the world",
         transform=axr.transAxes, ha="center", fontsize=9.4, color="0.15")

# ---- measured forces, so the caveat is visible not buried ---------------
fig.text(0.5, 0.065,
         "Measured foot force at this step (N):  " +
         "   ".join(f"{n.decode() if isinstance(n, bytes) else n}={v:.1f}"
                    for n, v in zip(order, F)) +
         f"\nForce and IMU do report contact and world motion. The contrast above is with joint angles, "
         f"which is what this pipeline conditions on.",
         ha="center", fontsize=8.3, color="0.4", linespacing=1.6)

fig.savefig(OUT, dpi=170)
print(f"saved: {OUT}")
