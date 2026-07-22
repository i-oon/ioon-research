"""One command, three bodies, three different physical states.

All values are read from the recorded episodes at the SAME timestep. The joint
command is bit-identical across the three morphologies (verified by assert), so
everything that differs below is consequence, not input.

This is the figure that answers "what does the latent action have to represent?"
The internal channel is identical across bodies and therefore cannot distinguish
them. Whatever separates these three cases is visible only from outside.

Usage:
  .venv/bin/python scripts/plot_same_command.py
"""
import matplotlib.pyplot as plt
import numpy as np

STEP = 25
THRESH = 0.5  # matches CONTACT_THRESH in step0_encode.py (lowered from 3.0 for physical honesty)
BODIES = [("long", "1.0x", "#c1121f"), ("medium", "0.75x", "#f77f00"), ("short", "0.5x", "#0466c8")]
OUT = "report/fig_same_command.png"

D = {b: np.load(f"data/step0_v2/{b}_ep0.npz") for b, _, _ in BODIES}
order = [n.decode() if isinstance(n, bytes) else n for n in D["long"]["foot_order"]]
q = D["long"]["actions"][STEP]
for b, _, _ in BODIES:
    assert np.array_equal(D[b]["actions"][STEP], q), f"{b} command differs; figure premise broken"

fig = plt.figure(figsize=(12.6, 8.0))
gs = fig.add_gridspec(3, 3, height_ratios=[0.60, 1.35, 0.85], hspace=0.34, wspace=0.22,
                      top=0.90, bottom=0.175, left=0.06, right=0.97)

fig.text(0.5, 0.965, "SAME COMMAND, THREE BODIES", ha="center", fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.928, f"data/step0_v2/*_ep0.npz, step {STEP}", ha="center",
         fontsize=9, color="0.4", style="italic")

# ---- the shared command, printed once ------------------------------------
axq = fig.add_subplot(gs[0, :])
axq.axis("off")
axq.text(0.5, 0.72, "  ".join(f"{v:.2f}" for v in q), ha="center", va="center",
         family="monospace", fontsize=10.5, transform=axq.transAxes)
axq.text(0.5, 0.28, r"$q_t \in \mathbb{R}^{18}$  —  bit-identical for all three bodies "
                    "(max pairwise difference 0.000000)",
         ha="center", va="center", fontsize=10, color="#9d0208", transform=axq.transAxes)

# ---- frames + forces -----------------------------------------------------
for col, (body, label, colour) in enumerate(BODIES):
    F = D[body]["forces"][STEP]
    contact = F > THRESH

    axi = fig.add_subplot(gs[1, col])
    axi.imshow(D[body]["frames"][STEP])
    axi.set_xticks([])
    axi.set_yticks([])
    for s in axi.spines.values():
        s.set_edgecolor(colour)
        s.set_linewidth(2.2)
    axi.set_title(f"{body}  ({label})", fontsize=11.5, fontweight="bold", color=colour, pad=6)

    axf = fig.add_subplot(gs[2, col])
    bars = axf.bar(range(6), F, color=[colour if c else "0.82" for c in contact], width=0.66)
    axf.axhline(THRESH, color="0.35", lw=1.1, ls="--")
    axf.text(5.55, THRESH + 0.4, f"{THRESH:g} N", fontsize=7.5, color="0.35", ha="right")
    axf.set_xticks(range(6))
    axf.set_xticklabels([n.lstrip("_") for n in order], fontsize=8.5)
    axf.set_ylim(0, 11)
    axf.set_ylabel("foot force (N)", fontsize=9)
    axf.grid(axis="y", alpha=0.25)
    axf.set_title(f"{contact.sum()} feet in contact", fontsize=9.5, color="0.25", pad=4)
    # Call out the left-middle foot: planted on long and short, airborne on medium.
    bars[1].set_edgecolor("0.1")
    bars[1].set_linewidth(1.6)

fig.text(0.5, 0.045,
         "The left-middle foot (outlined) carries 5.7 N on the long body and 9.3 N on the short body, "
         "but only 0.3 N on the medium body, where it is still airborne.\n"
         "Identical input, three different physical states. Nothing in the command distinguishes them; "
         "the difference exists only in the world, where the camera sees it.",
         ha="center", fontsize=9, color="0.2", linespacing=1.7)

fig.savefig(OUT, dpi=170)
print(f"saved: {OUT}")
