"""Slide 16's curve: how many clips of a new robot adapt the forward model, and to what horizon.

Four panels, one per rollout horizon. Both arms are scored against **holding the frame still on
the same clips**, so 1.0 is the line that matters: below it the rollout is worse than predicting no
motion and cannot support planning however low its training loss went.

**The horizon is in seconds, not steps, and that is why the panel titles carry both.** The earlier
version of this measurement used `data/allocentric/fwd_b1_50hz` at 20 ms per stored transition against the
insect's 50 ms (F74), so its h=10 spanned 0.2 s where this one spans 0.5 s. The two curves are not
comparable at matched step counts and this figure replaces rather than extends the old one.

  .venv/bin/python3 scripts/figures/plot_ftm_fewshot.py
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAL, GREY, INK = "#2a9d8f", "#adb5bd", "#22333b"
DT = 0.05  # seconds per stored transition, both robots, after the F74 fix
# Two pretraining sets that differ in *which kind* of variety they carry, drawn together because
# the point of the figure is that they land on top of each other.
ARMS = (("beh12_hexonly", TEAL, "insects: 1 body, 12 behaviours"),
        ("m3d_body", "#4c6ef5", "insects: 4 bodies, 1 behaviour"),
        ("scratch", "#e76f51", "random init, same budget"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/wm/stage2/measurements/ftm_fewshot_beh12.csv")
    ap.add_argument("--out", default="results/wm/stage2/figures/ftm_fewshot_beh12.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, args.csv))))
    horizons = [h for h in ("h1", "h3", "h5", "h10") if h in rows[0]]

    fig, axes = plt.subplots(1, len(horizons), figsize=(15.5, 4.2), sharey=True)
    for ax, h in zip(axes, horizons):
        steps = int(h[1:])
        for arm, colour, label in ARMS:
            pts = sorted((int(r["clips"]), float(r[h]), float(r[h + "_lo"]), float(r[h + "_hi"]))
                         for r in rows if r["backbone"] == arm)
            x = [p[0] for p in pts]
            ax.plot(x, [p[1] for p in pts], "-o", color=colour, lw=2, ms=5, label=label, zorder=3)
            ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts],
                            color=colour, alpha=0.16, lw=0, zorder=1)
        # the only line with a meaning: below it the model loses to predicting no motion at all
        ax.axhline(1.0, color=INK, ls="--", lw=1.1, zorder=2)
        ax.set_title(f"h = {steps}   ({steps * DT:.2f} s)", fontsize=11, color=INK)
        ax.set_xlabel("clips of the new robot", fontsize=9.5, color=INK)
        ax.set_xticks(x)
        ax.grid(alpha=0.18, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("rollout vs holding the frame still", fontsize=9.5, color=INK)
    axes[0].text(1.15, 1.02, "beats doing nothing", fontsize=8, color=INK, va="bottom")
    axes[0].legend(fontsize=9, frameon=False, loc="lower right")

    fig.suptitle("Insect pretraining buys a one-step forward model on a quadruped and does not "
                 "survive the rollout.\nWhich kind of variety it was pretrained on makes no "
                 "difference at all — the two teal-and-blue curves are separate runs.",
                 fontsize=12, color=INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
