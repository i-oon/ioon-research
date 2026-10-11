"""The 4-leg slide's figure: few-shot curve beside the latent ablation.

Both panels read CSVs, which is the point. The previous version of this figure was composed by
hand, its source was never committed, and when the base-geometry build was superseded by the
held-out `c08f09t09` one the figure could not be regenerated -- only transcribed out of a terminal
scrollback. Anything that reaches a slide has to be rebuildable from a file.

  left   held-out error against the number of clips used to fit the new head, pretrained backbone
         against a random one. From sweep_4leg_fewshot.py.
  right  the same head refitted with the latent zeroed and shuffled. From fit_4leg_head.py
         --z_ablation. Shuffled is the control that matters: it keeps the latent's distribution and
         destroys its alignment, so beating it is evidence the *aligned* latent carries something.

  .venv/bin/python3 scripts/figures/plot_4leg_fewshot_and_ablation.py \\
      --curve results/wm/stage2/4leg_head/fewshot_curve_c08f09t09.csv \\
      --ablation results/wm/stage2/4leg_head/ik_4leg_c08f09t09_clean10_ablation.csv \\
      --out results/wm/stage2/figures/4leg_fewshot_and_z_ablation_c08f09t09.png
"""
import argparse
import collections
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAL, ORANGE, GREY = "#2a9d8f", "#e76f51", "#8d99ae"
ABLATION_LABELS = {
    "pretrained_stage2": "real\naligned z",
    "pretrained_zero_z": "zero z",
    "pretrained_shuffled_z": "shuffled z",
    "random_backbone": "random\nbackbone",
}


def read_curve(path):
    by = collections.defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by[(int(r["budget"]), r["model"])].append(float(r["test_deg"]))
    budgets = sorted({b for b, _ in by})
    out = {}
    for model in ("pretrained_stage2", "random_backbone"):
        mean = [st.mean(by[(b, model)]) for b in budgets]
        # standard deviation, not standard error: three seeds is too few for the latter to mean
        # anything, and the spread is the honest thing to show
        sd = [st.stdev(by[(b, model)]) if len(by[(b, model)]) > 1 else 0.0 for b in budgets]
        out[model] = (mean, sd)
    return budgets, out


def read_ablation(path):
    with open(path) as fh:
        rows = {r["model"]: float(r["test_deg"]) for r in csv.DictReader(fh)}
    order = [k for k in ABLATION_LABELS if k in rows]
    return [ABLATION_LABELS[k] for k in order], [rows[k] for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    budgets, curve = read_curve(os.path.join(ROOT, args.curve))
    labels, values = read_ablation(os.path.join(ROOT, args.ablation))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    for model, colour, name in (("pretrained_stage2", TEAL, "pretrained Stage 2"),
                                ("random_backbone", ORANGE, "random backbone")):
        mean, sd = curve[model]
        ax1.errorbar(budgets, mean, yerr=sd, marker="o", capsize=3, color=colour, label=name)
    ax1.set_xlabel("4-leg clips used to fit the new head")
    ax1.set_ylabel("held-out error (deg / joint)")
    ax1.set_xticks(budgets)
    ax1.set_title("Few-shot calibration", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    # the pretrained bar coloured like its curve, the random bar like its own, the two ablations
    # grey: they are neither condition, they are the same backbone with the latent damaged
    colours = [TEAL] + [GREY] * (len(values) - 2) + [ORANGE]
    bars = ax2.bar(range(len(values)), values, color=colours[:len(values)])
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
                 ha="center", fontsize=9)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("held-out error (deg / joint)")
    ax2.set_ylim(0, max(values) * 1.18)
    ax2.set_title("Is the latent doing work? (5 clips)", fontsize=11)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
