"""Sanity-check figure: does frozen e_t contain useful locomotion information?

Three conditions, one classifier (logistic regression on e_t -> foot-contact
pattern, top-8 patterns, macro-F1, 5-fold CV):

  within-body  train and test on the SAME body      -> is contact decodable?
  shuffle      same, but labels permuted             -> is it real signal?
  cross-body   train on body A, test on body B        -> does it transfer?

The story: contact is strongly decodable within a body, collapses to chance
under shuffling (so it is real), and drops sharply across bodies (so the
representation is body-specific -- which is exactly what the latent action is
meant to fix).

Pilot diagnostic, not final evaluation: contact threshold 0.5 N, gait is a
wave/staggered pattern (not tripod), single session per body.

Usage:
  .venv/bin/python scripts/plot_sanity_check.py
"""
import collections
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "report/fig_sanity_check.png"
BODIES = ["long", "medium", "short"]
COL = {"long": "#c1121f", "medium": "#f77f00", "short": "#0466c8"}
CHANCE = 1 / 8  # top-8 contact patterns


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))


d = np.load("data/step0_v2/embeddings.npz")
E, m, F = d["e"], d["morph"], d["forces"]
code = (F > 0.5).astype(int).dot(1 << np.arange(6))
top8 = [c for c, _ in collections.Counter(code.tolist()).most_common(8)]
keep = np.isin(code, top8)
Ek, ck, mk = E[keep], code[keep], m[keep]

SHORT = {"long": "L", "medium": "M", "short": "S"}

rng = np.random.default_rng(0)
within, shuffle = {}, {}
for b in BODIES:
    s = mk == b
    within[b] = cross_val_score(clf(), Ek[s], ck[s], cv=5, scoring="f1_macro", n_jobs=-1).mean()
    shuffle[b] = cross_val_score(clf(), Ek[s], rng.permutation(ck[s]), cv=5,
                                 scoring="f1_macro", n_jobs=-1).mean()
cross = []  # (bar_label, value, train_body_colour)
for a in BODIES:
    for b in BODIES:
        if a != b:
            pred = clf().fit(Ek[mk == a], ck[mk == a]).predict(Ek[mk == b])
            cross.append((f"{SHORT[a]}→{SHORT[b]}", f1_score(ck[mk == b], pred, average="macro"), a))

fig, ax = plt.subplots(figsize=(11, 6.2))

# Only TWO real bar groups now. Shuffle is demoted to a background reference
# band (it is a control, not a third result), which is where the confusion was.
shuffle_mean = np.mean([shuffle[b] for b in BODIES])

groups = [
    ("Within body\ntrain & test SAME body",
     [(SHORT[b], within[b], COL[b]) for b in BODIES]),
    ("Cross body\ntrain one body → test another",
     [(lab, v, COL[a]) for lab, v, a in cross]),
]

# reference band: shuffled-label floor (which coincides with chance)
ax.axhspan(0, max(shuffle_mean, CHANCE), color="0.88", zorder=0)
ax.axhline(shuffle_mean, color="0.45", ls="--", lw=1.2, zorder=1)
ax.text(0.15, max(shuffle_mean, CHANCE) + 0.015,
        "floor: shuffled labels = %.2f  ≈  chance %.3f  (nothing below this means anything)"
        % (shuffle_mean, CHANCE),
        fontsize=8.5, color="0.35", style="italic")

x = 0
xticks, xlabels = [], []
for title, bars in groups:
    xs = np.arange(len(bars)) + x
    vals = [v for _, v, _ in bars]
    ax.bar(xs, vals, color=[c for _, _, c in bars], width=0.82, edgecolor="white",
           linewidth=0.6, zorder=3)
    for xi, (lab, v, _) in zip(xs, bars):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, zorder=4)
        ax.text(xi, -0.045, lab, ha="center", fontsize=9.5, fontweight="bold")
    ax.plot([xs[0] - 0.45, xs[-1] + 0.45], [np.mean(vals)] * 2, color="0.1", lw=2, zorder=5)
    ax.text(xs.mean(), np.mean(vals) + 0.055, f"mean {np.mean(vals):.2f}", ha="center",
            fontsize=10.5, fontweight="bold")
    xticks.append(xs.mean())
    xlabels.append(title)
    x += len(bars) + 1.4

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, fontsize=11)
ax.tick_params(axis="x", length=0, pad=22)
ax.set_ylabel("foot-contact decodability\n(macro-F1)", fontsize=11)
ax.set_ylim(-0.02, 1.0)
ax.set_xlim(-0.7, x - 1.0)
ax.set_title("Does frozen $e_t$ contain useful locomotion information?", fontsize=13, pad=12)
ax.grid(axis="y", alpha=0.25)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[b]) for b in BODIES]
ax.legend(handles, [f"{SHORT[b]} = {b}" for b in BODIES], title="bar colour = training body",
          fontsize=9, title_fontsize=9, loc="upper right", frameon=True, framealpha=0.9)

within_mean = np.mean([within[b] for b in BODIES])
cross_mean = np.mean([v for _, v, _ in cross])
fig.text(0.5, 0.015,
         "Bars are macro-F1 of a logistic-regression probe: input = e_t, target = which feet are planted. "
         "L/M/S = long/medium/short.\n"
         "Contact is strongly decodable within a body (mean %.2f), and drops sharply across bodies "
         "(mean %.2f, e.g. L→S = train on long, test on short)\n— the information is present but "
         "body-specific, which is exactly what the latent action must fix. The grey band is the "
         "shuffled-label floor: any\nbar inside it is indistinguishable from noise. "
         "Pilot diagnostic: threshold 0.5 N, wave gait (not tripod), one session per body."
         % (within_mean, cross_mean),
         ha="center", fontsize=8.4, color="0.3", linespacing=1.5)

fig.subplots_adjust(bottom=0.30, top=0.91)
fig.savefig(OUT, dpi=160)
print(f"saved: {OUT}")
print(f"within={within_mean:.3f}  cross={cross_mean:.3f}  shuffle={np.mean([shuffle[b] for b in BODIES]):.3f}")
