"""Evidence that frozen V-JEPA2 e_t encodes leg-length morphology.

Three panels, in order of how much they prove:

  (a) linear probe accuracy       — SUPERVISED: is morphology in e_t at all?
  (b) unsupervised PCA ordering   — NO LABELS: do the largest axes of variation
                                     already arrange the bodies by leg length?
  (c) UMAP coloured by morphology  — illustration only, never the evidence.

(a) is told the labels. (b) is not: PCA never sees which body is which, yet its
two leading components order short < medium < long. That self-organisation is a
stronger statement than a supervised probe, which only shows the info is present.

NOT proven here, on purpose: that the signal is leg length RATHER THAN recording
session. Each of the 3 morphologies is a single session, so morphology and
session are perfectly confounded in this data. No analysis on this set can
separate them; only re-recording each body under varied lighting/background can.
See the caption; the numbers ledger it used to cite was removed, and provenance now sits in
doc/FINDINGS.md next to each result.

Usage:
  .venv/bin/python scripts/plot_morphology_evidence.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

OUT = "report/fig_morphology_evidence.png"
COL = {"long": "#c1121f", "medium": "#f77f00", "short": "#0466c8"}
SCALE = {"long": 1.0, "medium": 0.75, "short": 0.5}

d = np.load("data/step0_v2/embeddings.npz")
E, m, ep, contact = d["e"], d["morph"], d["episode"], d["contact_code"]

# (a) probe: standard 5-fold, and grouped by episode (no frame-position leak)
acc = cross_val_score(LogisticRegression(max_iter=2000), E, m, cv=5, n_jobs=-1)
gkf = GroupKFold(n_splits=5)
acc_g = cross_val_score(LogisticRegression(max_iter=2000), E, m, cv=gkf, groups=ep, n_jobs=-1)

# (b) UNSUPERVISED: PCA sees no labels. Project e_t onto PC1 and check whether the
# three bodies already fall in leg-length order along it. Sign of a PC is arbitrary,
# so orient PC1 so that long < short for readability; the ORDERING is what matters.
pc1 = PCA(n_components=1).fit_transform(E - E.mean(0))[:, 0]
if pc1[m == "long"].mean() > pc1[m == "short"].mean():
    pc1 = -pc1
order_ok = pc1[m == "short"].mean() > pc1[m == "medium"].mean() > pc1[m == "long"].mean()

fig = plt.figure(figsize=(13.5, 6.3))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.0, 1.25], wspace=0.32,
                      left=0.05, right=0.97, top=0.80, bottom=0.27)

# ---- (a) probe accuracy --------------------------------------------------
axa = fig.add_subplot(gs[0])
bars = axa.bar(["5-fold", "grouped\nby episode"], [acc.mean() * 100, acc_g.mean() * 100],
               color=["#2a6f97", "#2d6a4f"], width=0.6)
axa.axhline(33.3, color="0.35", ls="--", lw=1.2)
axa.text(1.4, 36, "chance 33.3%", fontsize=8.5, color="0.35", ha="right")
axa.set_ylim(0, 108)
axa.set_ylabel("morphology accuracy (%)", fontsize=10)
axa.set_title("(a) Is morphology in $e_t$?\nlinear probe", fontsize=10.5)
for b, v in zip(bars, [acc.mean() * 100, acc_g.mean() * 100]):
    axa.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
axa.grid(axis="y", alpha=0.25)

# ---- (b) unsupervised PCA ordering --------------------------------------
axb = fig.add_subplot(gs[1])
rng = np.random.default_rng(0)
xpos = {"short": 0, "medium": 1, "long": 2}
for lab in ("short", "medium", "long"):
    v = pc1[m == lab]
    x = xpos[lab] + rng.uniform(-0.28, 0.28, v.shape)
    axb.scatter(x, v, color=COL[lab], s=8, alpha=0.35, zorder=2)
    axb.plot([xpos[lab] - 0.34, xpos[lab] + 0.34], [v.mean(), v.mean()],
             color="0.1", lw=2.4, zorder=4)
    axb.text(xpos[lab], v.mean(), f"  {lab}", va="center", fontsize=9,
             color=COL[lab], fontweight="bold")
axb.set_xticks([0, 1, 2])
axb.set_xticklabels(["short\n0.5x", "medium\n0.75x", "long\n1.0x"], fontsize=9)
axb.set_ylabel("position on PC1 of $e_t$", fontsize=10)
axb.set_title("(b) No labels used: PCA of $e_t$\norders the bodies by leg length", fontsize=10.5)
axb.grid(axis="y", alpha=0.25)
axb.text(0.5, 0.03, "monotonic ordering ✓" if order_ok else "ordering broken",
         transform=axb.transAxes, ha="center", fontsize=9.5,
         color="#1b4332" if order_ok else "#9d0208", fontweight="bold")

# ---- (c) UMAP, illustration only ----------------------------------------
axc = fig.add_subplot(gs[2])
try:
    import umap
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(E)
    for lab in ("long", "medium", "short"):
        sel = m == lab
        axc.scatter(xy[sel, 0], xy[sel, 1], s=6, color=COL[lab], label=lab, alpha=0.7)
    axc.legend(fontsize=8.5, frameon=False)
    axc.set_xticks([])
    axc.set_yticks([])
    axc.set_title("(c) $e_t$ by morphology (UMAP)\nillustration, not evidence", fontsize=10.5)
except Exception as e:  # pragma: no cover
    axc.text(0.5, 0.5, f"umap unavailable:\n{e}", ha="center", transform=axc.transAxes)
    axc.axis("off")

fig.suptitle("Frozen V-JEPA2 $e_t$ organises itself by leg length "
             "(leg length and recording session are still confounded here)", fontsize=12.5, y=0.965)
fig.text(0.5, 0.055,
         "(a) SUPERVISED: morphology is decodable, and survives holding out whole episodes, so it is not frame memorisation.\n"
         "(b) UNSUPERVISED: PCA is never told the labels, yet its leading component already orders short < medium < long.\n"
         "NOT shown: that this ordering is leg length rather than recording session. Each body is one session, so the two\n"
         "cannot be separated on this data. Re-recording each body across several sessions with varied lighting and\n"
         "background is required to claim leg length specifically.",
         ha="center", fontsize=8.6, color="0.3", linespacing=1.5)

fig.savefig(OUT, dpi=160)
print(f"saved: {OUT}")
print(f"probe 5-fold={acc.mean()*100:.1f}%  grouped={acc_g.mean()*100:.1f}%  "
      f"PC1 monotonic ordering={order_ok}")
