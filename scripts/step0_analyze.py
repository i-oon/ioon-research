"""Step 0 — Visual Encoder Sanity Check, quantified.

Criteria per direction_plan.md (rewritten 2026-07-17). Step 0 asks about
INFORMATION PRESENCE, not invariance:

  Check 1 (THE GATE): is gait phase decodable from e_t? If not, the ITM has
      nothing to extract and the project is blocked.
  Check 2 (BASELINE, not a gate): how strongly does e_t separate morphology?
      Expected high -- a 0.5x leg genuinely looks different. This is the
      "before" that Step 1.5's z_t must beat.
  Check 3 (NOISE FLOOR): how much does e_t vary at the SAME gait phase, same
      morphology, differing only in world position (the camera tracks the robot,
      so the background slides)? This is irrelevant variation. Phase signal must
      exceed it to mean anything.

Metrics follow QWM App. F-E (silhouette + between/within-class variance) so the
numbers are directly comparable to the prior art.

Usage:
  python scripts/step0_analyze.py --emb data/step0/embeddings.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def variance_decomposition(X, labels):
    """Fraction of total variance that is BETWEEN-class (QWM App. F-E style)."""
    grand = X.mean(axis=0)
    total = ((X - grand) ** 2).sum()
    between = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        between += len(Xc) * ((Xc.mean(axis=0) - grand) ** 2).sum()
    return between / total


def report(name, X, labels, note=""):
    sil = silhouette_score(X, labels)
    bvar = variance_decomposition(X, labels)
    print(f"  {name:34s} silhouette={sil:+.4f}   between-class var={bvar*100:5.1f}%   {note}")
    return sil, bvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, default="data/step0/embeddings.npz")
    ap.add_argument("--out", type=str, default="step0_umap.png")
    args = ap.parse_args()

    d = np.load(args.emb)
    E, morph, phase_bin, episode = d["e"], d["morph"], d["phase_bin"], d["episode"]
    step_idx = d["step_idx"]
    print(f"e_t: {E.shape}   morphologies={sorted(set(morph))}   phase bins={len(set(phase_bin))}")

    Z = StandardScaler().fit_transform(E)

    # ---------------- Check 3: NOISE FLOOR ----------------
    print("\n=== CHECK 3 — noise floor (same phase, same morphology, different world position) ===")
    # frames at the SAME gait phase differ only in where the robot is / what floor is behind it
    same_phase_d, diff_phase_d = [], []
    for m in np.unique(morph):
        for ep in np.unique(episode):
            sel = (morph == m) & (episode == ep)
            if sel.sum() == 0:
                continue
            Xe, si = Z[sel], step_idx[sel]
            ph = si % 64
            for p in np.unique(ph):
                idx = np.where(ph == p)[0]
                if len(idx) >= 2:
                    for i in range(len(idx)):
                        for j in range(i + 1, len(idx)):
                            same_phase_d.append(np.linalg.norm(Xe[idx[i]] - Xe[idx[j]]))
            rng = np.random.default_rng(0)
            for _ in range(300):
                i, j = rng.integers(0, len(Xe), 2)
                if ph[i] != ph[j]:
                    diff_phase_d.append(np.linalg.norm(Xe[i] - Xe[j]))
    same_phase_d, diff_phase_d = np.array(same_phase_d), np.array(diff_phase_d)
    print(f"  SAME phase  (noise floor): {same_phase_d.mean():7.2f} +/- {same_phase_d.std():.2f}   n={len(same_phase_d)}")
    print(f"  DIFF phase  (signal)     : {diff_phase_d.mean():7.2f} +/- {diff_phase_d.std():.2f}   n={len(diff_phase_d)}")
    ratio = diff_phase_d.mean() / same_phase_d.mean()
    print(f"  signal / noise ratio     : {ratio:.2f}x  -> {'phase signal EXCEEDS noise floor' if ratio > 1.2 else 'NO clear phase signal'}")

    # ---------------- Check 1: PHASE DECODABILITY (the gate) ----------------
    print("\n=== CHECK 1 (GATE) — is gait phase decodable from e_t? ===")
    chance = 1.0 / len(np.unique(phase_bin))
    for clf, nm in [(LogisticRegression(max_iter=2000, n_jobs=-1), "linear probe"),
                    (KNeighborsClassifier(n_neighbors=5), "k-NN (k=5)")]:
        acc = cross_val_score(clf, Z, phase_bin, cv=5, n_jobs=-1)
        print(f"  {nm:14s} phase-bin accuracy = {acc.mean()*100:5.1f}% +/- {acc.std()*100:.1f}   (chance {chance*100:.1f}%)")
    report("phase_bin", Z, phase_bin)

    # decode phase WITHIN each morphology (harder: no morphology shortcut available)
    print("  per-morphology (no cross-morphology shortcut possible):")
    for m in np.unique(morph):
        sel = morph == m
        acc = cross_val_score(LogisticRegression(max_iter=2000), Z[sel], phase_bin[sel], cv=5)
        print(f"    {m:7s} phase accuracy = {acc.mean()*100:5.1f}%  (chance {chance*100:.1f}%)")

    # ---------------- Check 2: MORPHOLOGY BASELINE (not a gate) ----------------
    print("\n=== CHECK 2 (BASELINE, not a gate) — how strongly does e_t encode morphology? ===")
    acc = cross_val_score(LogisticRegression(max_iter=2000), Z, morph, cv=5)
    print(f"  linear probe   morphology accuracy = {acc.mean()*100:5.1f}% +/- {acc.std()*100:.1f}   (chance 33.3%)")
    sil_m, bv_m = report("morphology", Z, morph, "<- BASELINE for Step 1.5")
    sil_p, bv_p = report("phase_bin", Z, phase_bin)

    print("\n=== SUMMARY ===")
    print(f"  silhouette(e_t | morphology) = {sil_m:+.4f}   between-var {bv_m*100:.1f}%   <- Step 1.5 must LOWER this")
    print(f"  silhouette(e_t | phase)      = {sil_p:+.4f}   between-var {bv_p*100:.1f}%   <- Step 1.5 must KEEP/RAISE this")

    # ---------------- UMAP (illustration only, never evidence) ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
        emb2 = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(Z)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
        for m in np.unique(morph):
            s = morph == m
            ax[0].scatter(emb2[s, 0], emb2[s, 1], s=6, alpha=0.6, label=m)
        ax[0].set_title(f"e_t by MORPHOLOGY  (silhouette {sil_m:+.3f})")
        ax[0].legend(fontsize=8)
        sc = ax[1].scatter(emb2[:, 0], emb2[:, 1], c=phase_bin, cmap="twilight", s=6, alpha=0.7)
        ax[1].set_title(f"e_t by GAIT PHASE  (silhouette {sil_p:+.3f})")
        fig.colorbar(sc, ax=ax[1], label="phase bin")
        fig.suptitle("Step 0: raw frozen V-JEPA2 e_t  —  UMAP is illustration, silhouette is the evidence")
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        print(f"\n  plot -> {args.out}")
    except Exception as e:
        print(f"  (umap plot skipped: {e})")


if __name__ == "__main__":
    main()
