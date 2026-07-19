"""Step 0 v2 — compare label schemes for the phase / behaviour signal.

The key question this answers: was the poor cross-morphology transfer (27-39% in
v1) caused by the ENCODER, or by a weak time-based label? We now have foot-contact
ground truth, so we test several labels and see which one transfers across bodies.

Labels compared:
  time_phase   : (step % 64) // 8   -- the OLD weak label (time, not real pose)
  contact_code : 6-bit foot-contact pattern grouped into the K most common codes
  n_support    : how many feet are planted (0..6) -- coarse but noise-robust

For each: probe accuracy WITHIN a morphology vs ACROSS morphologies (train on 2,
test on the held-out body). A label that reflects true body pose should transfer
BETTER across morphologies than the time label does.

Usage:
  python scripts/step0_analyze_v2.py --emb data/step0_v2/embeddings.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def top_k_labels(codes, k=8):
    """Keep the k most frequent contact patterns as classes; the rest -> 'other'.
    Avoids classes with too few samples to learn."""
    vals, counts = np.unique(codes, return_counts=True)
    keep = set(vals[np.argsort(-counts)[:k]])
    remap = {v: i for i, v in enumerate(sorted(keep))}
    other = k
    return np.array([remap.get(c, other) for c in codes])


def within_vs_across(Z, y, morph, name):
    """Probe accuracy within each body vs trained-on-2-tested-on-held-out."""
    classes = len(np.unique(y))
    chance = 1.0 / classes

    within = []
    for m in np.unique(morph):
        sel = morph == m
        if len(np.unique(y[sel])) < 2:
            continue
        acc = cross_val_score(LogisticRegression(max_iter=2000), Z[sel], y[sel], cv=5)
        within.append(acc.mean())
    within = np.mean(within)

    across = []
    for held in np.unique(morph):
        tr, te = morph != held, morph == held
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        # align: only score on classes seen in training
        mask = np.isin(y[te], clf.classes_)
        if mask.sum() == 0:
            continue
        across.append(clf.score(Z[te][mask], y[te][mask]))
    across = np.mean(across)

    transfer = across / within if within > 0 else 0
    print(f"  {name:16s} classes={classes:2d} chance={chance*100:4.1f}%   "
          f"within={within*100:5.1f}%   across={across*100:5.1f}%   "
          f"transfer-ratio={transfer:.2f}")
    return within, across


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, default="data/step0_v2/embeddings.npz")
    args = ap.parse_args()

    d = np.load(args.emb)
    E, morph = d["e"], d["morph"]
    Z = StandardScaler().fit_transform(E)
    print(f"e_t: {E.shape}   morphologies={sorted(set(morph.tolist()))}")

    # NOTE: the old "step mod 64" time label is deliberately NOT evaluated here.
    # That period is the length of the segment trimmed out of the animal recording, not a
    # natural gait period, and the loop seam jumps 14.75 degrees. It was compared once
    # (PROGRESS.md 10.13: 38.4% cross-body transfer against 55.2% for foot contact) and that
    # settled it. Re-running a known-broken instrument only clutters the comparison.
    labels = {
        "contact_8":  top_k_labels(d["contact_code"], k=8),  # 6-bit foot contact, top-8 patterns
        "n_support":  d["n_support"],                        # number of feet planted
    }

    print("\n=== phase/behaviour signal: WITHIN vs ACROSS morphology ===")
    print("  (transfer-ratio near 1.0 = label reflects true pose, generalises across bodies;")
    print("   near 0 = label is body-specific, does NOT transfer)\n")
    for name, y in labels.items():
        within_vs_across(Z, y, morph, name)

    print("\n=== morphology decodability (baseline; should stay high) ===")
    acc = cross_val_score(LogisticRegression(max_iter=2000), Z, morph, cv=5)
    print(f"  morphology probe = {acc.mean()*100:.1f}%  (chance 33.3%)")

    print("\n=== contact label sanity ===")
    for name, y in labels.items():
        vals, counts = np.unique(y, return_counts=True)
        print(f"  {name:12s}: {len(vals)} classes, sizes min={counts.min()} max={counts.max()}")


if __name__ == "__main__":
    main()
