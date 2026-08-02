"""Foot-contact macro-F1: within vs across morphology, + shuffled-label control.

Matches the pilot methodology used for the proposal's 0.84 number so the two are
directly comparable: restrict to the top-8 six-bit contact patterns (8 classes,
chance = 0.125), score with macro-F1 (averages classes equally -> fair under the
strong class imbalance), and report a shuffled-label control that should collapse
to chance.

Usage:
  python3 scripts/step0_macro_f1.py --emb data/step0_fixedcam/embeddings.npz
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def top8_mask(code):
    vals, counts = np.unique(code, return_counts=True)
    top8 = vals[np.argsort(counts)[::-1][:8]]
    return np.isin(code, top8)


def within_macro(Z, y, morph, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    per_body = []
    for m in np.unique(morph):
        sel = morph == m
        yy = y[sel].copy()
        if shuffle:
            rng.shuffle(yy)
        counts = np.unique(yy, return_counts=True)[1]
        ns = min(5, int(counts.min()))
        if ns < 2 or len(counts) < 2:
            continue
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=0)
        f1s = []
        for tr, te in skf.split(Z[sel], yy):
            clf = LogisticRegression(max_iter=2000).fit(Z[sel][tr], yy[tr])
            f1s.append(f1_score(yy[te], clf.predict(Z[sel][te]), average="macro"))
        per_body.append(np.mean(f1s))
    return float(np.mean(per_body))


def across_macro(Z, y, morph):
    per_hold = []
    for held in np.unique(morph):
        tr, te = morph != held, morph == held
        clf = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        mask = np.isin(y[te], clf.classes_)
        if mask.sum() == 0:
            continue
        per_hold.append(f1_score(y[te][mask], clf.predict(Z[te][mask]), average="macro"))
    return float(np.mean(per_hold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/step0_fixedcam/embeddings.npz")
    args = ap.parse_args()

    d = np.load(args.emb)
    Z = StandardScaler().fit_transform(d["e"])
    morph, code = d["morph"], d["contact_code"]

    m = top8_mask(code)
    Z, y, morph = Z[m], code[m], morph[m]
    k = len(np.unique(y))
    print(f"emb: {d['e'].shape}  top-8 frames: {m.sum()}/{len(m)} ({100*m.mean():.0f}%)  classes={k}  chance={1/k:.3f}")

    w = within_macro(Z, y, morph)
    a = across_macro(Z, y, morph)
    ctrl = within_macro(Z, y, morph, shuffle=True)
    print(f"within-body   macro-F1 = {w:.3f}")
    print(f"across-body   macro-F1 = {a:.3f}   (transfer ratio {a/w:.2f})")
    print(f"shuffled ctrl macro-F1 = {ctrl:.3f}")


if __name__ == "__main__":
    main()
