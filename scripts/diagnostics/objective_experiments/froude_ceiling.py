"""Companion to `family_z_ceiling.py`: is the ceiling ratio near 1.0 because z discards signal, or
because the recorded conditions are genuinely this physically similar to begin with?

`family_z_ceiling.py` measures the within/across ceiling ratio on real z = ITM(e_t, e_t+1). A ratio
near 1.0 there is ambiguous between "the task really is this redundant" and "z's large
action-irrelevant content is swamping a real but small action-relevant difference" (documented in
that script's own docstring, and F199's follow-up).

This script runs the exact same within/across pairing procedure, on the exact same clips, but on
the GROUND-TRUTH recorded Froude vector (`clip["body_motion"]`, forward/lateral/yaw) instead of z.
No encoder, no ITM, no model of any kind -- this is what the simulator actually recorded. If the
ceiling here is also near 1.0, the conditions genuinely are this similar in real units, independent
of any representation. If it is far from 1.0 (across >> within), the conditions really do differ
physically, and a ratio near 1.0 in z specifically points at z as the bottleneck, not the task.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/froude_ceiling.py \\
        --data data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \\
        --embodiment hexapod
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.insert(0, ROOT)
from wm.data.embodiment import REGISTRY, load  # noqa: E402

import re


def FAMILY(cond):
    """See family_z_ceiling.py's FAMILY for why a plain rsplit silently drops every beh24
    _bwd/_neg condition into its own singleton family."""
    c = re.sub(r"_(c|s)[\d.]+", "", cond)
    c = re.sub(r"_lvl\d+", "", c)
    if c == cond and "_" in cond:
        c = cond.rsplit("_", 1)[0]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--pairs_per_family", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reg = REGISTRY[args.embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))

    by_cond = defaultdict(list)
    for p in paths:
        with np.load(p, allow_pickle=True) as raw:
            cond = str(raw["condition"] if "condition" in raw.files else
                       raw["behavior"] if "behavior" in raw.files else "walk")
        clip = load(p, reg)
        by_cond[cond].append(np.asarray(clip["body_motion"], dtype=np.float64))

    by_family = defaultdict(list)
    for cond in by_cond:
        by_family[FAMILY(cond)].append(cond)

    rng = np.random.default_rng(args.seed)
    print(f"{len(by_cond)} conditions, {len(by_family)} families\n")
    print(f"{'family':>12}  {'within (noise floor)':>22}  {'across (family)':>18}  "
         f"{'ceiling ratio':>14}  n_within  n_across")
    overall_within, overall_across = [], []
    for fam, conds in sorted(by_family.items()):
        if len(conds) < 2:
            continue
        within_d, across_d = [], []
        for _ in range(args.pairs_per_family):
            c1 = conds[rng.integers(0, len(conds))]
            m1 = by_cond[c1]
            i = rng.integers(0, len(m1))
            t = rng.integers(0, m1[i].shape[0])
            a = m1[i][t]
            if len(m1) > 1:
                j = rng.integers(0, len(m1) - 1)
                j = j + 1 if j >= i else j
                mb = m1[j]
                tb = rng.integers(0, mb.shape[0])
                b_within = mb[tb]
            else:
                tb = rng.integers(0, m1[i].shape[0])
                b_within = m1[i][tb]
            within_d.append(((a - b_within) ** 2).mean())
            others = [c for c in conds if c != c1]
            c2 = others[rng.integers(0, len(others))]
            m2 = by_cond[c2]
            k = rng.integers(0, len(m2))
            tk = rng.integers(0, m2[k].shape[0])
            b_across = m2[k][tk]
            across_d.append(((a - b_across) ** 2).mean())
        w, a_ = float(np.mean(within_d)), float(np.mean(across_d))
        ratio = w / max(a_, 1e-9)
        overall_within.extend(within_d); overall_across.extend(across_d)
        print(f"{fam:>12}  {w:>22.6f}  {a_:>18.6f}  {ratio:>14.3f}  "
             f"{len(within_d):>8}  {len(across_d):>8}")

    w, a_ = float(np.mean(overall_within)), float(np.mean(overall_across))
    print(f"\noverall: within {w:.6f}  across {a_:.6f}  ceiling ratio {w / max(a_, 1e-9):.3f}")
    print("\nSame reading convention as family_z_ceiling.py, but on GROUND-TRUTH Froude, no model "
          "involved: ratio near 1.0 -> the conditions really are this similar in real physical "
          "units, independent of any representation. Far from 1.0 (across >> within) -> the "
          "conditions genuinely differ, and a near-1.0 ratio in z specifically means z is "
          "discarding real signal, not that the task is redundant.")


if __name__ == "__main__":
    main()
