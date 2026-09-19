"""Build stratified, documented train/val/held-out(test) directories for beh24, same methodology
as `make_clean_split.py` (seed=42, 2 clips per condition held out of training -- 1 val, 1 test --
24 conditions each, so 48 train / 24 val / 24 test per body instead of beh12's 24/12/12).

**No legacy test set to preserve here** -- unlike `make_clean_split.py`, which had to reproduce an
already-deployed 12-clip test set byte-for-byte via a careful two-pass RNG draw, beh24 is a new
dataset with no split committed anywhere yet, so there is nothing to stay compatible with. The
two-pass structure (all test picks, then all val picks) is kept anyway, not for compatibility but
because it is the same known-correct pattern: drawing test-then-val per condition in one interleaved
pass shifts the RNG stream between conditions in a way that's easy to get wrong silently (this is
exactly the bug `make_clean_split.py`'s own docstring documents catching).

    .venv/bin/python3 scripts/dataset/make_clean_split_beh24.py
"""
import os
import numpy as np

# derived from the file's own location, not hardcoded -- make_clean_split.py's ROOT is a literal
# path that needs manual editing per machine ("adjust if different on BIAS"), an easy step to
# forget; this one just works on whichever machine runs it.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPECS = [
    ("data/egocentric/beh24_c10f10t10_ego_flat",
     "data/egocentric/beh24_c10f10t10_ego_flat_cleantrain",
     "data/egocentric/beh24_c10f10t10_ego_flat_cleanval",
     "data/egocentric/beh24_c10f10t10_ego_flat_cleanheldout"),
    ("data/egocentric/beh24_b1_ego_flat",
     "data/egocentric/beh24_b1_ego_flat_cleantrain",
     "data/egocentric/beh24_b1_ego_flat_cleanval",
     "data/egocentric/beh24_b1_ego_flat_cleanheldout"),
]

for src, train_dir, val_dir, heldout_dir in SPECS:
    src_abs, train_abs, val_abs, heldout_abs = (
        os.path.join(ROOT, p) for p in (src, train_dir, val_dir, heldout_dir))
    for d in (train_abs, val_abs, heldout_abs):
        if os.path.exists(d) and os.listdir(d):
            raise SystemExit(f"{d} already has files; remove it first rather than mixing runs")
        os.makedirs(d, exist_ok=True)
    paths = sorted(f for f in os.listdir(src_abs) if f.endswith(".npz"))
    by_cond = {}
    for f in paths:
        d = np.load(os.path.join(src_abs, f), allow_pickle=True)
        cond = str(d["condition"])
        by_cond.setdefault(cond, []).append(f)
    rng = np.random.default_rng(42)
    # PASS 1: one test pick per condition, sorted-condition order.
    remainders = {}
    held = []
    for c in sorted(by_cond):
        files = sorted(by_cond[c])
        test_pick = rng.integers(0, len(files))
        held.append(files[test_pick])
        remainders[c] = [f for i, f in enumerate(files) if i != test_pick]
    # PASS 2: one val pick per condition, out of what's left after PASS 1, same sorted order.
    val, train = [], []
    for c in sorted(by_cond):
        remaining = remainders[c]
        val_pick = rng.integers(0, len(remaining))
        val.append(remaining[val_pick])
        train.extend(f for i, f in enumerate(remaining) if i != val_pick)
    for f in train:
        os.symlink(os.path.join(src_abs, f), os.path.join(train_abs, f))
    for f in val:
        os.symlink(os.path.join(src_abs, f), os.path.join(val_abs, f))
    for f in held:
        os.symlink(os.path.join(src_abs, f), os.path.join(heldout_abs, f))
    print(f"{src}: {len(by_cond)} conditions, {len(train)} train -> {train_dir}, "
         f"{len(val)} val -> {val_dir}, {len(held)} test -> {heldout_dir}")
    n = {len(v) for v in by_cond.values()}
    print(f"  clips per condition in source: {sorted(n)}"
         + ("" if len(n) == 1 else "   <- uneven, check before trusting this split"))
