"""Build stratified, documented train/val/held-out(test) directories for both bodies (seed=42,
2 clips per condition held out of training -- 1 val, 1 test -- 12 conditions each, 24/12/12) --
fixes the leakage found in beh12_hinge_multistep_anchor_v2, where the real held-out set
(last-5-by-episode) concentrated in one condition (side_R_lvl1) and left 11 of 12 conditions with
zero genuine held-out representation.

**val and test must be different clips, not the same directory used for both.** An earlier version
of this script produced only train(36)/test(12); pointing wm.train's validation at that same
test(12) directory made checkpoint SELECTION depend on performance on the exact clips Slides
22/23/Section 9 are later measured against -- the same class of leakage this split exists to
remove, just moved from gradients into model selection. Val and test must be disjoint and neither
may be touched by any decision but the final one.

**The test/held-out pick is UNCHANGED from the original 36/12 version** (same rng draw, same
sequence, same clips) -- it was already pushed to BIAS and used. This version only carves an
additional, disjoint val clip out of what used to be "train", shrinking train from 3/condition to
2/condition (36 -> 24 total). Re-running this script reproduces byte-identical test picks to the
original run; only train/val are new.
"""
import os
import numpy as np

ROOT = "/home/aria/ioon-research"  # adjust if different on BIAS

SPECS = [
    ("data/egocentric/beh12_c10f10t10_ego_flat",
     "data/egocentric/beh12_c10f10t10_ego_flat_cleantrain",
     "data/egocentric/beh12_c10f10t10_ego_flat_cleanval",
     "data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout"),
    ("data/egocentric/beh12_b1_ego_flat",
     "data/egocentric/beh12_b1_ego_flat_cleantrain",
     "data/egocentric/beh12_b1_ego_flat_cleanval",
     "data/egocentric/beh12_b1_ego_flat_cleanheldout"),
]

for src, train_dir, val_dir, heldout_dir in SPECS:
    src_abs, train_abs, val_abs, heldout_abs = (
        os.path.join(ROOT, p) for p in (src, train_dir, val_dir, heldout_dir))
    os.makedirs(train_abs, exist_ok=True)
    os.makedirs(val_abs, exist_ok=True)
    os.makedirs(heldout_abs, exist_ok=True)
    paths = sorted(f for f in os.listdir(src_abs) if f.endswith(".npz"))
    by_cond = {}
    for f in paths:
        d = np.load(os.path.join(src_abs, f), allow_pickle=True)
        cond = str(d["condition"])
        by_cond.setdefault(cond, []).append(f)
    rng = np.random.default_rng(42)
    # PASS 1: exactly one rng.integers() call per condition, same order -- byte-identical to the
    # original 36/12 script's own single-pass loop. Interleaving a second draw per condition here
    # (test then val, test then val, ...) shifts the rng stream and changes every pick after the
    # first condition -- caught by diffing against the already-deployed test set before trusting
    # this script.
    remainders = {}
    held = []
    for c in sorted(by_cond):
        files = sorted(by_cond[c])
        test_pick = rng.integers(0, len(files))
        held.append(files[test_pick])
        remainders[c] = [f for i, f in enumerate(files) if i != test_pick]
    # PASS 2: the new, additional draws, all appended after every original draw -- the original
    # picks above are computed before the rng advances any further.
    val, train = [], []
    for c in sorted(by_cond):
        remaining = remainders[c]
        val_pick = rng.integers(0, len(remaining))
        val.append(remaining[val_pick])
        train.extend(f for i, f in enumerate(remaining) if i != val_pick)
    for f in train:
        link = os.path.join(train_abs, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_abs, f), link)
    for f in val:
        link = os.path.join(val_abs, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_abs, f), link)
    for f in held:
        link = os.path.join(heldout_abs, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_abs, f), link)
    print(f"{src}: {len(train)} train -> {train_dir}, {len(val)} val -> {val_dir}, "
         f"{len(held)} test -> {heldout_dir}")
    print(f"  val:  {val}")
    print(f"  test: {held}")
