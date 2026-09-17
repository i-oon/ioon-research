"""Build stratified, documented train/held-out directories for both bodies (seed=42, 1 clip per
condition held out, 12 conditions each) -- fixes the leakage found in beh12_hinge_multistep_anchor_v2,
where the real held-out set (last-5-by-episode) concentrated in one condition (side_R_lvl1) and left
11 of 12 conditions with zero genuine held-out representation."""
import os
import numpy as np

ROOT = "/home/aria/ioon-research"  # adjust if different on BIAS

SPECS = [
    ("data/egocentric/beh12_c10f10t10_ego_flat", "data/egocentric/beh12_c10f10t10_ego_flat_cleantrain",
     "data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout"),
    ("data/egocentric/beh12_b1_ego_flat", "data/egocentric/beh12_b1_ego_flat_cleantrain",
     "data/egocentric/beh12_b1_ego_flat_cleanheldout"),
]

for src, train_dir, heldout_dir in SPECS:
    src_abs, train_abs, heldout_abs = (os.path.join(ROOT, p) for p in (src, train_dir, heldout_dir))
    os.makedirs(train_abs, exist_ok=True)
    os.makedirs(heldout_abs, exist_ok=True)
    paths = sorted(f for f in os.listdir(src_abs) if f.endswith(".npz"))
    by_cond = {}
    for f in paths:
        d = np.load(os.path.join(src_abs, f), allow_pickle=True)
        cond = str(d["condition"])
        by_cond.setdefault(cond, []).append(f)
    rng = np.random.default_rng(42)
    held, train = [], []
    for c in sorted(by_cond):
        files = sorted(by_cond[c])
        pick = rng.integers(0, len(files))
        held.append(files[pick])
        train.extend(f for i, f in enumerate(files) if i != pick)
    for f in train:
        link = os.path.join(train_abs, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_abs, f), link)
    for f in held:
        link = os.path.join(heldout_abs, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(src_abs, f), link)
    print(f"{src}: {len(train)} train -> {train_dir}, {len(held)} held-out -> {heldout_dir}")
    print(f"  held-out: {held}")
