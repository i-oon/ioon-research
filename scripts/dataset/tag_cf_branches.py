"""Tag counterfactual branches for training (idempotent):

  froude_height  the source clip's median reference height: CoM z if the source carries `com_pos` (F301),
                 else base z (Froude scale identical to the source)
  segment        0 before the branch point, 1 from it (Froude smoothing never crosses the switch)
  first_pair     the branch point (pretraining pairs start there: earlier pairs repeat the source)

    .venv/bin/python3 scripts/dataset/tag_cf_branches.py
"""
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data/egocentric_v3/beh24_b1_ego_flat")
heights, n = {}, 0
for split in ("train", "val", "heldout"):
    for p in sorted(glob.glob(os.path.join(ROOT, f"data/egocentric_v3/b1_cf_branches_{split}/*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        src = str(data["cf_source"])
        if src not in heights:
            with np.load(os.path.join(SRC, src), allow_pickle=True) as s:
                heights[src] = float(np.median((s["com_pos"] if "com_pos" in s.files else s["base_pos"])[:, 2]))
        b = int(data["cf_branch_index"])
        seg = (np.arange(len(data["frames"])) >= b).astype(np.int8)
        want = {"froude_height": np.float64(heights[src]), "segment": seg, "first_pair": np.int64(b)}
        if all(k in data and np.array_equal(np.asarray(data[k]), v) for k, v in want.items()):
            continue
        data.update(want)
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, p)
        n += 1
print(f"tagged {n} branch files ({len(heights)} sources)")
