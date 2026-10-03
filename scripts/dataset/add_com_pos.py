"""Back-fill the per-frame `com_pos` field (world, metres; FINDINGS F301) into existing clips (idempotent).

  hexapod (state_link_names / state_link_pose present): wm.data.com.hex_com (29 dynamic shapes, scene masses)
  B1 (base_pos / base_quat / joint_pos): wm.data.com.b1_com (MuJoCo subtree CoM of the trunk)

Clips without the needed state are skipped (and listed). Collectors write `com_pos` themselves since
2026-10-01 (collect_ik --record_state, collect_c10_replay_superseded cut, rollout_b1_mujoco, render_b1_replay,
build_b1_cf_branches).

    .venv/bin/python3 scripts/dataset/add_com_pos.py DIR [DIR ...]
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.com import b1_com, hex_com  # noqa: E402

for d in sys.argv[1:]:
    n = skip = have = 0
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            if "com_pos" in z.files:
                have += 1
                continue
            data = {k: z[k] for k in z.files}
        if "state_link_pose" in data and len(data["state_link_pose"]):
            com = hex_com(data["state_link_names"], data["state_link_pose"])
        elif all(k in data for k in ("base_pos", "base_quat", "joint_pos")):
            com = b1_com(data["base_pos"], data["base_quat"], data["joint_pos"])
        else:
            skip += 1
            print("  skip (no state)", p)
            continue
        data["com_pos"] = com.astype(np.float64)
        tmp = p + ".tmp.npz"
        np.savez_compressed(tmp, **data)
        os.replace(tmp, p)
        n += 1
    print(f"{d}: added {n}, already had {have}, skipped {skip}", flush=True)
