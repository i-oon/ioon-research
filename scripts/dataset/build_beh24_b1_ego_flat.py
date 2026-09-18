"""Assemble the final 24-condition egocentric B1 dataset, mirroring
`build_beh24_hex_ego_flat.py`'s logic for the other body.

**Simpler than the hexapod case**: `beh24_b1_new16_fovfix_raw` is already the rendered,
already-tagged output of `rerender_b1_framing.py` (confirmed directly -- its clips carry real
frame pixel statistics and `condition`/`behaviour`/`level`/`expert_episode`/`embodiment` fields
already, not raw MuJoCo-only data needing a fresh CoppeliaSim pass), so this is a straight copy +
combine, no re-tagging and no simulator run.

  KEPT UNCHANGED, from the original beh12 ego_flat:
    speed_vx0.30, speed_vx0.38, speed_vx0.40, speed_vx0.50    (forward, untouched)
    turn_w0.008, turn_w0.024, turn_w0.037, turn_w0.075         (positive, untouched)

  REPLACED, not kept: `side_L/R_lvl0/lvl1` in the original beh12 set are superseded by
  `beh24_b1_new16_fovfix_raw`'s recalibrated + FOV-fixed versions of the SAME conditions (all 4
  levels together), same reasoning as the hexapod side note.

  NEW, copied as-is (already tagged, episode range 2800-4303, no collision with beh12's 0-2303):
    beh24_b1_new16_fovfix_raw -> side_L/R_lvl0/1/2/3 (8, replaces old lvl0/1), speed_vx negative
    (4), turn_w negative (4) = 16 conditions.

    .venv/bin/python3 scripts/dataset/build_beh24_b1_ego_flat.py
"""
import glob
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/egocentric/beh24_b1_ego_flat")

OLD_SRC = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
KEEP_CONDITIONS = {
    "speed_vx0.30", "speed_vx0.38", "speed_vx0.40", "speed_vx0.50",
    "turn_w0.008", "turn_w0.024", "turn_w0.037", "turn_w0.075",
}
NEW_SRC = os.path.join(ROOT, "data/allocentric/beh24_b1_new16_fovfix_raw")


def main():
    if os.path.exists(OUT) and os.listdir(OUT):
        raise SystemExit(f"{OUT} already has files; remove it first rather than mixing runs")
    os.makedirs(OUT, exist_ok=True)

    kept = 0
    for f in sorted(glob.glob(os.path.join(OLD_SRC, "*.npz"))):
        with np.load(f, allow_pickle=True) as d:
            cond = str(d["condition"])
        if cond not in KEEP_CONDITIONS:
            continue
        shutil.copy2(f, os.path.join(OUT, os.path.basename(f)))
        kept += 1
    print(f"kept {kept} clips unchanged from {OLD_SRC}")

    added = 0
    for f in sorted(glob.glob(os.path.join(NEW_SRC, "*.npz"))):
        shutil.copy2(f, os.path.join(OUT, os.path.basename(f)))
        added += 1
    print(f"copied {added} clips from {NEW_SRC}")

    total = kept + added
    print(f"\n{total} clips -> {OUT}")
    by_cond = {}
    for f in glob.glob(os.path.join(OUT, "*.npz")):
        with np.load(f, allow_pickle=True) as d:
            by_cond[str(d["condition"])] = by_cond.get(str(d["condition"]), 0) + 1
    print(f"{len(by_cond)} distinct conditions:")
    for c in sorted(by_cond):
        print(f"  {c:16s} {by_cond[c]} clips")


if __name__ == "__main__":
    main()
