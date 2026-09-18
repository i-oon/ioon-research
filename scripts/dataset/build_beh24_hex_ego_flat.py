"""Assemble the final 24-condition egocentric hexapod dataset from four separate sources --
`merge_behaviour_dirs.py` can't do this directly, since it requires one single `--src` tree with
balanced speed/turn/side axis counts, and refuses to write into a directory that already has files
(so it can't be run additively across sources either).

**Sources, and why each one is used or skipped, confirmed by file-timestamp correlation against
`results/beh24_final_v2/hex/` (the already-reviewed QC videos) before trusting any of them:**

  KEPT UNCHANGED, from the original beh12 ego_flat (already tagged, already egocentric):
    speed_c5.8, speed_c7.1, speed_c8.15, speed_c8.8          (forward, untouched)
    turn_s0.05, turn_s0.15, turn_s0.29, turn_s0.56            (positive, untouched)

  REPLACED, not kept: `side_L/R_lvl0/lvl1` in the ORIGINAL beh12 set are superseded by
  `beh24_hex_side_raw`'s recalibrated versions of the SAME conditions (all 4 levels recalibrated
  together, not just the 2 new ones) -- confirmed by video-vs-raw mtime correlation, the QC videos
  came from the new dir, not the old one. Keeping both would double-count these two conditions with
  inconsistent calibration.

  NEW, from raw collections still needing the merge_behaviour_dirs-style tagging
  (condition/behaviour/level/embodiment), already have frames (hexapod collection renders live,
  `--ego` is a camera flag on the same CoppeliaSim call, not a separate render pass):
    beh24_hex_side_raw       -> side_L/R_lvl0/1/2/3 (8 conditions, replaces the old lvl0/1 above)
    beh24_hex_speedbwd_raw   -> speed_c*_bwd (4 conditions)
    beh24_hex_turnneg_raw    -> turn_s*_neg (4 conditions)

Episode numbers for the new clips start at 10000 (existing beh12 episodes top out at 2303) so
nothing collides; old clips keep their existing filenames untouched.

    .venv/bin/python3 scripts/dataset/build_beh24_hex_ego_flat.py
"""
import glob
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/egocentric/beh24_c10f10t10_ego_flat")

OLD_SRC = os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat")
KEEP_CONDITIONS = {
    "speed_c5.8", "speed_c7.1", "speed_c8.15", "speed_c8.8",
    "turn_s0.05", "turn_s0.15", "turn_s0.29", "turn_s0.56",
}

NEW_SOURCES = {
    "data/allocentric/beh24_hex_side_raw": "side",
    "data/allocentric/beh24_hex_speedbwd_raw": "speed",
    "data/allocentric/beh24_hex_turnneg_raw": "turn",
}


def main():
    if os.path.exists(OUT) and os.listdir(OUT):
        raise SystemExit(f"{OUT} already has files; remove it first rather than mixing runs")
    os.makedirs(OUT, exist_ok=True)

    kept, total = 0, 0
    for f in sorted(glob.glob(os.path.join(OLD_SRC, "*.npz"))):
        with np.load(f, allow_pickle=True) as d:
            cond = str(d["condition"])
        if cond not in KEEP_CONDITIONS:
            continue
        shutil.copy2(f, os.path.join(OUT, os.path.basename(f)))  # byte-identical copy, no re-serialize
        kept += 1
    total += kept
    print(f"kept {kept} clips unchanged from {OLD_SRC}")

    episode = 10000
    for src_rel, axis in NEW_SOURCES.items():
        src = os.path.join(ROOT, src_rel)
        conditions = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
        n_this_src = 0
        for cond_i, cond in enumerate(conditions):
            paths = sorted(p for p in glob.glob(os.path.join(src, cond, "*.npz"))
                           if "manifest" not in os.path.basename(p))
            for clip_i, path in enumerate(paths):
                with np.load(path, allow_pickle=True) as clip:
                    data = {k: clip[k] for k in clip.files}
                data["expert_episode"] = np.array(episode)
                data["condition"] = np.array(cond)
                data["behaviour"] = np.array(axis)
                data["level"] = np.array(cond_i)
                data["embodiment"] = np.array("hexapod")
                np.savez_compressed(os.path.join(OUT, f"hexapod_ep{episode}.npz"), **data)
                episode += 1
                n_this_src += 1
        total += n_this_src
        print(f"tagged and added {n_this_src} clips from {src_rel} ({len(conditions)} conditions)")

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
