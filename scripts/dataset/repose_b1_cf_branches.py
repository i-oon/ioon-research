"""F293: rewrite base_pos/base_quat of the B1 counterfactual-branch clips with the corrected face-forward.

Sources are already fixed (fix_face_forward_f293.py), so build_b1_cf_branches.locate now returns the
correct transform. Per (source, t) the branch physics is re-run exactly (no rendering) and every
file's poses replaced: prefix = the fixed source's rows, branch = correct transform of the raw poses.
Every other field must come out identical (checked); old poses kept as base_pos_buggy/base_quat_buggy.
Frames are NOT re-rendered -- they were rendered from the buggy-convention orientation, like the v3
source frames themselves; the per-file heading change between the two conventions is reported.

    .venv/bin/python3 scripts/dataset/repose_b1_cf_branches.py
"""
import csv
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
FIX = "F293 face_forward fix 2026-10-01"


def per_source(job):
    import build_b1_cf_branches as B
    from wm.data.embodiment import heading
    split, src, rows = job
    sdir = os.path.join(ROOT, "data/egocentric_v3",
                        "beh24_b1_ego_flat_cleantrain" if split == "train" else "beh24_b1_ego_flat_cleanval")
    out = os.path.join(ROOT, f"data/egocentric_v3/b1_cf_branches_{split}")
    sp = os.path.join(sdir, src)
    d = np.load(sp, allow_pickle=True)
    loc = B.locate(sp)
    assert loc["tf_res"] < 1e-5, (src, loc["tf_res"])
    stats = []
    for t in sorted({int(r["t"]) for r in rows}):
        r_, rec = B.rollout_to(loc, loc["off"] + B.KEEP[t])
        snap = r_.snapshot()
        for row in [r for r in rows if int(r["t"]) == t]:
            fn = os.path.join(out, row["file"])
            with np.load(fn, allow_pickle=True) as f:
                old = {k: f[k] for k in f.files}
            if "pose_fix" in old:
                continue
            P0 = int(old["cf_branch_index"])
            nb = len(old["frames"]) - P0 - 1
            offs = [int(v) for v in np.round(2.5 * (t + np.arange(1, nb + 1))) - B.KEEP[t]]
            br, fell = B.run_branch(r_, snap, tuple(float(v) for v in old["cf_command"]), offs)
            P, Q = loc["tf"](br["base_pos"], br["base_quat"])
            a = t - P0
            newp = np.concatenate([d["base_pos"][a:t + 1], P]).astype(np.float32)
            newq = np.concatenate([d["base_quat"][a:t + 1], Q]).astype(np.float32)
            for k in ("joint_pos", "joint_vel", "action", "command", "foot_contact"):
                ref = np.concatenate([d[k][a:t + 1], br[k]]).astype(np.float32)
                assert np.abs(ref - old[k]).max() < (1e-4 if k == "joint_vel" else 1e-5), (row["file"], k)
            dp = float(np.abs(newp - old["base_pos"]).max())
            dpsi = np.degrees(np.abs(np.angle(np.exp(1j * (heading(newq, "b1") - heading(old["base_quat"], "b1"))))).max())
            old["base_pos_buggy"], old["base_quat_buggy"] = old["base_pos"], old["base_quat"]
            old["base_pos"], old["base_quat"], old["pose_fix"] = newp, newq, np.array(FIX)
            tmp = fn[:-4] + ".f293tmp.npz"
            np.savez_compressed(tmp, **old)
            os.replace(tmp, fn)
            stats.append((row["file"], dp, float(np.abs(newq - old["base_quat_buggy"]).max()), float(dpsi)))
    return stats


def main():
    jobs = []
    for split in ("train", "val"):
        rows = list(csv.DictReader(open(os.path.join(ROOT, f"data/egocentric_v3/b1_cf_branches_{split}/manifest.csv"))))
        rows = [r for r in rows if r["file"]]
        for src in sorted({r["source"] for r in rows}):
            jobs.append((split, src, [r for r in rows if r["source"] == src]))
    with Pool(12) as pool:
        S = [s for st in pool.map(per_source, jobs, chunksize=1) for s in st]
    S = np.array([s[1:] for s in S]) if S else np.zeros((0, 3))
    print(f"reposed {len(S)} files; max |base_pos change| {S[:, 0].max():.2e}; max |quat change| {S[:, 1].max():.3f}; "
          f"files with quat change > 1e-4: {(S[:, 1] > 1e-4).sum()}; heading change (deg) max {S[:, 2].max():.2f}, "
          f"files > 1 deg: {(S[:, 2] > 1).sum()}")


if __name__ == "__main__":
    main()
