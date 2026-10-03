"""F293: rewrite base_pos/base_quat of every B1 clip that went through the buggy `_face_forward`.

The old `_face_forward` (recollect_b1_turns / recollect_b1_more / recollect_b1_flatreal /
collect_babble_b1) read quaternion columns through views it had already overwritten. For each REAL
(non-symlink) B1 npz in the scoped directories this:
  1. re-runs the clip's source rollout exactly (build_b1_cf_branches.locate: joint_pos match < 1e-5),
  2. takes the raw window poses, applies the OLD buggy transform and the CORRECT one,
  3. if the buggy transform reproduces the stored poses (< 1e-5): rewrites base_pos/base_quat with the
     correct ones, keeps the old as base_pos_buggy/base_quat_buggy, pose_fix = "F293 face_forward fix
     2026-10-01". Frames and every other field untouched.
  4. if the stored poses already equal a correct/rigid transform: left untouched (reported).
  5. if the clip cannot be reproduced: pose_fix = "unverified" added, poses untouched (reported).
Idempotent: files whose pose_fix starts with "F293" (already fixed) are skipped. Files marked
"unverified" by an earlier run are RETRIED (the source rollout may be locatable now); a retry that
fixes them overwrites the marker, a retry that shows they never had the bug removes it, and a retry
that fails again keeps it. Every retry is logged as "retry: ..." and counted in the summary.

    .venv/bin/python3 scripts/dataset/fix_face_forward_f293.py [--dry]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
sys.path.insert(0, os.path.join(ROOT, "scripts/diagnostics/dataset"))

DIRS = ("data/egocentric_v3/beh24_b1_ego_flat", "data/egocentric_v3/beh12_b1_ego_flat",
        "data/egocentric/beh24_b1_ego_flat", "data/egocentric/beh12_b1_ego_flat",
        "data/egocentric/beh12_b1_more_ego_flat", "data/egocentric/babble_b1_pilot",
        "data/allocentric/beh12_b1_flat", "data/allocentric/beh24_b1_new12_raw",
        "data/allocentric/beh24_b1_new12_raw_v2", "data/allocentric/beh24_b1_new16_fovfix_raw",
        "data/allocentric/beh24_b1_side01_recal_raw", "data/proprioceptive/beh12_b1_flatreal")
FIX = "F293 face_forward fix 2026-10-01"


def buggy_face_forward(pos, quat):
    """The pre-F293 function, verbatim (views overwritten in place) -- only to recognise affected files."""
    pos, quat = np.asarray(pos, float).copy(), np.asarray(quat, float).copy()
    w, x, y, z = quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = pos[:, :2] - pos[0, :2]
    pos[:, 0], pos[:, 1] = pos[0, 0] + c * d[:, 0] - s * d[:, 1], pos[0, 1] + s * d[:, 0] + c * d[:, 1]
    rw, rz = np.cos(-yaw / 2), np.sin(-yaw / 2)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    quat[:, 0] = rw * qw - rz * qz
    quat[:, 1] = rw * qx - rz * qy
    quat[:, 2] = rw * qy + rz * qx
    quat[:, 3] = rw * qz + rz * qw
    return pos, quat


def one(args):
    path, dry = args
    import build_b1_cf_branches as B
    from check_face_forward_labels import raw_window
    from recollect_b1_more import _face_forward as correct_face_forward   # fixed version
    with np.load(path, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}
    prev = str(data["pose_fix"]) if "pose_fix" in data else ""
    if prev.startswith("F293"):
        return path, f"skip (already fixed: {prev!r})", 0.0
    if prev and prev != "unverified":
        return path, f"skip (unknown pose_fix {prev!r}, inspect by hand)", float("nan")
    retry = prev == "unverified"
    tag = "retry: " if retry else ""

    def clear_marker():
        # a retried clip that turned out not to need the fix: the "unverified" marker is stale
        if retry and not dry:
            data.pop("pose_fix")
            _write(path, data)
    S_p, S_q = data["base_pos"].astype(np.float64), data["base_quat"].astype(np.float64)
    ep = int(data["expert_episode"]) if "expert_episode" in data else -1
    if ("beh12_b1_more_ego_flat" in path and ep >= 4000) or ("flatreal" in path and ep >= 4000):
        # recollect_b1_noisy (own, correct transform: reads views of the OLD array, writes a copy) and
        # recollect_b1_flatreal's recovery clips (no face-forward at all): never went through the bug
        clear_marker()
        return path, tag + "untouched (not from the buggy function: recollect_b1_noisy / flatreal recovery)", 0.0
    try:
        loc = B.locate(path)
    except Exception as e:  # noqa: BLE001
        if not dry and not retry:
            data["pose_fix"] = np.array("unverified")
            _write(path, data)
        return path, tag + f"unverified ({type(e).__name__}: {e})", float("nan")
    P, Q = raw_window(loc)
    if len(P) != len(S_p):
        if not dry and not retry:
            data["pose_fix"] = np.array("unverified")
            _write(path, data)
        return path, tag + f"unverified (length mismatch {len(P)} vs {len(S_p)})", float("nan")
    Pb, Qb = buggy_face_forward(P, Q)
    res_b = max(np.abs(Pb - S_p).max(), np.abs(Qb - S_q).max())
    Pc, Qc = correct_face_forward(P, Q)
    if res_b < 1e-5:
        if not dry:
            data["base_pos_buggy"], data["base_quat_buggy"] = data["base_pos"], data["base_quat"]
            data["base_pos"] = Pc.astype(data["base_pos"].dtype)
            data["base_quat"] = Qc.astype(data["base_quat"].dtype)
            data["pose_fix"] = np.array(FIX)
            _write(path, data)
        return path, tag + "fixed", float(np.abs(Qc - S_q).max())
    clear_marker()
    return path, tag + f"untouched (not the buggy transform: residual {res_b:.1e}, tf_res {loc['tf_res']:.1e})", 0.0


def _write(path, data):
    tmp = path[:-4] + ".f293tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    paths = []
    for d in DIRS:
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
            if not os.path.islink(p):
                paths.append(p)
    with Pool(args.workers) as pool:
        res = pool.map(one, [(p, args.dry) for p in paths], chunksize=1)
    summary = {}
    for p, status, dq in res:
        key = (os.path.relpath(os.path.dirname(p), ROOT), status.split(" (")[0])
        summary[key] = summary.get(key, 0) + 1
        if not status.startswith("fixed"):
            print(f"  {os.path.relpath(p, ROOT)}: {status}")
    print("\n=== per directory")
    for (d, s), n in sorted(summary.items()):
        print(f"  {d:<48} {s:<12} {n}")
    fixed = [dq for _, s, dq in res if s.endswith("fixed")]
    print(f"fixed {len(fixed)} files; max |quat change| {max(fixed) if fixed else 0:.3f}")
    unver = [p for p, s, _ in res if "unverified" in s and not s.startswith("skip")]
    retried = [p for p, s, _ in res if s.startswith("retry: ")]
    print(f"retried {len(retried)} previously-unverified files; "
          f"{len(unver)} file(s) STILL UNVERIFIED (poses untouched, marked pose_fix='unverified', "
          f"retried on the next run){' -- dry run, nothing written' if args.dry else ''}")
    for p in unver:
        print(f"  UNVERIFIED {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
