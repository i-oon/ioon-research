"""Back-fill `cam_pose` (per-frame world pose of the ego camera) into the v4 main clips (DATA_PLAN stage 3 A).

The camera is rigidly parented to one body at render time, so its world pose is
    cam(t) = parent(t) * L,
with `parent(t)` the recorded pose the renderer set for frame t and `L` the camera's pose in the parent's
frame, fixed by the mount rule (`ego_camera.attach_ego` at render time):
- hexapod (`render_hex_replay.setup`): parent = the `/head` shape, whose recorded world pose is in
  `state_link_pose` (links mode, re-centred); L is the same for every clip (authored scene pose, room 8 m).
- B1 (`render_b1_replay --ego --match_floor`): parent = the base (`base_visual`) at (`base_pos`, `base_quat`
  w,x,y,z); L depends on the clip (the camera is aimed along the clip's frame-0 heading while the base is still
  at its authored pose -- the pre-2026-10-02 rule, now `render_b1_replay --legacy_mount`), so it is read per
  clip from the same setup (`render_b1_replay.ego_setup(legacy_mount=True)`). NOTE: with that rule the camera
  yaw relative to the body = the clip's start heading; v4 B1 main clips look 0-168 deg off the body axis.
L is read from CoppeliaSim (`getObjectPose(cam, parent)`), the composition is done here in float64.
Convention: render_b1_replay.CAM_POSE_CONVENTION (stored as `cam_pose_convention`).

    .venv/bin/python3 scripts/dataset/add_cam_pose.py fill  [--port P]
    .venv/bin/python3 scripts/dataset/add_cam_pose.py verify [--port P]   (re-render 2 clips per body, compare)
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
from render_b1_replay import CAM_POSE_CONVENTION  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
B1_SCENE = os.path.join(ROOT, "sim/env/b1_flat.ttt")


def compose(parent, local):
    """parent (n, 7) world poses (xyz + quat xyzw), local (7,) -> (n, 7)."""
    Rp = Rt.from_quat(parent[:, 3:7])
    p = parent[:, :3] + Rp.apply(local[:3])
    q = (Rp * Rt.from_quat(local[3:7])).as_quat()
    q *= np.sign(q[:, 3:4] + 1e-300)            # scalar >= 0 (q and -q are the same rotation)
    return np.concatenate([p, q], 1)


def hex_local(sim):
    from render_hex_replay import setup
    cam, body, jm, R = setup(sim, 0, (0.0, 0.0))
    head = sim.getObject("/head")
    return np.asarray(sim.getObjectPose(cam, head), np.float64)


def b1_local(sim, pos0, quat0, legacy=True):
    from render_b1_replay import ego_setup
    # the v4 main clips were rendered with the pre-2026-10-02 mount rule (render_b1_replay --legacy_mount)
    root, joints, cam = ego_setup(sim, B1_SCENE, pos0, quat0, seed=0, legacy_mount=legacy)
    return np.asarray(sim.getObjectPose(cam, root), np.float64)


def hex_cam(d, L):
    names = [str(n) for n in d["state_link_names"]]
    head = np.asarray(d["state_link_pose"], np.float64)[:, names.index("/head")]
    return compose(head, L)


def b1_cam(d, L):
    pos = np.asarray(d["base_pos"], np.float32).astype(np.float64)
    q = np.asarray(d["base_quat"], np.float32).astype(np.float64)      # w x y z, as the renderer sets it
    return compose(np.concatenate([pos, q[:, [1, 2, 3, 0]]], 1), L)


def files_of(prefix):
    return sorted(p for s in ("train", "val", "heldout") for p in glob.glob(os.path.join(CW, f"{prefix}_{s}", "*.npz")))


# `fill` was run on the stage-1/2 clips (2026-10-01), now under _superseded/
FILL_DIR = {"hex": "_superseded/c10_replay_noise/hex_main", "b1": "_superseded/b1_camera_yawed/b1_main"}


def files(body):
    return sorted(p for s in ("train", "val", "heldout") for p in glob.glob(os.path.join(CW, FILL_DIR[body] + f"_{s}", "*.npz")))


def write(p, extra):
    with np.load(p, allow_pickle=True) as f:
        data = {k: f[k] for k in f.files}
    data.update(extra)
    tmp = p[:-4] + ".camtmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, p)


def do_fill(sim):
    L = hex_local(sim)
    print("hexapod camera in /head frame:", np.round(L, 6))
    for p in files("hex"):
        with np.load(p, allow_pickle=True) as d:
            cp = hex_cam(d, L)
        write(p, dict(cam_pose=cp, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
                      cam_pose_local=L, cam_pose_parent=np.array("/head")))
    print(f"hexapod: {len(files('hex'))} files")
    for p in files("b1"):
        with np.load(p, allow_pickle=True) as d:
            L = b1_local(sim, d["base_pos"][0], d["base_quat"][0])
            cp = b1_cam(d, L)
        write(p, dict(cam_pose=cp, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
                      cam_pose_local=L, cam_pose_parent=np.array("base_visual")))
    print(f"B1: {len(files('b1'))} files")


def do_verify(sim, port, b1_dirs=(FILL_DIR["b1"],), legacy=True):
    """Re-render 2 clips per body with the renderers (which now report cam_pose) and compare."""
    import subprocess
    import tempfile
    from render_hex_replay import render
    tmp = tempfile.mkdtemp(prefix="campose_")
    out = []
    for p in files("hex")[::47][:2]:
        d = dict(np.load(p, allow_pickle=True))
        rec = dict(d)
        rec["head"] = d["head"].astype(np.float64)
        f, off, R, cp = render(sim, rec, 0, len(d["frames"]), int(d["room_seed"]), recentre=True, return_cam=True)
        # stored clip is already re-centred (head[0] xy = 0), so the renderer's offset must be ~0
        dpos = np.abs(cp[:, :3] - d["cam_pose"][:, :3]).max()
        dq = np.abs(np.abs((cp[:, 3:] * d["cam_pose"][:, 3:]).sum(1)) - 1).max()
        pix = np.abs(f.astype(int) - d["frames"].astype(int)).mean()
        out.append(("hex", os.path.basename(p), dpos, dq, float(np.abs(off).max()), pix))
    for p in [q for b in b1_dirs for q in files_of(b)][::47][:2]:
        d = dict(np.load(p, allow_pickle=True))
        piece = {k: d[k] for k in ("joint_pos", "joint_vel", "action", "command", "foot_contact", "base_pos",
                                   "base_quat", "com_pos", "joint_order_sdk")}
        piece["dt"] = np.float64(0.05)
        src = os.path.join(tmp, "b.npz"); np.savez(src, **piece)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "sim/render/render_b1_replay.py"), "--port", str(port),
                            "--scene", "sim/env/b1_flat.ttt", "--traj", src, "--out", tmp, "--ego", "--match_floor",
                            "--ground_uv_mult", "1.0", "--ego_seed", str(int(d["room_seed"])), "--spawn", "0", "0"]
                           + (["--legacy_mount"] if legacy else []),
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        o = np.load(os.path.join(tmp, "b.npz"))
        cp = o["cam_pose"]
        dpos = np.abs(cp[:, :3] - d["cam_pose"][:, :3]).max()
        dq = np.abs(np.abs((cp[:, 3:] * d["cam_pose"][:, 3:]).sum(1)) - 1).max()
        pix = np.abs(o["frames"].astype(int) - d["frames"].astype(int)).mean()
        out.append(("b1", os.path.basename(p), dpos, dq, 0.0, pix))
    print(f"{'body':<5} {'clip':<22} {'max|dpos| m':>12} {'max(1-|q.q|)':>13} {'recentre off':>13} {'pixel MAE':>10}")
    for r in out:
        print(f"{r[0]:<5} {r[1]:<22} {r[2]:12.2e} {r[3]:13.2e} {r[4]:13.2e} {r[5]:10.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fill", "verify", "verify_remount"))
    ap.add_argument("--port", type=int, default=23110)
    a = ap.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    {"fill": lambda: do_fill(sim), "verify": lambda: do_verify(sim, a.port),
     "verify_remount": lambda: do_verify(sim, a.port, ("b1_clips",), legacy=False)}[a.step]()


if __name__ == "__main__":
    main()
