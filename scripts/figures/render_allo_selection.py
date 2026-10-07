"""Third-person (allocentric) chase-camera renders of the rr clips used by the offline-selection videos
(render_offline_selection.py --allo). Kinematic replay of each clip's STORED state (no physics), in the clip's own random
room (rr_room_size / rr_room_offset / room seed, built by render_shift_heldout.render_file exactly as the rr data).

Per clip: render_file renders the ego view (compared with the stored frames -> room check, ego_mae), then a second vision
sensor (copy of the ego sensor, unparented, FOV 60) is placed every frame at base - 1.5 L * heading + 0.6 L * up, looking
at the base (L = body length: hexapod 0.6 m, B1 1.0 m (with legs / camera framing); heading = stored ego camera's optical axis projected on the floor,
smoothed over 9 frames; base = B1 base_pos, hexapod mean link position). Output: <out>/<clip name>.npz (allo frames,
ego_mae).

    nice -n 19 .venv/bin/python3 scripts/dataset/render_shift_heldout.py launch --ports 25400
    nice -n 19 .venv/bin/python3 scripts/figures/render_allo_selection.py --port 25400
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
import render_shift_heldout as RS  # noqa: E402

DUMP = os.path.join(ROOT, "results/deck/weekly_1008/sel_dump")
NAME = "round1_branches_s0_rr"
BODY_LEN = {"c10": 0.6, "b1": 1.0}
FOV_DEG = 60.0


def needed(conds):
    paths = set()
    for s in ("recorded", "vision"):
        d = np.load(os.path.join(DUMP, f"{s}.npz"), allow_pickle=True)
        P = [str(p) for p in d["cand_paths"]]
        for c in conds:
            paths.add(str(d[f"goal_path|{c}"]))
            for m in ("direct", "roll_live"):
                paths.update(P[i] for i in d[f"{NAME}|w21|{m}|{c}|cand"])
    return sorted(paths)


def quat_zaxis(q):
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def chase_poses(base, cam_pose, L):
    fwd = np.array([quat_zaxis(q)[:2] for q in cam_pose[:, 3:]])
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    k = 4
    pad = np.concatenate([np.repeat(fwd[:1], k, 0), fwd, np.repeat(fwd[-1:], k, 0)])
    sm = np.array([pad[i:i + 2 * k + 1].mean(0) for i in range(len(fwd))])
    sm /= np.linalg.norm(sm, axis=1, keepdims=True)
    M = []
    for b, h in zip(base, sm):
        p = b + np.array([-1.5 * L * h[0], -1.5 * L * h[1], 0.6 * L])
        z = b - p; z /= np.linalg.norm(z)                   # optical axis
        x = np.cross(np.array([0, 0, 1.0]), z); x /= np.linalg.norm(x)   # image left
        y = np.cross(z, x)                                  # image up
        R = np.stack([x, y, z], 1)
        M.append([R[0, 0], R[0, 1], R[0, 2], p[0], R[1, 0], R[1, 1], R[1, 2], p[1], R[2, 0], R[2, 1], R[2, 2], p[2]])
    return M


def render_one(sim, path):
    body = "b1" if os.path.basename(path).startswith("b1_") else "c10"
    d0 = RS.load(path)
    room = RS.rs_room(float(d0["rr_room_size"]))
    d, fr, cp, R = RS.render_file(sim, body, path, room, d0["rr_room_offset"])
    ego_mae = float(np.abs(fr.astype(int) - d0["frames"].astype(int)).mean())
    from render_b1_replay import capture, SENSOR
    ego = sim.getObject("/" + SENSOR)
    cam = sim.copyPasteObjects([ego], 0)[0]
    sim.setObjectParent(cam, -1, True)
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(FOV_DEG)))
    sim.setObjectFloatParam(cam, sim.visionfloatparam_near_clipping, 0.02)
    if body == "b1":
        from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS
        jm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
        joints = [jm[a] for a in JOINT_ALIASES_SDK]
        sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
        root = sm[ROOT_ALIAS]
        base = np.asarray(d["base_pos"], float)
    else:
        lh = [sim.getObject(str(n)) for n in d["state_link_names"]]
        lp = np.asarray(d["state_link_pose"], float)
        base = lp[:, :, :3].mean(1)
    M = chase_poses(base, np.asarray(d["cam_pose"], float), BODY_LEN[body])
    out = []
    for t in range(len(base)):
        if body == "b1":
            q = d["base_quat"][t]
            sim.setObjectPosition(root, sim.handle_world, [float(v) for v in d["base_pos"][t]])
            sim.setObjectQuaternion(root, sim.handle_world, [float(q[1]), float(q[2]), float(q[3]), float(q[0])])
            for h, a in zip(joints, d["joint_pos"][t]):
                sim.setJointPosition(h, float(a))
        else:
            for h, p in zip(lh, lp[t]):
                sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        sim.setObjectMatrix(cam, sim.handle_world, [float(v) for v in M[t]])
        out.append(capture(sim, cam))
    sim.removeObjects([cam])   # a loose copied sensor left in the scene crashed the next loadScene (signal 11)
    return np.asarray(out, np.uint8), ego_mae, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=25400)
    ap.add_argument("--conds", nargs="+", default=["turn_s0.29", "speed_c8.8"])
    ap.add_argument("--out", default="results/deck/weekly_1008/allo_cache")
    a = ap.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    J = needed(a.conds)
    print(f"{len(J)} clips", flush=True)
    t0 = time.time()
    for p in J:
        dst = os.path.join(out, os.path.basename(p))
        if os.path.exists(dst):
            continue
        fr, mae, body = render_one(sim, p)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, frames=fr, ego_mae=mae, source=p, body_len=BODY_LEN[body], fov=FOV_DEG)
        os.replace(tmp, dst)
        print(f"{time.time() - t0:.0f}s {os.path.basename(p)} ego_mae {mae:.3f}", flush=True)


if __name__ == "__main__":
    main()
