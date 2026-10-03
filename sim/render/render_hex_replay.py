"""Render hexapod (c10f10t10) egocentric frames from a recorded physical state (kinematic replay).

DATA_PLAN v2, correction 1: a long hexapod walk is simulated once (`collect_ik.py --record_state`,
physics only) and each 66-frame window is rendered afterwards in its own room. No physics here: every
frame sets the abdomen world pose and the measured position of every joint (18 leg + 4 body joints)
and captures the head camera, mounted exactly as `collect_ik.drive_and_record` mounts it (same scene,
`room_for` / `scale_floor` / `randomise_ground` / `attach_ego` at the authored pose, then the room
built around the spawn point).

**Re-centring (DATA_PLAN 0.2).** With `recentre=True` the whole window is translated in x, y so its
first frame's head is at (0, 0) and the room is centred there -- the B1 v3 convention
(`render_b1_replay.py`: spawn (0, 0), room centred on the first replayed frame); orientation is kept.
With `recentre=False` the recorded world positions are used and the room is centred at the live
collector's spawn (`rec_spawn`), which is what reproduces a live-rendered clip (gate A).

  .venv/bin/python3 sim/render/render_hex_replay.py --rec clip.npz --start 0 --len 66 --seed 3 --out f.npz
"""
import argparse
import os
import sys
import time

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
from ego_camera import (attach_ego, build_texture_box, insect_forward,   # noqa: E402
                        randomise_ground, room_for, scale_floor, WALK_PITCH)

ENV = os.path.join(ROOT, "sim", "env")
SCENE = "medauroidea_c10f10t10.ttt"
SENSOR, TRACK, BODY = "vjepa_cam", "/head", "/abdomen"
EGO_FOV = 90.0
ROOM = 8.0          # collect_ik --view egocentric sets ego_box 8 m


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.05)


# per-frame ego-camera world pose (DATA_PLAN stage 3 A); one convention for every renderer
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
from render_b1_replay import CAM_POSE_CONVENTION, cam_pose  # noqa: E402,F401


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def setup(sim, seed, centre, scene=SCENE, room=ROOM):
    """Load the scene and build camera + room exactly as the live collector does."""
    settle(sim)
    sim.loadScene(os.path.join(ENV, scene))
    settle(sim)
    cam, track, body = sim.getObject("/" + SENSOR), sim.getObject(TRACK), sim.getObject(BODY)
    R = room_for(sim.getObjectPosition(track, sim.handle_world)[2])
    R["size"] = room
    scale_floor(sim, R["size"])
    randomise_ground(sim, seed=seed, uv=R["ground_uv"])
    attach_ego(sim, cam, track, insect_forward(sim), (0, 0, 0), offset_frac=R["offset_frac"],
               pitch_comp=WALK_PITCH["hexapod"])
    build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed,
                      centre=(float(centre[0]), float(centre[1])))
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(EGO_FOV)))
    jm = {sim.getObjectAlias(h, 1): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    return cam, body, jm, R


def render(sim, rec, start, length, seed, recentre=True, room=ROOM, mode="links", return_cam=False,
           frames_on=True, offset_xy=None, scene=SCENE):
    """Frames of rec[start:start+length] in room `seed`. Returns (frames, offset_xy, room dict)
    [+ cam_pose (length, 7), see CAM_POSE_CONVENTION, if return_cam; frames_on=False skips the capture].

    mode "joints": abdomen pose + all joint positions (the design); "links": every shape's recorded
    world pose (diagnostic fallback). `scene`: the body's scene file in sim/env (default c10f10t10; the c08
    test set passes medauroidea_c08f09t09.ttt)."""
    sl = slice(start, start + length)
    apos = np.asarray(rec["state_abdomen_pos"][sl], float).copy()
    aq = np.asarray(rec["state_abdomen_quat"][sl], float)
    jpos = np.asarray(rec["state_joint_pos"][sl], float)
    names = [str(n) for n in rec["state_joint_names"]]
    head = np.asarray(rec["head"][sl], float)
    if offset_xy is not None:
        # a given translation (counterfactual branches: the SOURCE window's re-centring), room at (0, 0)
        off = np.asarray(offset_xy, float).copy()
        centre = (0.0, 0.0)
    elif recentre:
        off = -head[0, :2]
        centre = (0.0, 0.0)
    else:
        off = np.zeros(2)
        centre = tuple(np.asarray(rec["rec_spawn"], float)) if "rec_spawn" in rec else (0.0, 0.0)
    apos[:, :2] += off
    cam, body, jm, R = setup(sim, seed, centre, scene=scene, room=room)
    handles = [jm[n] for n in names]
    if mode == "links":
        lnames = [str(n) for n in rec["state_link_names"]]
        lh = [sim.getObject(n) for n in lnames]
        lp = np.asarray(rec["state_link_pose"][sl], float).copy()
        lp[:, :, :2] += off
    frames, cams = [], []
    for t in range(len(apos)):
        if mode == "links":
            for h, p in zip(lh, lp[t]):
                sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        else:
            sim.setObjectPosition(body, sim.handle_world, [float(v) for v in apos[t]])
            sim.setObjectQuaternion(body, sim.handle_world, [float(v) for v in aq[t]])
            for h, v in zip(handles, jpos[t]):
                sim.setJointPosition(h, float(v))
        if frames_on:
            frames.append(capture(sim, cam))
        cams.append(cam_pose(sim, cam))
    if return_cam:
        return np.asarray(frames, np.uint8), off, R, np.asarray(cams, np.float64)
    return np.asarray(frames, np.uint8), off, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--rec", required=True, help="collect_ik --record_state clip")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--len", type=int, default=66)
    ap.add_argument("--seed", type=int, required=True, help="room seed")
    ap.add_argument("--no_recentre", action="store_true")
    ap.add_argument("--mode", choices=("joints", "links"), default="links")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    with np.load(a.rec, allow_pickle=True) as d:
        rec = {k: d[k] for k in d.files}
    f, off, _, cp = render(sim, rec, a.start, a.len, a.seed, not a.no_recentre, mode=a.mode, return_cam=True)
    np.savez_compressed(a.out, frames=f, offset_xy=off, room_seed=a.seed, window_start=a.start, cam_pose=cp,
                        cam_pose_convention=CAM_POSE_CONVENTION)
    print(f"{f.shape} -> {a.out}")


if __name__ == "__main__":
    main()
