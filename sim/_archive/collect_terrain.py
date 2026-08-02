"""Step 2 (terrain sub-experiment): collect clips on the terrain scene, one body.

Drives the base body with the EXPERT gait (motor_cmd replay, +x forward -- matches
the fixed side camera and the terrain placement) across the terrain, logging
everything both world models need:
  frames (vision) + full proprioception (joint pos/vel, body orientation + angular
  velocity = IMU, foot forces) + head pose.

Camera is truly fixed (no recenter) so the terrain stays put in frame while the
robot walks through it. Uses the straight expert episodes for clean forward walks.

python3 sim/collect_terrain.py --scene sim/env/medauroidea_stick_insect_terrain.ttt \
    --episodes 926,521,625 --out data/terrain_v1
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV = "/home/aria/ioon-research/sim/env"
CSV = f"{ENV}/expert_66k_aug3c_fcontact.csv"
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
JN = ["m1", "m2", "m3"]
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}      # joint -> expert motor column suffix
SENSOR = "vjepa_cam"
TRACK = "/head"
FORCE_NAMES = [f"/forceSensor_{leg}" for leg in LEGS]
EP = 66


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def read_forces(sim, force_h):
    out = np.zeros(6, np.float32)
    for i, h in enumerate(force_h):
        r = sim.readForceSensor(h)
        fv = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else [0, 0, 0]
        out[i] = float(np.sqrt(fv[0] ** 2 + fv[1] ** 2 + fv[2] ** 2))
    return out


def expert_cmds(df, ep):
    """(66,18) motor commands for one episode, leg-major [FL m1..m3, ML ...]."""
    rows = df.iloc[ep * EP:(ep + 1) * EP]
    cols = [f"motor_cmd_{leg}_{SEG[jn]}" for leg in LEGS for jn in JN]
    return rows[cols].to_numpy(np.float32)


def run_episode(sim, scene, cmds, warmup, travel, heading_k=0.0):
    sim.loadScene(os.path.abspath(scene))
    settle(sim)
    cam = sim.getObject("/" + SENSOR)
    body = sim.getObject(TRACK)
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in JN]  # matches cmds
    force_h = [sim.getObject(n) for n in FORCE_NAMES]
    LEFT_M1, RIGHT_M1 = (0, 3, 6), (9, 12, 15)    # ThC (fore-aft) indices, left / right legs

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(warmup):                       # settle holding the first pose; camera FIXED
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
    yaw0 = sim.getObjectOrientation(body, sim.handle_world)[2]   # heading to hold

    frames, jpos, jvel, orient, angvel, forces, heads, act = [], [], [], [], [], [], [], []
    start_xy = None
    for t in range(len(cmds)):
        cmd = cmds[t].copy()
        if heading_k != 0.0:                      # differential-stride yaw correction (steer back to yaw0)
            yaw = sim.getObjectOrientation(body, sim.handle_world)[2]
            err = (yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi
            for i in LEFT_M1:
                cmd[i] += heading_k * err
            for i in RIGHT_M1:
                cmd[i] -= heading_k * err
        for h, v in zip(joints, cmd):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
        p = np.array(sim.getObjectPosition(body, sim.handle_world))
        if start_xy is None:
            start_xy = p[:2].copy()
        frames.append(capture(sim, cam))
        act.append(cmd.copy())      # record the actually-applied command (incl. heading correction)
        jpos.append([sim.getJointPosition(h) for h in joints])
        jvel.append([sim.getJointVelocity(h) for h in joints])
        orient.append(sim.getObjectOrientation(body, sim.handle_world))
        lin, ang = sim.getObjectVelocity(body)
        angvel.append(ang)
        forces.append(read_forces(sim, force_h))
        heads.append(p)
        if travel > 0 and float(np.linalg.norm(p[:2] - start_xy)) >= travel:
            break
    sim.stopSimulation(); settle(sim)
    return dict(frames=np.asarray(frames, np.uint8), actions=np.asarray(act, np.float32),
                joint_pos=np.asarray(jpos, np.float32), joint_vel=np.asarray(jvel, np.float32),
                body_orient=np.asarray(orient, np.float32), body_angvel=np.asarray(angvel, np.float32),
                forces=np.asarray(forces, np.float32), head=np.asarray(heads, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episodes", type=str, default="926,521,625")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--travel", type=float, default=0.0, help="0 = full episode (66 steps)")
    ap.add_argument("--heading_k", type=float, default=0.0, help="yaw-correction gain (0 = off, open-loop)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = [int(x) for x in args.episodes.split(",")]
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(CSV)
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim")
    settle(sim)

    manifest = []
    for ep in episodes:
        d = run_episode(sim, args.scene, expert_cmds(df, ep), args.warmup, args.travel, args.heading_k)
        tag = f"terrain_ep{ep}"
        np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                            step_idx=np.arange(len(d["frames"])), foot_order=np.array(LEGS),
                            expert_episode=ep, **d)
        dx = d["head"][-1, 0] - d["head"][0, 0]
        dist = float(np.linalg.norm(d["head"][-1, :2] - d["head"][0, :2])) if len(d["head"]) else 0.0
        manifest.append(dict(tag=tag, n=len(d["frames"]), dist=dist))
        print(f"  {tag:14s} frames={d['frames'].shape} moved={dist:.2f}m (dx={dx:+.2f})")

    np.save(os.path.join(args.out, "manifest.npy"), manifest, allow_pickle=True)
    print(f"\n{len(manifest)} clips, {sum(m['n'] for m in manifest)} frames -> {args.out}")


if __name__ == "__main__":
    main()
