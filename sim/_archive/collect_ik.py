"""Step A collector: IK-retargeted dataset with the fixed camera.

For each chosen expert episode (66-step forward walk) and each morphology:
  1. shared foot path  = base body's expert motor_pos -> foot-in-abdomen (FK)
  2. per-body commands = IK that path (scaled to be reachable) for THIS body
  3. drive the commands open-loop, fixed world-frame camera, distance-gated
  4. record RGB frames + a_t (the IK commands) + measured foot forces + head pose

Behaviour (the foot path) is shared; commands differ per body -> non-vacuous.
Forces are the *measured* contact on each body (not the expert's), so contact
labels reflect what actually happened.

Straight episodes (clean forward walk): 926,521,625,144,285,997,727,728
Curvy episodes (turning, for later):    472,111,630

Usage (CoppeliaSim up, launched from the venv):
  python3 sim/collect_ik.py --episodes 926,521,625 --scale 0.5 --travel 0.8 --out data/ik_v1
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
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}
SCENES = [("long", "medauroidea_stick_insect.ttt"),
          ("medium", "medauroidea_stick_insect_medium.ttt"),
          ("short", "medauroidea_stick_insect_short.ttt")]
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


def body_rel_via_fk(sim, df, rows):
    """Shared foot path in the abdomen frame (base body FK on expert motor_pos)."""
    sim.loadScene(f"{ENV}/{SCENES[0][1]}")
    abd = sim.getObjectParent(sim.getObject("/m1_FL"))
    jh = {(leg, jn): sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG}
    foot_h = {leg: sim.getObject(f"/foot_{leg}") for leg in LEGS}
    out = {leg: [] for leg in LEGS}
    for t in rows:
        r = df.iloc[t]
        for (leg, jn), h in jh.items():
            sim.setJointPosition(h, float(r[f"motor_pos_{leg}_{SEG[jn]}"]))
        for leg in LEGS:
            out[leg].append(np.array(sim.getObjectPosition(foot_h[leg], abd)))
    return {leg: np.array(v) for leg, v in out.items()}


def precompute_commands(sim, simIK, scene, brel, scale):
    """Kinematic IK -> (T,18) joint commands, leg-major [FL m1..m3, ML ...]."""
    sim.loadScene(f"{ENV}/{scene}")
    T = len(next(iter(brel.values())))
    cmds = np.zeros((T, 18), np.float32)
    col = 0
    for leg in LEGS:
        base = sim.getObjectParent(sim.getObject(f"/m1_{leg}"))
        tip = sim.getObject(f"/foot_{leg}")
        m1_local = np.array(sim.getObjectPosition(sim.getObject(f"/m1_{leg}"), base))
        joints = [sim.getObject(f"/{jn}_{leg}") for jn in SEG]
        target = sim.createDummy(0.01)
        sim.setObjectParent(target, base, True)
        env = simIK.createEnvironment()
        grp = simIK.createGroup(env)
        simIK.addElementFromScene(env, grp, base, tip, target, simIK.constraint_position)
        for t in range(T):
            tgt = m1_local + scale * (brel[leg][t] - m1_local)
            sim.setObjectPosition(target, base, list(tgt))
            simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
            for k, j in enumerate(joints):
                cmds[t, col + k] = sim.getJointPosition(j)
        sim.removeObjects([target])
        col += 3
    return cmds


def drive_and_record(sim, scene, cmds, travel, warmup):
    """Open-loop drive of cmds with the FIXED camera; returns frames/actions/forces/head."""
    sim.loadScene(f"{ENV}/{scene}")
    settle(sim)
    cam = sim.getObject("/" + SENSOR)
    track = sim.getObject(TRACK)
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG]  # matches cmds order
    force_h = [sim.getObject(n) for n in FORCE_NAMES]

    # authored camera offset (encodes RUNWAY_AIM); re-applied once after warmup
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off_xy, cam_z = cam0[:2] - trk0[:2], cam0[2]

    sim.setStepping(True)
    sim.startSimulation()
    # settle holding the first pose
    for _ in range(warmup):
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()

    frames, actions, forces, heads = [], [], [], []
    start_xy = None
    for t in range(len(cmds)):
        for h, v in zip(joints, cmds[t]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
        p = np.array(sim.getObjectPosition(track, sim.handle_world))
        if start_xy is None:
            start_xy = p[:2].copy()
            sim.setObjectPosition(cam, sim.handle_world, [p[0] + off_xy[0], p[1] + off_xy[1], cam_z])
        frames.append(capture(sim, cam))
        actions.append(cmds[t].copy())
        forces.append(read_forces(sim, force_h))
        heads.append(p)
        if travel > 0 and float(np.linalg.norm(p[:2] - start_xy)) >= travel:
            break
    sim.stopSimulation(); settle(sim)
    return (np.asarray(frames, np.uint8), np.asarray(actions, np.float32),
            np.asarray(forces, np.float32), np.asarray(heads, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=str, default="926,521,625",
                    help="comma-separated expert episode indices (each is 66 steps)")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--travel", type=float, default=0.8, help="distance gate (m); keeps robot in the fixed frame")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    episodes = [int(x) for x in args.episodes.split(",")]
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(CSV)
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim"); simIK = c.require("simIK")
    settle(sim)

    manifest = []
    for ep in episodes:
        rows = list(range(ep * EP, ep * EP + EP))
        brel = body_rel_via_fk(sim, df, rows)                      # shared, computed once per episode
        for morph, scene in SCENES:
            cmds = precompute_commands(sim, simIK, scene, brel, args.scale)
            f, a, fc, h = drive_and_record(sim, scene, cmds, args.travel, args.warmup)
            tag = f"{morph}_ep{ep}"
            np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                frames=f, actions=a, forces=fc, head=h,
                                foot_order=np.array(LEGS), step_idx=np.arange(len(f)),
                                morph=morph, expert_episode=ep, scale=args.scale)
            dist = float(np.linalg.norm(h[-1, :2] - h[0, :2])) if len(h) else 0.0
            manifest.append(dict(tag=tag, morph=morph, ep=ep, n=len(f), dist=dist))
            print(f"  {tag:14s} frames={f.shape} moved={dist:.2f}m")

    np.save(os.path.join(args.out, "manifest.npy"), manifest, allow_pickle=True)
    tot = sum(m["n"] for m in manifest)
    print(f"\n{len(manifest)} clips, {tot} frames -> {args.out}")


if __name__ == "__main__":
    main()
