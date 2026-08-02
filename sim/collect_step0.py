"""Collect the Step 0 pilot dataset: walk episodes from all 3 morphologies.

Walk-only pilot (see direction_plan.md "Data collection policy"): the gait replay
in main_script.py is the same deterministic 63-row CSV loop for every body, so
a_t is identical across morphologies at the same gait phase, and the geometric
scaling of the foot path happens automatically (forward kinematics is linear in
link length -- same joint angles on a 0.5x leg trace a 0.5x-scaled foot path).
No scaling law is applied or needed.

Episodes are collected by RELOADING the scene each time. That is deliberate:
reload introduces a ~1e-16 difference which contact-rich legged dynamics amplify
chaotically (see sim/SOURCES.md), so each episode is a genuinely different draw.
Within one scene load the sim is bit-exact deterministic, which would give
identical episodes -- useless for statistics.

Usage (CoppeliaSim must be running):
  python sim/collect_step0.py --episodes 3 --steps 200 --out data/step0
"""
import argparse
import os
import time

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV_DIR = "/home/aria/ioon-research/sim/env"
MORPHOLOGIES = [
    ("long", "medauroidea_stick_insect.ttt"),          # 1.0x  (base)
    ("medium", "medauroidea_stick_insect_medium.ttt"),  # 0.75x
    ("short", "medauroidea_stick_insect_short.ttt"),    # 0.5x
]
SENSOR_NAME = "vjepa_cam"
TRACK_OBJ = "/head"
LEG_SUFFIXES = ["_FL", "_ML", "_HL", "_FR", "_MR", "_HR"]   # foot/force order
JOINT_NAMES = ["/m1", "/m2", "/m3"]
FORCE_NAMES = [f"/forceSensor{s}" for s in LEG_SUFFIXES]    # 6 foot force sensors


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.2)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def read_forces(sim, force_handles):
    """Raw force magnitude per foot (6,). Store raw, not thresholded, so the
    stance/swing contact threshold can be chosen later in analysis."""
    out = np.zeros(6, np.float32)
    for i, h in enumerate(force_handles):
        res = sim.readForceSensor(h)
        fv = res[1] if isinstance(res, (list, tuple)) and len(res) >= 2 else [0, 0, 0]
        out[i] = float(np.sqrt(fv[0] ** 2 + fv[1] ** 2 + fv[2] ** 2))
    return out


def run_episode(sim, scene, steps, warmup, travel=0.0):
    sim.loadScene(os.path.abspath(scene))
    settle(sim)

    cam = sim.getObject("/" + SENSOR_NAME)
    track = sim.getObject(TRACK_OBJ)
    joints = [sim.getObject(f"{j}{leg}") for leg in LEG_SUFFIXES for j in JOINT_NAMES]
    force_h = [sim.getObject(n) for n in FORCE_NAMES]

    # fixed world-frame camera: capture the authored offset (encodes add_camera's
    # RUNWAY_AIM framing), re-apply it ONCE after warmup so framing is anchored to
    # the post-warmup body position (drift-compensated, identical across bodies),
    # then hold the camera fixed so the robot visibly travels across a static frame.
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off_xy, cam_z = cam0[:2] - trk0[:2], cam0[2]

    sim.setStepping(True)
    sim.startSimulation()

    frames, actions, heads, forces = [], [], [], []
    start_xy = None
    for k in range(warmup + steps):
        sim.step()
        p = np.array(sim.getObjectPosition(track, sim.handle_world))
        if k < warmup:
            continue
        if start_xy is None:
            start_xy = p[:2].copy()
            sim.setObjectPosition(cam, sim.handle_world, [p[0] + off_xy[0], p[1] + off_xy[1], cam_z])
        frames.append(capture(sim, cam))
        actions.append([sim.getJointTargetPosition(h) for h in joints])
        heads.append(p)
        forces.append(read_forces(sim, force_h))   # (6,) raw foot forces
        # distance-gate: keep the robot inside the fixed frame and give every body
        # the same spatial window. travel=0 disables (record the full --steps).
        if travel > 0 and float(np.linalg.norm(p[:2] - start_xy)) >= travel:
            break

    sim.stopSimulation()
    settle(sim)
    return (np.asarray(frames, np.uint8),
            np.asarray(actions, np.float32),
            np.asarray(heads, np.float32),
            np.asarray(forces, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200, help="max steps (cap; --travel may stop earlier)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--travel", type=float, default=0.0,
                    help="stop once the body has travelled this many metres (keeps it in the fixed frame)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")
    settle(sim)

    manifest = []
    for morph, fname in MORPHOLOGIES:
        for ep in range(args.episodes):
            t0 = time.time()
            f, a, h, fc = run_episode(sim, os.path.join(ENV_DIR, fname), args.steps, args.warmup, args.travel)
            tag = f"{morph}_ep{ep}"
            np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                frames=f, actions=a, head=h, forces=fc,
                                foot_order=np.array(LEG_SUFFIXES),
                                step_idx=np.arange(len(f)))
            dist = float(np.linalg.norm(h[-1, :2] - h[0, :2]))
            manifest.append(dict(tag=tag, morph=morph, n=len(f), dist=dist))
            print(f"  {tag:12s} frames={f.shape} a_t={a.shape} moved={dist:5.2f}m  "
                  f"({time.time()-t0:.0f}s)")

    np.save(os.path.join(args.out, "manifest.npy"), manifest, allow_pickle=True)
    total = sum(m["n"] for m in manifest)
    print(f"\n{len(manifest)} episodes, {total} frames total -> {args.out}")
    for morph, _ in MORPHOLOGIES:
        ds = [m["dist"] for m in manifest if m["morph"] == morph]
        print(f"  {morph:7s} distance: {np.mean(ds):.3f} +/- {np.std(ds):.3f} m")


if __name__ == "__main__":
    main()
