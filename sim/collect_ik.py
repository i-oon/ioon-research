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
  python3 sim/collect_ik.py --port 23000 --episodes 926,521,625 --scale 0.5 --travel 0.8 --out data/ik_v1
  python3 sim/collect_ik.py --port 23000 --episodes 472 --loops 3 --behavior turn --travel 0 --out data/ik_turn_v1
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV = os.path.join(ROOT, "sim", "env")
CSV = f"{ENV}/expert_66k_aug3c_fcontact.csv"
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}
SCENES = [("long", "medauroidea_stick_insect.ttt"),
          ("medium", "medauroidea_stick_insect_medium.ttt"),
          ("short", "medauroidea_stick_insect_short.ttt")]
# The reference body whose forward kinematics turn the expert's joint angles into the shared
# Cartesian foot path. It must NOT follow --morphs: that flag replaces SCENES, so reading the
# reference from SCENES[0] silently rebuilt the trajectory out of whichever body was listed
# first, and every body then chased a path derived from something other than the base insect.
REFERENCE_SCENE = "medauroidea_stick_insect.ttt"
SENSOR = "vjepa_cam"
TRACK = "/head"
ROBOT_ROOT = "/abdomen"
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


def leg_length(sim, leg="FL"):
    """Total rigid-link length, measured directly from the loaded scene."""
    points = [sim.getObject(f"/{jn}_{leg}") for jn in SEG]
    points.append(sim.getObject(f"/foot_{leg}"))
    xyz = [np.asarray(sim.getObjectPosition(h), dtype=float) for h in points]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(xyz[:-1], xyz[1:])))


# forward > 0.30 m and lateral < 0.20 m per episode; every sound body in ik_walk_8body clears
# both, the two that collapse fail forward, the two that veer fail lateral
WALK_FORWARD_M = 0.30
WALK_LATERAL_M = 0.20


def walk_check(head):
    """Signed forward travel and lateral drift, reported separately.

    Never collapse these into one distance. The earlier check used
    `norm(head[-1,:2] - head[0,:2])`, which is unsigned, and a body that tipped over and rotated
    on the spot read a healthy 0.46 m -- two such bodies reached the dataset and trained into
    every Stage 2 run before anyone looked at the frames (FINDINGS.md F42). A body can also
    travel a long way forward while crabbing just as far sideways, which one number hides.

    A pass here still is not proof. No statistic distinguishes "walks oddly" from "fell over and
    is now spinning"; watch the clip.
    """
    if not len(head):
        return 0.0, 0.0, "EMPTY"
    forward = float(head[-1, 0] - head[0, 0])
    lateral = float(abs(head[-1, 1] - head[0, 1]))
    if forward < WALK_FORWARD_M:
        verdict = "FAILS forward" + (" (BACKWARDS)" if forward < 0 else "")
    elif lateral > WALK_LATERAL_M:
        verdict = "FAILS lateral, veering"
    else:
        verdict = "ok"
    return forward, lateral, verdict


def body_rel_via_fk(sim, df, rows):
    """Shared foot path in the abdomen frame (base body FK on expert motor_pos)."""
    sim.loadScene(f"{ENV}/{REFERENCE_SCENE}")
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
    target_leg_length = leg_length(sim)
    residuals_mm = []
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
            # One shared absolute Cartesian behavior for every morphology.  IK
            # must therefore produce different joint commands for different links.
            tgt = m1_local + scale * (brel[leg][t] - m1_local)
            sim.setObjectPosition(target, base, list(tgt))
            simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
            foot = np.asarray(sim.getObjectPosition(tip, base), dtype=float)
            residuals_mm.append(float(np.linalg.norm(foot - tgt) * 1000.0))
            for k, j in enumerate(joints):
                cmds[t, col + k] = sim.getJointPosition(j)
        sim.removeObjects([target])
        col += 3
    diagnostic = dict(target_leg_length=target_leg_length,
                      scale=scale,
                      residual_mean_mm=float(np.mean(residuals_mm)),
                      residual_max_mm=float(np.max(residuals_mm)))
    return cmds, diagnostic


def drive_and_record(sim, scene, cmds, travel, warmup, cam_dx=0.0, cam_dy=0.0, spawn=None):
    """Open-loop drive of cmds with the FIXED camera; returns frames/actions/forces/head.

    cam_dx/cam_dy shift the camera in the world plane on top of the scene's authored offset.
    With the authored offset alone the robot starts against the right image edge and stays
    partly outside it for roughly the first two thirds of every clip.
    """
    sim.loadScene(f"{ENV}/{scene}")
    settle(sim)
    cam = sim.getObject("/" + SENSOR)
    track = sim.getObject(TRACK)

    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG]  # matches cmds order
    force_h = [sim.getObject(n) for n in FORCE_NAMES]

    # authored camera offset (encodes RUNWAY_AIM); must be read BEFORE any respawn, or it
    # measures the camera against the moved robot instead of the authored framing
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off_xy, cam_z = cam0[:2] - trk0[:2], cam0[2]

    # The scene spawns the robot near the floor's edge, which puts the floor corner inside the
    # frame. Re-spawning at the floor centre keeps the edge outside the field of view, and using
    # the same spawn for every embodiment makes them stand on identical floor -- without that,
    # the insect and B1 backgrounds differ across ~27% of pixels.
    if spawn is not None:
        root = sim.getObject(ROBOT_ROOT)
        pos = sim.getObjectPosition(root, sim.handle_world)
        head = sim.getObjectPosition(track, sim.handle_world)
        sim.setObjectPosition(root, sim.handle_world,
                              [spawn[0] + pos[0] - head[0], spawn[1] + pos[1] - head[1], pos[2]])

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
            sim.setObjectPosition(cam, sim.handle_world,
                                  [p[0] + off_xy[0] + cam_dx, p[1] + off_xy[1] + cam_dy, cam_z])
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
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--episodes", type=str, default="926,521,625",
                    help="comma-separated expert episode indices (each is 66 steps)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="shared absolute foot-path scale about each target body's hip")
    ap.add_argument("--travel", type=float, default=0.8, help="distance gate (m); keeps robot in the fixed frame")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cam_dx", type=float, default=0.0,
                    help="shift the fixed camera along world x; see drive_and_record")
    ap.add_argument("--cam_dy", type=float, default=0.0, help="shift the fixed camera along world y")
    ap.add_argument("--spawn", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="respawn the robot head at this world x y; use the same value for every\n                         embodiment so they stand on identical floor")
    ap.add_argument("--repeats", type=int, default=1,
                    help="record each (episode,morph) this many times (fresh chaotic draw each) "
                         "-> repeated same-body-same-behavior for the render-lock gate")
    ap.add_argument("--loops", type=int, default=1,
                    help="repeat each 66-step expert foot path into one longer clip")
    ap.add_argument("--behavior", type=str, default="walk",
                    help="behavior label saved in every clip (walk / turn / stop)")
    ap.add_argument("--stop", type=int, default=0,
                    help="if >0: STOP mode — hold the stance for this many frames (no stepping)")
    ap.add_argument("--turn_bias", type=float, default=0.0,
                    help="legacy asymmetric ThC offset (rad) added left/subtracted right")
    ap.add_argument("--morphs", type=str, nargs="+", default=None, metavar="NAME=SCENE",
                    help="bodies to record, e.g. c10f06t10=medauroidea_c10f06t10.ttt. Scene paths "
                         "are relative to sim/env. Names must not contain '_' because "
                         "wm/data/dataset.py reads the body from the filename prefix. "
                         "Defaults to the three uniform-scale bodies.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    episodes = [int(x) for x in args.episodes.split(",")]
    global SCENES
    if args.morphs:
        SCENES = []
        for spec in args.morphs:
            name, _, scene = spec.partition("=")
            if not scene:
                raise SystemExit(f"--morphs wants NAME=SCENE, got {spec!r}")
            if "_" in name:
                raise SystemExit(f"body name {name!r} must not contain '_'")
            if not os.path.exists(os.path.join(ENV, scene)):
                raise SystemExit(f"scene not found: {os.path.join(ENV, scene)}")
            SCENES.append((name, scene))
        print("bodies: " + ", ".join(f"{n} <- {s}" for n, s in SCENES))
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(CSV)
    c = RemoteAPIClient("localhost", port=args.port)
    sim = c.require("sim"); simIK = c.require("simIK")
    settle(sim)

    # ---- STOP mode: hold the walk's first stance (no stepping) ----
    if args.stop > 0:
        rows = list(range(episodes[0] * EP, episodes[0] * EP + EP))
        brel = body_rel_via_fk(sim, df, rows)
        man = []
        for morph, scene in SCENES:
            cmds, ikdiag = precompute_commands(sim, simIK, scene, brel, args.scale)
            print(f"  {morph:6s} leg={ikdiag['target_leg_length']:.4f}m "
                  f"shared-scale={ikdiag['scale']:.3f} "
                  f"IK residual mean/max={ikdiag['residual_mean_mm']:.2f}/{ikdiag['residual_max_mm']:.2f}mm")
            stance = np.tile(cmds[0], (args.stop, 1))               # hold pose -> stand still
            for rep in range(args.repeats):
                f, a, fc, h = drive_and_record(sim, scene, stance, 0.0, args.warmup, args.cam_dx, args.cam_dy, args.spawn)
                tag = f"{morph}_stop_r{rep}" if args.repeats > 1 else f"{morph}_stop"
                np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                    frames=f, actions=a, forces=fc, head=h,
                                    foot_order=np.array(LEGS), step_idx=np.arange(len(f)),
                                    morph=morph, expert_episode=-1, repeat=rep, scale=args.scale,
                                    behavior="stop")
                fwd, lat, verdict = walk_check(h)
                man.append(dict(tag=tag, morph=morph, ep=-1, rep=rep, n=len(f),
                                forward=fwd, lateral=lat, verdict=verdict))
                print(f"  {tag:16s} frames={f.shape} forward={fwd:+.2f}m "
                      f"lateral={lat:.2f}m  {verdict}")
        np.save(os.path.join(args.out, "manifest_stop.npy"), man, allow_pickle=True)
        print(f"\n{len(man)} stop clips -> {args.out}")
        return

    manifest = []
    for ep in episodes:
        rows = list(range(ep * EP, ep * EP + EP))
        brel = body_rel_via_fk(sim, df, rows)  # shared Cartesian behavior, once per episode
        for morph, scene in SCENES:
            cmds, ikdiag = precompute_commands(sim, simIK, scene, brel, args.scale)
            print(f"  {morph:6s} leg={ikdiag['target_leg_length']:.4f}m "
                  f"shared-scale={ikdiag['scale']:.3f} "
                  f"IK residual mean/max={ikdiag['residual_mean_mm']:.2f}/{ikdiag['residual_max_mm']:.2f}mm")
            if args.turn_bias != 0.0:
                cmds = cmds.copy()
                cmds[:, [0, 3, 6]] += args.turn_bias
                cmds[:, [9, 12, 15]] -= args.turn_bias
            if args.loops > 1:
                cmds = np.tile(cmds, (args.loops, 1))
            pre = "" if args.behavior == "walk" else f"{args.behavior}_"
            for rep in range(args.repeats):
                f, a, fc, h = drive_and_record(sim, scene, cmds, args.travel, args.warmup, args.cam_dx, args.cam_dy, args.spawn)  # fresh draw each
                tag = f"{morph}_{pre}ep{ep}_r{rep}" if args.repeats > 1 else f"{morph}_{pre}ep{ep}"
                np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                    frames=f, actions=a, forces=fc, head=h,
                                    foot_order=np.array(LEGS), step_idx=np.arange(len(f)),
                                    morph=morph, expert_episode=ep, repeat=rep, scale=args.scale,
                                    behavior=args.behavior)
                fwd, lat, verdict = walk_check(h)
                manifest.append(dict(tag=tag, morph=morph, ep=ep, rep=rep, n=len(f),
                                     forward=fwd, lateral=lat, verdict=verdict))
                print(f"  {tag:16s} frames={f.shape} forward={fwd:+.2f}m "
                      f"lateral={lat:.2f}m  {verdict}")

    np.save(os.path.join(args.out, "manifest.npy"), manifest, allow_pickle=True)
    tot = sum(m["n"] for m in manifest)
    print(f"\n{len(manifest)} clips, {tot} frames -> {args.out}")


if __name__ == "__main__":
    main()
