"""Watch the IK-retargeted gait walk in the CoppeliaSim GUI (no recording).

Two passes:
  1. kinematic IK -> per-timestep 18-D joint commands for the chosen body
     (shared foot path from the base body's expert motor_pos, scaled to fit).
  2. reload, run physics, drive those joint targets in real time -> it walks.

This is the collector's core minus recording. Watch the GUI window while it runs.

Usage (CoppeliaSim up, GUI visible):
  python3 sim/play_ik.py --port 23000 --morph short --loops 4
  python3 sim/play_ik.py --port 23000 --morph all --loops 3
"""
import argparse
import time

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV = "/home/aria/ioon-research/sim/env"
CSV = f"{ENV}/expert_66k_aug3c_fcontact.csv"
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}
SCENES = {"long": "medauroidea_stick_insect.ttt",
          "medium": "medauroidea_stick_insect_medium.ttt",
          "short": "medauroidea_stick_insect_short.ttt"}


def leg_length(sim, leg="FL"):
    points = [sim.getObject(f"/{jn}_{leg}") for jn in SEG]
    points.append(sim.getObject(f"/foot_{leg}"))
    xyz = [np.asarray(sim.getObjectPosition(h), dtype=float) for h in points]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(xyz[:-1], xyz[1:])))


def body_rel_via_fk(sim, df, rows):
    """Shared foot path in the abdomen frame from the base body's expert motor_pos."""
    sim.loadScene(f"{ENV}/{SCENES['long']}")
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
    """Kinematic IK -> (T, 18) joint commands, leg-major [FL m1..m3, ML ...]."""
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
            tgt = m1_local + scale * (brel[leg][t] - m1_local)
            sim.setObjectPosition(target, base, list(tgt))
            simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
            foot = np.asarray(sim.getObjectPosition(tip, base), dtype=float)
            residuals_mm.append(float(np.linalg.norm(foot - tgt) * 1000.0))
            for k, j in enumerate(joints):
                cmds[t, col + k] = sim.getJointPosition(j)
        sim.removeObjects([target])
        col += 3
    return cmds, dict(target_leg_length=target_leg_length,
                      scale=scale,
                      residual_mean_mm=float(np.mean(residuals_mm)),
                      residual_max_mm=float(np.max(residuals_mm)))


def play(sim, scene, cmds, dt, warmup=20):
    sim.loadScene(f"{ENV}/{scene}")
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG]  # leg-major, matches cmds
    sim.setStepping(True)
    sim.startSimulation()
    # settle at the first pose
    for _ in range(warmup):
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
    for t in range(len(cmds)):
        for h, v in zip(joints, cmds[t]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
        time.sleep(dt)
    sim.stopSimulation()
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--morph", default="short", choices=list(SCENES) + ["all"])
    ap.add_argument("--scale", type=float, default=0.5,
                    help="shared absolute foot-path scale about each target body's hip")
    ap.add_argument("--episode", type=int, default=926,
                    help="episode index; each is 66 steps. Straightest: 926,521,625,144,285")
    ap.add_argument("--loops", type=int, default=3, help="repeat the episode for a longer watch")
    ap.add_argument("--dt", type=float, default=0.05, help="sleep per step (0.05 = ~real time 20 Hz)")
    args = ap.parse_args()

    EP = 66  # every expert episode is exactly 66 steps (sim_time resets between them)
    df = pd.read_csv(CSV)
    a = args.episode * EP
    rows = list(range(a, a + EP))
    c = RemoteAPIClient("localhost", port=args.port)
    sim = c.require("sim"); simIK = c.require("simIK")

    print("computing shared foot path (base FK)...")
    brel = body_rel_via_fk(sim, df, rows)

    morphs = list(SCENES) if args.morph == "all" else [args.morph]
    commands_by_morph = {}
    for m in morphs:
        cmds, diag = precompute_commands(sim, simIK, SCENES[m], brel, args.scale)
        commands_by_morph[m] = cmds.copy()
        print(f"IK-retargeting '{m}': leg={diag['target_leg_length']:.4f}m "
              f"shared-scale={diag['scale']:.3f} "
              f"residual mean/max={diag['residual_mean_mm']:.2f}/{diag['residual_max_mm']:.2f}mm")
        if args.loops > 1:
            cmds = np.tile(cmds, (args.loops, 1))   # repeat the 66-step walk for a longer watch
        print(f"playing '{m}' -- watch the GUI ({len(cmds)} steps, episode {args.episode})")
        play(sim, SCENES[m], cmds, args.dt)
    if len(commands_by_morph) > 1:
        print("joint-command RMS differences (rad):")
        names = list(commands_by_morph)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rms = float(np.sqrt(np.mean((commands_by_morph[a] - commands_by_morph[b]) ** 2)))
                print(f"  {a:6s} vs {b:6s}: {rms:.6f}")
    print("done")


if __name__ == "__main__":
    main()
