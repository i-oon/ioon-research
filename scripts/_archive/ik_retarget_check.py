"""Validate the core of Step A (IK retargeting) before building the collector.

Converts the expert world foot paths -> body frame (via sim transforms, so no
Euler-convention risk), scales them about each leg base to fit the short body,
then IK-solves per morphology and reports:
  - reach residual (small = the scaled target is reachable by that body)
  - the solved joint angles (must DIFFER across bodies = non-vacuous commands)

Usage (CoppeliaSim up):
  python3 scripts/ik_retarget_check.py --scale 0.5 --rows 0,40,80,120
"""
import argparse

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV = "/home/aria/ioon-research/sim/env"
CSV = f"{ENV}/expert_66k_aug3c_fcontact.csv"
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]


SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}   # joint -> expert motor column suffix (ThC/CTr/FTi)


def body_rel_via_fk(sim, df, rows):
    """Shared foot path in the ABDOMEN frame, via forward kinematics: set the base
    body's joints to the expert's motor_pos and read each foot relative to the
    abdomen. No external body-pose reference, so no frame/convention mismatch."""
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


def setup_ik(sim, simIK, leg):
    base = sim.getObjectParent(sim.getObject(f"/m1_{leg}"))  # the body the legs attach to
    tip = sim.getObject(f"/foot_{leg}")
    target = sim.createDummy(0.01)
    sim.setObjectParent(target, base, True)                  # child of body -> local coords = body frame
    env = simIK.createEnvironment()
    grp = simIK.createGroup(env)
    simIK.addElementFromScene(env, grp, base, tip, target, simIK.constraint_position)
    return env, grp, tip, target, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--rows", type=str, default="0,40,80,120")
    args = ap.parse_args()
    rows = [int(x) for x in args.rows.split(",")]

    df = pd.read_csv(CSV)
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim"); simIK = c.require("simIK")

    sim.loadScene(f"{ENV}/medauroidea_stick_insect.ttt")
    brel = body_rel_via_fk(sim, df, rows)
    print(f"scale={args.scale}  rows={rows}")
    print("body-relative FL foot (m):", np.round(brel["FL"], 3).tolist())

    for name, scene in [("long", "medauroidea_stick_insect.ttt"),
                        ("medium", "medauroidea_stick_insect_medium.ttt"),
                        ("short", "medauroidea_stick_insect_short.ttt")]:
        sim.loadScene(f"{ENV}/{scene}")
        print(f"\n=== {name} ===")
        for leg in ["FL", "HR"]:
            env, grp, tip, target, base = setup_ik(sim, simIK, leg)
            m1_local = np.array(sim.getObjectPosition(sim.getObject(f"/m1_{leg}"), base))
            joints = [sim.getObject(f"/{j}_{leg}") for j in ["m1", "m2", "m3"]]
            errs, first_ja = [], None
            for i in range(len(rows)):
                tgt = m1_local + args.scale * (brel[leg][i] - m1_local)      # scale about the leg base
                sim.setObjectPosition(target, base, list(tgt))
                simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
                foot = np.array(sim.getObjectPosition(tip, base))
                errs.append(np.linalg.norm(foot - tgt) * 1000)
                if i == 0:
                    first_ja = [round(sim.getJointPosition(j), 3) for j in joints]
            print(f"  {leg}: residual mean={np.mean(errs):5.1f}mm max={np.max(errs):5.1f}mm   joints@row0={first_ja}")
            sim.removeObjects([target])


if __name__ == "__main__":
    main()
