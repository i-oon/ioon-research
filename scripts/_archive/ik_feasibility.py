"""IK feasibility probe: can simIK retarget a foot to a Cartesian target, per body?

Not the collector -- just proves the mechanism before we build Step A:
  1. base body: target = current foot pos  -> expect ~0 reach error (recovers config)
  2. base body: perturb target by 5 cm     -> foot should track it
  3. short body: same world target (beyond its reach) -> best-effort, small residual

Run with CoppeliaSim up:
  python3 scripts/ik_feasibility.py
"""
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV = "/home/aria/ioon-research/sim/env"
LEG = "_FL"


def setup_leg_ik(sim, simIK, leg):
    base = sim.getObject("/base_dummy") if False else sim.getObjectParent(sim.getObject(f"/m1{leg}"))
    tip = sim.getObject(f"/foot{leg}")
    m1 = sim.getObject(f"/m1{leg}")
    # target dummy at the current foot pose
    target = sim.createDummy(0.01)
    sim.setObjectPosition(target, sim.handle_world, sim.getObjectPosition(tip, sim.handle_world))
    env = simIK.createEnvironment()
    grp = simIK.createGroup(env)
    simIK.addElementFromScene(env, grp, base, tip, target, simIK.constraint_position)
    return env, grp, tip, target, base


def solve_to(sim, simIK, env, grp, tip, target, world_xyz, joints):
    sim.setObjectPosition(target, sim.handle_world, list(world_xyz))
    res = simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
    foot = np.array(sim.getObjectPosition(tip, sim.handle_world))
    err = float(np.linalg.norm(foot - np.array(world_xyz)))
    ja = [round(sim.getJointPosition(j), 3) for j in joints]
    return res, err, foot, ja


def probe(scene, perturb=None):
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim"); simIK = c.require("simIK")
    sim.loadScene(scene)
    joints = [sim.getObject(f"/{j}{LEG}") for j in ["m1", "m2", "m3"]]
    env, grp, tip, target, base = setup_leg_ik(sim, simIK, LEG)
    print(f"  base='{sim.getObjectAlias(base)}'  tip='foot{LEG}'")

    foot0 = np.array(sim.getObjectPosition(tip, sim.handle_world))
    r, e, f, ja = solve_to(sim, simIK, env, grp, tip, target, foot0, joints)
    print(f"  [recover]  reach err={e*1000:6.2f} mm  joints={ja}")

    tgt = foot0 + (perturb if perturb is not None else np.array([0.05, 0.0, 0.0]))
    r, e, f, ja = solve_to(sim, simIK, env, grp, tip, target, tgt, joints)
    print(f"  [track +5cm] target={np.round(tgt,3)} foot={np.round(f,3)} err={e*1000:6.2f} mm joints={ja}")

    sim.removeObjects([target])
    return foot0


def main():
    print("=== BASE (1.0x) ===")
    foot0 = probe(f"{ENV}/medauroidea_stick_insect.ttt")
    print("\n=== SHORT (0.5x): same world target as base's foot (likely beyond reach) ===")
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim"); simIK = c.require("simIK")
    sim.loadScene(f"{ENV}/medauroidea_stick_insect_short.ttt")
    joints = [sim.getObject(f"/{j}{LEG}") for j in ["m1", "m2", "m3"]]
    env, grp, tip, target, base = setup_leg_ik(sim, simIK, LEG)
    r, e, f, ja = solve_to(sim, simIK, env, grp, tip, target, foot0, joints)
    print(f"  [reach base's far target] target={np.round(foot0,3)} foot={np.round(f,3)} "
          f"residual={e*1000:6.2f} mm joints={ja}")
    print("\n(recover ~0 mm = IK works; track follows; short-body residual = expected, that's the retarget signal)")


if __name__ == "__main__":
    main()
