"""Q22, the real version: drive B1 with its ACTUAL trained policy, live, closed-loop, under
CoppeliaSim's own Bullet dynamics (convex-decomposed scene) -- not a static open-loop pose hold.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/b1_coppelia_live_policy.py

**Why this replaces the earlier static-standing probe.** Every gain/balance issue chased in
`b1_coppelia_dynamics_probe.py` was fighting an OPEN-LOOP problem: holding one fixed pose against
gravity with no feedback. This uses the policy this project's own "expert" B1 data was recorded
with (`sim/assets/b1_policy/base_gait3/model_600.pt`, loaded exactly like
`sim/collect/rollout_b1_mujoco.py`'s `load_actor()`) -- a CLOSED-LOOP controller that senses and
corrects for exactly the kind of drift this session's static test kept failing on. If the SAME
policy produces different motion here than in its own MuJoCo recording, that IS Q22's answer --
the real, meaningful comparison, not a stand-in.

**Uses the convex-decomposed scene** (`sim/env/b1_flat_convex.ttt`) so Bullet doesn't hang on
non-convex collision meshes (see `doc/OPEN_QUESTION.md` Q22 for the full diagnosis). Engine is
left at Bullet (0), matching hexapod's own pretraining scenes -- never switch to Newton here, that
was already tried and correctly reverted (reconfounds the comparison Q22 needs isolated).

**Observation recipe, copied verbatim from `rollout_b1_mujoco.py`, not re-derived**:
`[lin(3), ang(3), grav(3), cmd(3), jpos(12), jvel(12), last_action(12), foot(4)]` = 52-dim.
`lin`/`ang` are base velocity in the BODY frame (rotate world-frame `sim.getObjectVelocity` by the
inverse of the body's own orientation, matching MuJoCo's `velocimeter` sensor convention). `foot`
uses the CoppeliaSim force sensors at each foot (`FR_foot_fixed` etc., confirmed present in the
convex scene), thresholded the same way MuJoCo's touch sensors are.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
from rollout_b1_mujoco import (  # noqa: E402
    DEFAULT_IL, ACTION_SCALE, il_to_sdk, sdk_to_il, FOOT_FORCE_THRESH, _TOUCH_SDK_TO_IL_LEG,
    load_actor,
)

# **Inlined rather than imported, deliberately -- same reasoning as `wm/policy/b1_coppelia_env.py`.**
# This was pulled from `collect_b1_cpg_babble.py` purely for the sanity cross-check below; the
# import broke once already when a parallel session moved that file to `sim/collect/_archive/`.
DEFAULT_IL_CPG = np.array([0.061, -0.066, 0.058, -0.054,
                           1.064,  1.060, 1.077,  1.068,
                          -1.914, -1.935, -1.914, -1.913])

assert np.allclose(DEFAULT_IL, DEFAULT_IL_CPG), "DEFAULT_IL differs between the two scripts -- fix before trusting anything below"

JOINT_ALIASES_SDK = [f"{leg}_{seg}_joint"
                     for leg in ("FR", "FL", "RR", "RL")
                     for seg in ("hip", "thigh", "calf")]
FOOT_ALIASES_SDK = [f"{leg}_foot_fixed" for leg in ("FR", "FL", "RR", "RL")]
ROOT_ALIAS = "trunk_respondable"
POLICY = os.path.join(ROOT, "sim/assets/b1_policy/base_gait3/model_600.pt")


def quat_to_R(w, x, y, z):
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1 - 2*(x*x+y*y)]])


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def p(*a):
    print(*a, flush=True)


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    cmd = np.array([0.4, 0.0, 0.0], np.float32)   # forward walk, matching typical babble/expert defaults

    p("loading policy...")
    actor = load_actor(POLICY)

    p("connecting...")
    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim")
    settle(sim)
    sim.loadScene(os.path.join(ROOT, "sim/env/b1_flat_convex.ttt"))
    settle(sim)
    sim.setInt32Param(sim.intparam_dynamic_engine, 0)   # Bullet -- MUST match hexapod, never Newton here
    p("engine:", sim.getInt32Param(sim.intparam_dynamic_engine), "(0=Bullet, required)")

    jm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [jm[a] for a in JOINT_ALIASES_SDK]
    sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root = sm[ROOT_ALIAS]
    fs = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_forcesensor_type)}
    feet = [fs[a] for a in FOOT_ALIASES_SDK]

    sim.setObjectInt32Param(root, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(root, sim.shapeintparam_respondable, 1)

    GAINS = {"hip": (550.0, 2.0), "thigh": (700.0, 3.0), "calf": (970.0, 3.0)}
    FORCE = {"hip": 91.0, "thigh": 93.0, "calf": 140.0}
    for alias, h in zip(JOINT_ALIASES_SDK, joints):
        seg = alias.split("_")[1]
        kp, kd = GAINS[seg]
        sim.setJointMode(h, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
        sim.setObjectFloatParam(h, sim.jointfloatparam_pid_p, kp)
        sim.setObjectFloatParam(h, sim.jointfloatparam_pid_d, kd)
        sim.setJointMaxForce(h, FORCE[seg])
        sim.setJointTargetPosition(h, float(il_to_sdk(DEFAULT_IL)[JOINT_ALIASES_SDK.index(alias)]))
    p("joints configured, holding standing pose for warmup...")

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(20):
        sim.step()

    last_action = np.zeros(12, np.float32)
    heights, uprights = [], []
    for i in range(n_steps):
        pos = np.array(sim.getObjectPosition(root, sim.handle_world))
        quat_xyzw = sim.getObjectQuaternion(root, sim.handle_world)   # CoppeliaSim: (x,y,z,w)
        qx, qy, qz, qw = quat_xyzw
        R = quat_to_R(qw, qx, qy, qz)

        lin_w, ang_w = sim.getObjectVelocity(root)
        lin = R.T @ np.asarray(lin_w)
        ang = R.T @ np.asarray(ang_w)
        grav = R.T @ np.array([0., 0., -1.])

        jpos_sdk = np.array([sim.getJointPosition(h) for h in joints])
        jvel_sdk = np.array([sim.getJointVelocity(h) for h in joints])
        jpos = sdk_to_il(jpos_sdk) - DEFAULT_IL
        jvel = sdk_to_il(jvel_sdk)

        touch = np.array([np.linalg.norm(sim.readForceSensor(h)[1]) for h in feet])
        foot = (touch[_TOUCH_SDK_TO_IL_LEG] > FOOT_FORCE_THRESH).astype(np.float32)

        obs = np.concatenate([lin, ang, grav, cmd, jpos, jvel, last_action, foot]).astype(np.float32)
        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        last_action = action
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        for h, v in zip(joints, target):
            sim.setJointTargetPosition(h, float(np.clip(v, -10, 10)))
        sim.step()

        up_z = R[2, 2]
        heights.append(pos[2]); uprights.append(up_z)
        if i % 10 == 0:
            p(f"  step {i:4d}  z={pos[2]:.4f}  up.z={up_z:.4f}  xy=({pos[0]:.3f},{pos[1]:.3f})")

    sim.stopSimulation()
    settle(sim)
    heights, uprights = np.asarray(heights), np.asarray(uprights)
    p(f"\nheight: min={heights.min():.4f} max={heights.max():.4f} final={heights[-1]:.4f}")
    p(f"up.z:   min={uprights.min():.4f} max={uprights.max():.4f} final={uprights[-1]:.4f}")


if __name__ == "__main__":
    main()
