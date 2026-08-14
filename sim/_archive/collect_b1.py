"""Cross-embodiment data collection: drive the Unitree B1 quadruped in CoppeliaSim
with the trained PPO policy, under the SAME fixed camera as the stick insect, and log
frames + full proprioception + the policy command (for the world-model dataset).

WHY CoppeliaSim and not the MuJoCo deploy: the insect clips are rendered by
CoppeliaSim's renderer; B1 frames must come from the SAME renderer + fixed camera or
V-JEPA2 would key on renderer/lighting differences and confound the cross-embodiment
comparison. So we re-render B1 here.

The policy math is ported verbatim from b1_deployment/deploy_mujoco.py (obs layout,
IL<->SDK joint remap, DEFAULT pose, ACTION_SCALE, 50 Hz decimation, trot gait clock).
Only the state source (MuJoCo -> CoppeliaSim) and the actuator (MuJoCo position
actuator -> torque-mode PD) change.

Scene prerequisites (set up once in the GUI, see IMPORT notes at bottom):
  - b1_coppelia.urdf imported; base link alias "trunk", dynamic + respondable
  - a vision sensor aliased "vjepa_cam" (the fixed camera)
  - 4 foot force sensors aliased FR_foot, FL_foot, RR_foot, RL_foot at the calf tips

  # stand test (no recording) -- verify it holds the default pose upright:
  python3 sim/collect_b1.py --scene sim/env/b1_flat.ttt --check
  # walk forward and record a clip:
  python3 sim/collect_b1.py --scene sim/env/b1_flat.ttt --vx 0.4 --steps 300 --out data/b1_v1
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ---------------------------------------------------------------------------
# Constants ported from deploy_mujoco.py (must match training exactly)
# ---------------------------------------------------------------------------
CKPT = ("/home/aria/ioon-research/sim/assets/policy/base_gait3/model_600.pt")

# IsaacLab articulation order is TYPE-MAJOR:
#   FL_hip FR_hip RL_hip RR_hip  FL_thigh ...  FL_calf ...
# CoppeliaSim URDF import + Unitree SDK/MuJoCo order is LEG-MAJOR: FR FL RR RL x (hip,thigh,calf).
# IL_TO_SDK[il] = sdk index of the joint at IsaacLab position il.
IL_TO_SDK = np.array([3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8])

def sdk_to_il(v):
    return np.asarray(v)[IL_TO_SDK]

def il_to_sdk(v):
    out = np.empty(12); out[IL_TO_SDK] = np.asarray(v); return out

# Default standing pose in IsaacLab (type-major) order -- the pose the policy stands at.
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,   # hips   FL FR RL RR
                       1.064,  1.060, 1.077,  1.068,    # thighs FL FR RL RR
                      -1.914, -1.935, -1.914, -1.913])  # calves FL FR RL RR
DEFAULT_SDK = il_to_sdk(DEFAULT_IL)

# CoppeliaSim joints in SDK leg-major order (aliases from the URDF import).
JOINT_ALIASES_SDK = [f"{leg}_{seg}_joint"
                     for leg in ("FR", "FL", "RR", "RL")
                     for seg in ("hip", "thigh", "calf")]
# Foot shapes from the URDF import, SDK leg order [FR, FL, RR, RL]; remapped to obs
# order [FL, FR, RL, RR]. Contact is a foot-height proxy (foot tip near the floor),
# which is robust and avoids splicing force sensors into the imported chain.
FOOT_ALIASES_SDK = ["FR_foot_respondable", "FL_foot_respondable",
                    "RR_foot_respondable", "RL_foot_respondable"]
_TOUCH_SDK_TO_IL_LEG = [1, 0, 3, 2]
FOOT_Z_CONTACT = 0.05                 # foot-tip world z below this => in contact (floor at 0)

ACTION_SCALE = 0.25
DECIMATION = 4                       # physics substeps per control step
PHYS_DT = 0.005                      # 0.005 s x 4 = 0.02 s control (50 Hz)
CONTROL_DT = DECIMATION * PHYS_DT    # 0.02 s -- also the gait-clock time base
GAIT_FREQ = 2.0                      # run.gait_freq from base_gait3/train_config.yaml
FOOT_FORCE_THRESH = 1.0
SPAWN_Z = 0.55                       # ~ base height target 0.556

KP = 300.0
KD = 5.0
# Newton's spring-damper is stable; use a generous force cap (the config's real EFFORT
# [91,93.3,140] makes the legs sag under the 29 kg trunk) so K=300 tracks q_des cleanly.
MAX_FORCE = 1000.0

SENSOR = "vjepa_cam"
BASE_ALIAS = "trunk_respondable"

# heading-command deployment (matches training heading_command=True): wz is a yaw RATE
# advancing a held heading; the policy is fed a heading-derived wz that holds heading.
HEAD_K = 0.5


# ---------------------------------------------------------------------------
# Policy (ported from deploy_mujoco.load_actor / quat_to_R)
# ---------------------------------------------------------------------------
def load_actor(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.")})
    layers, n_lin = [], 0
    for i in idx:
        w = sd.get(f"actor.{i}.weight")
        if w is None:
            continue
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data = w; lin.bias.data = sd[f"actor.{i}.bias"]
        if n_lin:
            layers.append(nn.ELU())
        layers.append(lin); n_lin += 1
    return nn.Sequential(*layers).eval()


def quat_to_R(q):
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1 - 2*(x*x+y*y)]])


# ---------------------------------------------------------------------------
# CoppeliaSim helpers
# ---------------------------------------------------------------------------
def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def alias_map(sim, obj_type):
    out = {}
    for h in sim.getObjectsInTree(sim.handle_scene, obj_type):
        try:
            out[sim.getObjectAlias(h)] = h
        except Exception:
            pass
    return out


class B1:
    """Holds handles + reads the 60-dim observation (deploy_mujoco._obs, sim-sourced)."""

    def __init__(self, sim):
        self.sim = sim
        joints = alias_map(sim, sim.object_joint_type)
        missing = [a for a in JOINT_ALIASES_SDK if a not in joints]
        if missing:
            raise SystemExit(f"missing B1 joints (check URDF import aliases): {missing}\n"
                             f"found joints: {sorted(joints)}")
        self.joints = [joints[a] for a in JOINT_ALIASES_SDK]         # SDK order

        bases = alias_map(sim, sim.object_shape_type)
        self.feet = [bases.get(a) for a in FOOT_ALIASES_SDK]
        if any(h is None for h in self.feet):
            print(f"  WARNING: foot shapes not all found ({FOOT_ALIASES_SDK}); "
                  f"foot_contact will read 0. found: {sorted(bases)}")
        self.base = bases.get(BASE_ALIAS) or sim.getObject(f"/{BASE_ALIAS}")
        self.root = self.base                              # articulation root (for spawn move)
        while sim.getObjectParent(self.root) != -1:
            self.root = sim.getObjectParent(self.root)
        self.cam = sim.getObject("/" + SENSOR)

        # spring-damper (PD) mode: the Newton solver applies tau = KP*(target-q) - KD*qd
        # each physics step (stable), matching the policy's Kp/Kd.
        for h in self.joints:
            sim.setObjectInt32Param(h, sim.jointintparam_motor_enabled, 1)
            sim.setObjectInt32Param(h, sim.jointintparam_ctrl_enabled, 1)
            sim.setObjectInt32Param(h, sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectFloatParam(h, sim.jointfloatparam_kc_k, KP)
            sim.setObjectFloatParam(h, sim.jointfloatparam_kc_c, KD)
            sim.setJointTargetForce(h, MAX_FORCE)

        self.last_action_il = np.zeros(12, np.float32)
        self.step_i = 0
        self.heading_target = None

    # ---- state reads (world frame -> body frame) ----
    def _R(self):
        q = self.sim.getObjectQuaternion(self.base, self.sim.handle_world)  # [x,y,z,w]
        return quat_to_R([q[3], q[0], q[1], q[2]])

    def yaw(self):
        R = self._R()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def qpos_sdk(self):
        return np.array([self.sim.getJointPosition(h) for h in self.joints])

    def qvel_sdk(self):
        return np.array([self.sim.getJointVelocity(h) for h in self.joints])

    def foot_contact_il(self):
        z = np.full(4, 1e3, np.float32)                    # SDK order [FR, FL, RR, RL]
        for i, h in enumerate(self.feet):
            if h is not None:
                z[i] = self.sim.getObjectPosition(h, self.sim.handle_world)[2]
        z = z[_TOUCH_SDK_TO_IL_LEG]                         # -> [FL, FR, RL, RR]
        return (z < FOOT_Z_CONTACT).astype(np.float32)

    def obs(self, cmd):
        R = self._R()
        lin_w, ang_w = self.sim.getObjectVelocity(self.base)
        lin = R.T @ np.asarray(lin_w)                       # body-frame base lin vel
        ang = R.T @ np.asarray(ang_w)                       # body-frame base ang vel
        grav = R.T @ np.array([0.0, 0.0, -1.0])            # projected gravity
        jpos = sdk_to_il(self.qpos_sdk()) - DEFAULT_IL
        jvel = sdk_to_il(self.qvel_sdk())
        foot = self.foot_contact_il()
        o = np.concatenate([lin, ang, grav, cmd, jpos, jvel,
                            self.last_action_il, foot]).astype(np.float32)
        # 8-dim trot clock (obs 52 -> 60), continuous phase off step_i (deploy_mujoco).
        t = self.step_i * CONTROL_DT
        off = np.array([0.0, np.pi, np.pi, 0.0], np.float32)   # FL FR RL RR
        phi = 2.0 * np.pi * GAIT_FREQ * t + off
        return np.concatenate([o, np.sin(phi), np.cos(phi)]).astype(np.float32)

    def apply_pd(self, qdes_sdk):
        """One control step = DECIMATION physics substeps with qdes held constant; the
        MuJoCo spring-damper applies tau = KP*(qdes-q) - KD*qd each substep."""
        for h, v in zip(self.joints, qdes_sdk):
            self.sim.setJointTargetPosition(h, float(v))
        for _ in range(DECIMATION):
            self.sim.step()

    def hold(self, steps, qdes_sdk=DEFAULT_SDK):
        for _ in range(steps):
            self.apply_pd(qdes_sdk)


def heading_cmd(b1, vx, vy, wz):
    """Port of deploy's heading_command: wz is a yaw rate advancing a held heading."""
    cur = b1.yaw()
    if b1.heading_target is None:
        b1.heading_target = cur
    b1.heading_target += wz * CONTROL_DT
    err = float(np.arctan2(np.sin(b1.heading_target - cur), np.cos(b1.heading_target - cur)))
    return np.array([vx, vy, float(np.clip(HEAD_K * err, -1.0, 1.0))], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300, help="control steps to record (50 Hz)")
    ap.add_argument("--warmup", type=int, default=50, help="hold DEFAULT pose before policy")
    ap.add_argument("--check", action="store_true",
                    help="stand test: hold DEFAULT, report base height + tilt, no recording")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    actor = load_actor(args.checkpoint)
    obs_dim = actor[0].in_features
    assert obs_dim == 60, f"expected 60-dim obs, actor wants {obs_dim}"

    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim")
    settle(sim)
    sim.loadScene(os.path.abspath(args.scene))
    settle(sim)
    try:
        sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_newton)  # stable articulated-body engine
        sim.setFloatParam(sim.floatparam_simulation_time_step, PHYS_DT)
    except Exception as e:
        print("  (could not set engine/dt, using scene default)", e)

    # Configure joints (spring mode) + initial pose BEFORE start: the MuJoCo engine
    # compiles its model at startSimulation, so post-start config is ignored.
    b1 = B1(sim)
    p = sim.getObjectPosition(b1.root, sim.handle_world)
    sim.setObjectPosition(b1.root, sim.handle_world, [p[0], p[1], SPAWN_Z])
    for h, v in zip(b1.joints, DEFAULT_SDK):
        sim.setJointPosition(h, float(v))
        sim.setJointTargetPosition(h, float(v))
    sim.setStepping(True)
    sim.startSimulation()
    b1.hold(args.warmup)

    if args.check:
        p = sim.getObjectPosition(b1.base, sim.handle_world)
        grav = b1._R().T @ np.array([0.0, 0.0, -1.0])
        print(f"  STAND CHECK: base height {p[2]:.3f} m (target ~0.556), "
              f"tilt {grav[0]**2 + grav[1]**2:.4f} (0 = upright), grav_body={np.round(grav,3)}")
        print("  -> if height collapses or tilt is large, check joint sign/zero conventions "
              "(URDF import vs IsaacLab) before recording.")
        sim.stopSimulation(); settle(sim)
        return

    log = {k: [] for k in ("frames", "action", "command", "joint_pos", "joint_vel",
                           "joint_target", "body_orient", "body_angvel", "foot_contact", "head")}
    for _ in range(args.steps):
        cmd = heading_cmd(b1, args.vx, args.vy, args.wz)
        obs = b1.obs(cmd)
        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        b1.last_action_il = action
        qdes_sdk = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)

        # log BEFORE stepping (state the obs/action correspond to), frame is current view
        log["frames"].append(capture(sim, b1.cam))
        log["action"].append(action.copy())                 # 12, IL order (policy output)
        log["command"].append(cmd.copy())                   # 3, post-heading [vx,vy,wz]
        log["joint_pos"].append(b1.qpos_sdk())              # 12, SDK order
        log["joint_vel"].append(b1.qvel_sdk())
        log["joint_target"].append(qdes_sdk.copy())         # 12, SDK order (q_des)
        log["body_orient"].append(sim.getObjectOrientation(b1.base, sim.handle_world))
        log["body_angvel"].append(sim.getObjectVelocity(b1.base)[1])
        log["foot_contact"].append(b1.foot_contact_il())
        log["head"].append(sim.getObjectPosition(b1.base, sim.handle_world))

        b1.apply_pd(qdes_sdk)
        b1.step_i += 1

    sim.stopSimulation(); settle(sim)

    d = {k: np.asarray(v, np.float32 if k != "frames" else np.uint8) for k, v in log.items()}
    head = d["head"]
    dist = float(np.linalg.norm(head[-1, :2] - head[0, :2])) if len(head) else 0.0
    print(f"  frames={d['frames'].shape} moved={dist:.2f}m "
          f"dx={head[-1,0]-head[0,0]:+.2f} final_z={head[-1,2]:.3f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = f"b1_vx{args.vx}_vy{args.vy}_wz{args.wz}"
        np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                            joint_order_sdk=np.array(JOINT_ALIASES_SDK),
                            action_order="IsaacLab type-major", **d)
        print(f"  saved -> {os.path.join(args.out, tag + '.npz')}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# IMPORT NOTES (one-time GUI setup to produce sim/env/b1_flat.ttt)
# ---------------------------------------------------------------------------
# 1. File > Import > URDF  ->  sim/assets/b1_description/b1_coppelia.urdf
#      - keep collision meshes; enable "convex decomposition" for the collision shapes
#      - make the base (trunk) NON-static (dynamic) and respondable
# 2. Confirm aliases: base link = "trunk"; joints = FR_hip_joint ... RL_calf_joint
#      (rename if the importer prefixed them). Foot force sensors are NOT created by
#      import -- add 4 (Add > Force sensor), parent each under a calf, position at the
#      foot tip, alias them FR_foot / FL_foot / RR_foot / RL_foot.
# 3. Add the fixed camera aliased "vjepa_cam" (reuse sim/scene/add_camera.py, re-aimed/scaled
#      for B1 -- it is much larger than the stick insect).
# 4. Add a floor if the scene has none. Save as sim/env/b1_flat.ttt.
# 5. Sanity: `python3 sim/collect_b1.py --scene sim/env/b1_flat.ttt --check`
#      -> base height should hold ~0.55 m upright. If it collapses, the joint sign/zero
#      convention differs from IsaacLab; tell me and we fix the remap before recording.
