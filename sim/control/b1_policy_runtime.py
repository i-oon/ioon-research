"""The B1's own walking policy in MuJoCo, as a steppable object: the body's low-level controller.

Same observation, actor, action scaling and decimation as `sim/collect/rollout_b1_mujoco.py` (which
collected every B1 clip), refactored so a closed loop can hand it a velocity command each decision
instead of a whole clip's schedule. Commands are (vx, vy, yaw_cmd): the three numbers the policy
itself reads, exactly as stored per frame in the clips' `command` field -- replaying a candidate's
recorded command replays its behaviour, executed by the policy from wherever the body now is.

No heading controller: the third command entry is the yaw command the policy reads (in a recording,
the collector's heading-PI output), so replaying a recorded `command` needs no gains. The policy
identity and model ARE settings of the body: pass `policy=` ("sym" / "gait3", the collectors' table)
when continuing or replaying a specific recording; `settings()` reports what this walker runs, and
`snapshot()` / `restore()` carry them (and sensordata) so a restored state cannot silently continue
under different settings.
"""
import os
import sys

import mujoco
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
POLICY_TABLE = {"sym": ("sim/assets/b1_policy/base_1.7hz_sym/model_600.pt", 1.7),   # recollect_b1_more.POLICIES
                "gait3": ("sim/assets/b1_policy/base_gait3/model_600.pt", 2.0)}
from rollout_b1_mujoco import (ACTION_SCALE, CKPT, DECIMATION, DEFAULT_IL, FOOT_FORCE_THRESH,  # noqa: E402
                               GAIT_FREQ, MODEL, SPAWN_Z, _TOUCH_SDK_TO_IL_LEG, il_to_sdk, load_actor,
                               quat_to_R, sdk_to_il)


class B1Walker:
    def __init__(self, checkpoint=CKPT, model=MODEL, gait_freq=GAIT_FREQ, warmup=25, policy=None):
        if policy is not None:                       # a named collector policy: checkpoint + gait clock
            if policy not in POLICY_TABLE:
                raise ValueError(f"unknown B1 policy {policy!r}; known {sorted(POLICY_TABLE)}")
            checkpoint, gait_freq = os.path.join(ROOT, POLICY_TABLE[policy][0]), POLICY_TABLE[policy][1]
        self.checkpoint, self.model_path = checkpoint, model
        self.actor = load_actor(checkpoint)
        self.use_clock = self.actor[0].in_features == 60
        self.gait_freq = gait_freq
        self.m = mujoco.MjModel.from_xml_path(model)
        self.d = mujoco.MjData(self.m)
        self.adr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i): self.m.sensor_adr[i]
                    for i in range(self.m.nsensor)}
        d = self.d
        d.qpos[0:3] = [0, 0, SPAWN_Z]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:19] = il_to_sdk(DEFAULT_IL)
        d.ctrl[:] = il_to_sdk(DEFAULT_IL)
        mujoco.mj_forward(self.m, d)
        for _ in range(warmup * DECIMATION):                       # settle the standing pose
            mujoco.mj_step(self.m, d)
        self.last = np.zeros(12, np.float32)
        self.step_i = 0

    def policy_step(self, cmd):
        """One 20 ms policy step under command (vx, vy, yaw_cmd); returns the policy action."""
        m, d, adr = self.m, self.d, self.adr
        cmd = np.asarray(cmd, np.float32)
        lin = d.sensordata[adr["base_linvel"]:adr["base_linvel"] + 3]
        ang = d.sensordata[adr["base_angvel"]:adr["base_angvel"] + 3]
        grav = quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"] + 4]).T @ np.array([0., 0., -1.])
        jpos = sdk_to_il(d.qpos[7:19]) - DEFAULT_IL
        jvel = sdk_to_il(d.qvel[6:18])
        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"] + 4])[_TOUCH_SDK_TO_IL_LEG]
        foot = (touch > FOOT_FORCE_THRESH).astype(np.float32)
        obs = np.concatenate([lin, ang, grav, cmd, jpos, jvel, self.last, foot]).astype(np.float32)
        if self.use_clock:
            t = self.step_i * DECIMATION * m.opt.timestep
            phi = 2 * np.pi * self.gait_freq * t + np.array([0., np.pi, np.pi, 0.])
            obs = np.concatenate([obs, np.sin(phi), np.cos(phi)]).astype(np.float32)
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs)).numpy()
        self.last = action
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        self.step_i += 1
        self.foot = foot
        return action

    def settings(self):
        ck = os.path.relpath(os.path.abspath(self.checkpoint), ROOT)
        pol = [k for k, (c, g) in POLICY_TABLE.items() if c == ck and g == self.gait_freq]
        return {"policy": pol[0] if pol else "custom", "policy_checkpoint": ck, "gait_freq": float(self.gait_freq),
                "model": os.path.relpath(os.path.abspath(self.model_path), ROOT)}

    def snapshot(self):
        """Exact state: mjSTATE_INTEGRATION + sensordata (the policy reads the pre-step sensor values that
        mj_forward would overwrite) + last action + gait clock + settings."""
        s = np.empty(mujoco.mj_stateSize(self.m, STATE))
        mujoco.mj_getState(self.m, self.d, s, STATE)
        return s, self.d.sensordata.copy(), self.last.copy(), self.step_i, self.settings()

    def restore(self, snap):
        s, sens, last, step_i, settings = snap
        if settings != self.settings():
            raise RuntimeError(f"restore under different settings: {settings} vs {self.settings()}")
        mujoco.mj_setState(self.m, self.d, s, STATE)
        mujoco.mj_forward(self.m, self.d)
        self.d.sensordata[:] = sens
        self.last, self.step_i = last.copy(), step_i

    def state(self):
        d = self.d
        return (d.qpos[0:3].copy(), d.qpos[3:7].copy(), d.qpos[7:19].copy(), d.qvel[6:18].copy())

    def upright(self):
        w, x, y, z = self.d.qpos[3:7]
        return float(1 - 2 * (x * x + y * y))
