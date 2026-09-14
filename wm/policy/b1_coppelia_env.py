"""CoppeliaSim-native version of `b1_mujoco_env.py` -- same interface (`reset`, `step`,
same observation layout, same world-model reward), different physics backend. Built per an
explicit request to train Q21 step 3's RL controller entirely in CoppeliaSim-Bullet (F196's
dynamics) instead of MuJoCo, so the resulting policy is finally "a controller that works with
CoppeliaSim physics" rather than another attempt to transplant the MuJoCo-trained policy (already
confirmed to fail there -- the documented sim2sim gap in `rollout_b1_mujoco.py`).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/b1_coppelia_env.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1.pt

**Reuses F196's exact plumbing** (`b1_coppelia_cpg_controller.py`): scene `sim/env/b1_flat_convex.ttt`
(Bullet, engine 0, verified elsewhere), the same joint aliases/PID/max-force setup, the same
`DEFAULT_IL`/`ACTION_SCALE`/`il_to_sdk` action convention every other B1 script in this project
uses, so an action vector means the same thing here as it does in MuJoCo.

**Reset, without reloading the scene file every episode.** Reloading an 18 MB `.ttt` scene per
episode would make training impossibly slow. Instead: stop the simulation, re-apply the neutral
joint pose (position AND target, matching F196's own init-order fix -- setting only the target
without the position was the original standing-collapse bug), restart. This returns dynamics to
the scene's saved initial state without the reload cost.

**Known cost, accepted going in**: CoppeliaSim is driven through a remote API (one round-trip per
`sim.step()`), meaningfully slower per step than MuJoCo's native stepping. Expect fewer
environment steps per wall-clock hour than the MuJoCo version; budget training accordingly.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

# **Inlined rather than imported, deliberately.** These were pulled from `collect_b1_cpg_babble.py`
# / `b1_coppelia_cpg_controller.py` until that import broke when the file got moved to `_archive/`
# by a parallel session, taking down 7 dependent scripts with it. Copied here so this file has no
# cross-file dependency to break. **Trade-off, stated plainly**: this is now a second copy of the
# B1 action-space convention -- if `DEFAULT_IL`/`ACTION_SCALE`/`IL_TO_SDK` ever change at the
# source, this copy will silently drift out of sync with every other B1 script. Worth reconciling
# back to a single source of truth once the pipeline this is being built for stabilizes; not done
# now, to stop this specific breakage from recurring while that's in flux.
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
IL_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])

# F203's per-joint remap is REVERTED (F206) -- see b1_mujoco_env.py's comment for the measurement
# that found it was corrupting every action fed to the physics.


def il_to_sdk(v):
    out = np.empty(12)
    out[IL_TO_SDK] = np.asarray(v)
    return out


JOINT_ALIASES_SDK = [f"{leg}_{seg}_joint"
                     for leg in ("FR", "FL", "RR", "RL")
                     for seg in ("hip", "thigh", "calf")]
ROOT_ALIAS = "trunk_respondable"
JOINT_LIMITS = {
    "hip": (-0.75, 0.75),
    "thigh": (-1.0, 3.5),
    "calf": (-2.6, -0.6),
}
MAX_FORCE = {"hip": 91.0, "thigh": 93.0, "calf": 140.0}

OBS_DIM = 12 + 12 + 4 + 6
ACTION_DIM = 12
FALL_HEIGHT = 0.35   # matches F196's own fell criterion
FALL_UPZ = 0.5        # matches F196's own fell criterion
ALIVE_BONUS = 0.1      # per step, unconditional -- kept small on purpose
TRACKING_SCALE = 1.0   # exp(-TRACKING_SCALE * |pred-goal|), bounded to (0, 1]
TRACKING_WEIGHT = 5.0  # multiplies the bounded tracking term -- see b1_mujoco_env.py's identical
                       # constants for the full history: a fixed ALIVE_BONUS=1.0 over an UNBOUNDED
                       # cost first made dying fast reward-optimal; then, once bounded,
                       # ALIVE_BONUS=0.5 alone was large enough (0.5 * 200 steps ~= 100) that a
                       # real training run learned to stand still and never improved tracking at
                       # all (confirmed by watching it render). This rebalances the same two
                       # existing terms rather than adding a new one.
FALL_PENALTY = 50.0    # one-time, on the step a fall is detected
DEFAULT_GOAL_CLIP = "data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep100.npz"
DEFAULT_GOAL_EMBODIMENT = "hexapod"


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def read_vision_goal(itm, md, encoder, offset, goal_path, horizon=1, max_pairs=None):
    """This project's own `vision_goal()` (`sim/control/close_loop_direct_froude.py`), copied for
    the same file-fragility reason as the constants above. See `b1_mujoco_env.py`'s copy for
    the full explanation, including why the horizon is 1 and not 5 -- identical here."""
    with np.load(os.path.join(ROOT, goal_path), allow_pickle=True) as gd:
        gframes = gd["frames"]
    n_pairs = max(1, len(gframes) - horizon)
    idx0 = (np.arange(n_pairs) if max_pairs is None else
            np.linspace(0, n_pairs - 1, min(n_pairs, max_pairs)).round().astype(int))
    n_pairs = len(idx0)
    idx1 = idx0 + horizon
    ge = encode_clip(encoder, gframes[np.concatenate([idx0, idx1])], 2).float()
    if offset is not None:
        ge = ge - offset.to(ge.device)
    g0, g1 = ge[:n_pairs], ge[n_pairs:]
    dev = next(itm.parameters()).device
    with torch.no_grad():
        z = itm(g0.to(dev), g1.to(dev))
        goals_per_pair = md.body(None, z)
    return goals_per_pair.mean(0).cpu().numpy()


class B1CoppeliaEnv:
    def __init__(self, ckpt_path, scene=None, goal_clip=DEFAULT_GOAL_CLIP,
                goal_embodiment=DEFAULT_GOAL_EMBODIMENT, horizon=200,
                port=23000, pid_p=300.0, pid_d=5.0, device="cpu"):
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        self.device = torch.device(device)
        ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
        cfg = from_checkpoint(ck["config"])
        self.md = MotionDecoder(cfg, {"b1": ACTION_DIM}).to(self.device).eval()
        self.md.load_state_dict(ck["md"], strict=False)
        self.proj = ActionProjector(cfg, action_dims_from(ck)).to(self.device).eval()
        self.proj.load_state_dict(ck["projector"])
        itm = InverseTransitionModel(cfg).to(self.device).eval()
        itm.load_state_dict(ck["itm"])
        for p in list(self.md.parameters()) + list(self.proj.parameters()) + list(itm.parameters()):
            p.requires_grad_(False)

        encoder = VJEPA2FrameEncoder(device="cpu", dtype=torch.float32)
        offset = offset_for(ck, goal_embodiment)
        self._goal_std_full = read_vision_goal(itm, self.md, encoder, offset, goal_clip)
        self.goal_scale = 1.0
        del encoder, itm
        torch.cuda.empty_cache()
        self.horizon = horizon
        self.t = 0

        self.client = RemoteAPIClient("localhost", port=port)
        self.sim = self.client.require("sim")
        settle(self.sim)
        self.sim.loadScene(os.path.join(ROOT, scene or "sim/env/b1_flat_convex.ttt"))
        settle(self.sim)
        if self.sim.getInt32Param(self.sim.intparam_dynamic_engine) != 0:
            raise RuntimeError("b1_flat_convex.ttt must use Bullet (engine 0)")

        sim = self.sim
        joints_by_name = {sim.getObjectAlias(h): h
                          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
        self.joints = [joints_by_name[name] for name in JOINT_ALIASES_SDK]
        shapes_by_name = {sim.getObjectAlias(h): h
                          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
        self.root = shapes_by_name[ROOT_ALIAS]
        self.pid_p, self.pid_d = pid_p, pid_d
        self._configure_joints()
        self.dt = None

    def _configure_joints(self):
        sim = self.sim
        neutral = il_to_sdk(DEFAULT_IL)
        for alias, handle, target in zip(JOINT_ALIASES_SDK, self.joints, neutral):
            segment = alias.split("_")[1]
            sim.setJointMode(handle, sim.jointmode_dynamic, 0)
            sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
            sim.setObjectInt32Param(handle, sim.jointintparam_motor_enabled, 1)
            sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_p, self.pid_p)
            sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_d, self.pid_d)
            sim.setJointMaxForce(handle, MAX_FORCE[segment])
            sim.setJointPosition(handle, float(target))       # F196's own fix: position AND target
            sim.setJointTargetPosition(handle, float(target))

    @torch.no_grad()
    @property
    def goal_std(self):
        return self._goal_std_full * self.goal_scale

    def set_goal_scale(self, scale):
        self.goal_scale = float(scale)

    def _reward(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.proj(a, "b1")
        pred = self.md.body(None, z).cpu().numpy()[0]
        distance = float(np.abs(pred - self.goal_std).sum())
        return float(np.exp(-TRACKING_SCALE * distance))

    def _obs(self):
        sim = self.sim
        qpos = np.asarray([sim.getJointPosition(h) for h in self.joints])
        qvel = np.asarray([sim.getJointVelocity(h) for h in self.joints])
        qx, qy, qz, qw = sim.getObjectQuaternion(self.root, sim.handle_world)
        lin, ang = sim.getObjectVelocity(self.root)
        return np.concatenate([qpos, qvel, [qw, qx, qy, qz], lin, ang]).astype(np.float32)

    def reset(self, seed=None):
        settle(self.sim)
        self._configure_joints()
        self.sim.setStepping(True)
        self.sim.startSimulation()
        for _ in range(10):     # brief settle, matching F196's warmup convention
            self.sim.step()
        self.dt = float(self.sim.getSimulationTimeStep())
        self.t = 0
        return self._obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = np.clip(il_to_sdk(DEFAULT_IL + ACTION_SCALE * action),
                         [JOINT_LIMITS[a.split("_")[1]][0] for a in JOINT_ALIASES_SDK],
                         [JOINT_LIMITS[a.split("_")[1]][1] for a in JOINT_ALIASES_SDK])
        for handle, value in zip(self.joints, target):
            self.sim.setJointTargetPosition(handle, float(value))
        self.sim.step()
        self.t += 1

        pos = self.sim.getObjectPosition(self.root, self.sim.handle_world)
        qx, qy, qz, qw = self.sim.getObjectQuaternion(self.root, self.sim.handle_world)
        upz = 1 - 2 * (qx * qx + qy * qy)
        height = float(pos[2])
        fell = (height < FALL_HEIGHT) or (upz < FALL_UPZ)
        # A fall no longer ends the episode -- see b1_mujoco_env.py for why (standing still was
        # reward-optimal once falling forfeited every remaining step's reward, even with a
        # correctly-scaled ground-truth velocity reward).
        done = self.t >= self.horizon
        tracking_reward = self._reward(action)
        reward = TRACKING_WEIGHT * tracking_reward + ALIVE_BONUS - (FALL_PENALTY if fell else 0.0)
        info = {"height": height, "upz": upz, "fell": fell, "tracking_reward": tracking_reward}
        if fell:
            self._configure_joints()
        return self._obs(), reward, done, info

    def close(self):
        settle(self.sim)


def main():
    """Smoke test: random actions, no policy, no training."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = B1CoppeliaEnv(args.ckpt)
    for ep in range(args.episodes):
        obs = env.reset()
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != expected {(OBS_DIM,)}"
        total_r, steps = 0.0, 0
        last_info = {}
        done = False
        while not done:
            action = rng.normal(0, 0.3, size=ACTION_DIM)
            obs, r, done, info = env.step(action)
            total_r += r; steps += 1; last_info = info
        print(f"episode {ep}: steps={steps}  total_reward={total_r:.2f}  "
             f"mean_reward={total_r / max(steps, 1):.3f}  fell={last_info.get('fell')}  "
             f"final_height={last_info.get('height', 0):.3f}  final_upz={last_info.get('upz', 0):.3f}")
    env.close()


if __name__ == "__main__":
    main()
