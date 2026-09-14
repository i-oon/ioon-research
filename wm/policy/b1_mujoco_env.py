"""A real-physics (not imagined-rollout) B1 environment: reward from the world model
(`body_head(proj(action))` vs a Froude goal), real MuJoCo physics steps the world, proprioceptive
observation. Q21 step 3's actual mechanism -- deliberately NOT the thing F179 killed.

    .venv/bin/python3 wm/policy/b1_mujoco_env.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1.pt

**Why this is not F179's failure mode.** F179 trained a policy entirely inside the FTM's own
imagined rollout (~100 autoregressive steps, no real physics at all) and died of compounding
prediction error. Here, real MuJoCo (`mj_step`) supplies every next state -- the world model is
consulted only to score the action just taken, a single forward pass through `proj`+`body_head`,
never rolled forward. There is no rollout to compound.

**Why the reward is `body_head(proj(action))`, not ground-truth Froude, even though ground truth is
available in this MuJoCo sandbox.** At real deployment on a genuinely novel body there is no
ground-truth Froude to read off -- only whatever the world model estimates from a small amount of
babble data. Training against the ground truth here would validate a reward the deployed system
could never actually have. This environment exists to test whether the WM-only reward (the one
`reward_quality_gate_b1.py`/F195/F199/F201 spent this whole arc improving) is good enough to train
a real controller -- so it is used as the reward here on purpose, with ground truth kept only for a
diagnostic side-channel (`info["true_froude"]`), never for training.

**Observation is proprioceptive** (joint pos/vel, base orientation and velocity -- all free inside
MuJoCo), not vision. Running V-JEPA2 (a ~1B-parameter frozen ViT) every environment step is not
practical at the 10^5-10^6 steps PPO needs. This is a deliberate, named scope narrowing: the
resulting controller needs proprioceptive sensing at deployment, not only a camera -- a smaller
claim than this project's vision-only positioning elsewhere, and should be reported as such.

**The goal IS read from vision, once per environment construction, and this matters for the
claim.** An earlier version of this file hardcoded a made-up goal `(0.15, 0, 0)` -- disconnected
from the actual mechanism this project is built on. Corrected: the goal is now read via this
project's own `vision_goal()` convention (`sim/control/close_loop_direct_froude.py`) -- encode a
real reference clip's frames through the frozen V-JEPA2 encoder + ITM, read the resulting Froude
via `body_head`, average over every valid frame-pair in the clip. This is genuinely "watch another
body's video, derive what it's doing, drive B1 toward it" -- the ITM is not needed for scoring the
policy's own actions (see this file's own docstring on that), but it is exactly the right tool for
reading a goal from a demonstration, and this is a one-time cost at construction, not a per-step
one, so V-JEPA2's weight is irrelevant here.
"""
import argparse
import os
import sys

import numpy as np
import torch
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import (BODY_WINDOW_S, HEXAPOD, body_velocity,  # noqa: E402
                                load as load_clip, yaw_rate)
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

# **Inlined rather than imported** -- see `b1_coppelia_env.py` for why (a cross-file import to
# `collect_b1_cpg_babble.py` broke once already when that file was moved by a parallel session).
# Same trade-off applies: this is a second copy of the action-space convention, reconcile back to
# one source of truth once this pipeline stabilizes.
MODEL = os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat_real.xml")
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
IL_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])

# F203's ACTION_LO/ACTION_HI per-joint remap is REVERTED (F206): it was calibrated from
# beh12_b1_ego_flat's recorded action magnitudes (up to 3.59), assumed to be "the range a working
# gait needs" -- but that dataset comes from the Isaac Lab EXPERT POLICY (rollout_b1_mujoco.py),
# whose actor has an unbounded Gaussian output (no tanh), so values past +-1 are just what an
# unclipped policy's raw output looks like, not a physical requirement. Verified directly: the real
# CPG gait (collect_b1_cpg_babble.py's own, default amplitude, values within [-1, 1]) produces 2.12m
# of real forward travel / 5s (Froude 0.196) fed through the plain `DEFAULT_IL + ACTION_SCALE *
# action` mapping below -- and produces ~0 Froude when the SAME action is remapped through the now-
# removed scale_action() first. That remap was corrupting every action fed to the physics, on every
# RL run since it was introduced.


def il_to_sdk(v):
    out = np.empty(12)
    out[IL_TO_SDK] = np.asarray(v)
    return out


OBS_DIM = 12 + 12 + 4 + 6   # joint pos, joint vel, base quat, base lin+ang vel
ACTION_DIM = 12
# **Stability criteria, matched to the two systems known to work on this task.** Hu et al.'s
# egocentric self-model aborts on roll or pitch past 30 degrees; Isaac Lab's velocity task
# terminates on `bad_orientation` past ~46 degrees AND carries a large continuous flat-orientation
# penalty (-2.5, among its heaviest). Both treat ORIENTATION as the primary criterion and both
# TERMINATE on it. An earlier version of this file removed termination entirely and kept only a
# permissive height check, which let a ~50-degree crouch lunge forward and fall twice per 4s episode
# while the reward paid for the forward speed that the lunge produced.
FALL_HEIGHT = 0.30          # secondary: a collapse. Kept, but no longer the main criterion.
FALL_UPZ = 0.6              # secondary, ~53 degrees -- more permissive than either reference
TILT_LIMIT_DEG = 40.0       # PRIMARY: terminate past this, between Hu et al.'s 30 and Isaac's 46
ORIENTATION_WEIGHT = 2.5    # continuous penalty on being off-level, Isaac Lab's own weight
ALIVE_BONUS = 0.1           # per step, unconditional -- kept small on purpose, see below
TRACKING_SCALE = 1.0        # exp(-TRACKING_SCALE * |pred-goal|), bounded to (0, 1] -- calibrated to
                            # body_head's own output scale (typical distance ~1.6-5)
TRACKING_SCALE_TRUE_FROUDE = 15.0  # true_froude mode's distances live in real dimensionless Froude
                            # units (typical ~0-0.2) -- TRACKING_SCALE alone would give exp(-0.16)
                            # = 0.85 for literally standing still, leaving almost no reward gradient
                            # between "did nothing" and "hit the goal exactly" (measured directly:
                            # a 50-update run stayed flat at 0.85 from update 0)
TRACKING_WEIGHT = 5.0       # multiplies the bounded tracking term -- the actual "strengthen the
                            # Froude reward" fix
FALL_PENALTY = 50.0         # one-time, on the step a fall is detected
AIR_TIME_THRESHOLD = 0.5    # seconds -- matches velocity_env_cfg.py's own feet_air_time threshold
AIR_TIME_WEIGHT = 0.1       # matches velocity_env_cfg.py's own weight; sparse (nonzero only at the
                            # step a foot lands), so this stays a small shaping term next to
                            # TRACKING_WEIGHT/ALIVE_BONUS, not a term that could dominate them
# **Two real bugs found in sequence, both from watching actual training runs, not guessed.**
# (1) A first attempt added ALIVE_BONUS=1.0 on top of the RAW, unbounded abs-distance tracking
# cost -- confirmed to still make dying fast reward-optimal (mean_len collapsed 37.9->6.9). Fixed
# by squashing the tracking term through exp(-distance), bounding it to (0, 1].
# (2) With that fixed, a full 50-update run showed `mean_return`/`mean_len` climbing to the full
# 200-step horizon while `tracking` (the raw exp(-distance) term, logged separately for exactly
# this reason) stayed FLAT at ~0.04 the entire time -- confirmed by watching the policy render:
# it was standing still. Cause: `ALIVE_BONUS=0.5` alone, over 200 steps, already nets ~100 reward
# regardless of tracking quality -- standing still is cheap, safe, and reward-competitive, so PPO
# never had to improve tracking to look successful. Fix: shrink ALIVE_BONUS to a small,
# still-positive floor (prevents the ORIGINAL die-fast bug from recurring) and multiply the
# tracking term by TRACKING_WEIGHT so it actually dominates the reward when achieved well --
# rebalancing the existing terms, not adding a new one.
DEFAULT_GOAL_CLIP = "data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep100.npz"
DEFAULT_GOAL_EMBODIMENT = "hexapod"
GOAL_HORIZON = 1            # frame spacing for reading the goal -- matches what every stage that
                            # fits the reading path was trained on (see read_vision_goal)


def read_vision_goal(itm, md, encoder, offset, goal_path, horizon=1, max_pairs=None):
    """This project's own `vision_goal()` (`sim/control/close_loop_direct_froude.py`), copied
    rather than imported for the same file-fragility reason as the constants above. Encodes the
    goal clip's own frames -- no recorded/label number involved anywhere -- and returns the
    clip-averaged Froude, already in the same standardized space `body_head` predicts in (so it
    can be compared to `pred` directly, no separate mean/std normalization needed).

    **`horizon=1`, not 5.** Every stage that fits this reading path trains on ADJACENT frame pairs
    (the pretrain inverse model, stage-1 adaptation, the projector, the body head). Reading at 5
    deploys the head on a spacing it never saw: measured across 48 source clips, reading at 1 instead
    of 5 cuts the goal-read error roughly 30-40%. The 5 originally came from the planner's rollout
    depth, where it is correct, and leaked here through a shared flag -- those are now separate.

    **`max_pairs=None` (all pairs).** Subsampling was added to make a one-time cost cheaper, but it
    is the mean of a noisy per-pair estimate: 300 random 12-pair draws of one clip gave a median
    error of 0.046 against 0.049 for the full average, spanning 0.019-0.096 -- i.e. it buys variance,
    not accuracy. The result is cached anyway, so pay the cost once and average everything."""
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


def compute_goal_std(ckpt_path, goal_clip=DEFAULT_GOAL_CLIP, goal_embodiment=DEFAULT_GOAL_EMBODIMENT):
    """The vision-goal read, standalone -- so `VecB1MuJoCoEnv` can do it ONCE in the parent process
    and hand every worker the resulting number, instead of each of `n_envs` workers independently
    loading and running the same ~1B-parameter encoder on the same clip. Cached on disk too: a
    single encode of this clip through the encoder takes 5+ minutes on CPU alone, and this exact
    (ckpt, clip) pair always produces the same number -- no reason to pay that cost more than once
    ever, across every future run."""
    # the read settings are part of the key: a goal cached at one horizon is simply a different
    # number, and serving it for another is the stale-cache trap this project has hit before
    cache = os.path.join(ROOT, "results/wm/cache",
                         f"goal_std_{os.path.basename(ckpt_path)}_{os.path.basename(goal_clip)}"
                         f"_h{GOAL_HORIZON}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    md = MotionDecoder(cfg, {"b1": ACTION_DIM}).eval()
    md.load_state_dict(ck["md"], strict=False)
    itm = InverseTransitionModel(cfg).eval()
    itm.load_state_dict(ck["itm"])
    encoder = VJEPA2FrameEncoder(device="cpu", dtype=torch.float32)
    offset = offset_for(ck, goal_embodiment)
    goal_std = read_vision_goal(itm, md, encoder, offset, goal_clip, horizon=GOAL_HORIZON)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, goal_std)
    return goal_std


class B1MuJoCoEnv:
    def __init__(self, ckpt_path, model_path=None, goal_clip=DEFAULT_GOAL_CLIP,
                goal_embodiment=DEFAULT_GOAL_EMBODIMENT, horizon=200, decimation=4, device="cpu",
                reward_mode="wm", goal_std=None):
        # reward_mode="true_froude": diagnostic only -- rewards real measured base velocity
        # (never available on a genuinely novel body) instead of the WM's action-only estimate.
        # Isolates whether the RL/env stack can learn to walk at all when given a reward that does
        # see real motion, separate from whether body_head(proj(action)) is a good enough proxy.
        assert reward_mode in ("wm", "true_froude")
        self.reward_mode = reward_mode
        self.device = torch.device(device)
        ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
        cfg = from_checkpoint(ck["config"])
        self.md = MotionDecoder(cfg, {"b1": ACTION_DIM}).to(self.device).eval()
        self.md.load_state_dict(ck["md"], strict=False)
        self.proj = ActionProjector(cfg, action_dims_from(ck)).to(self.device).eval()
        self.proj.load_state_dict(ck["projector"])
        for p in list(self.md.parameters()) + list(self.proj.parameters()):
            p.requires_grad_(False)

        # goal_std passed in (VecB1MuJoCoEnv's shared, precomputed value) skips loading VJEPA2 and
        # the ITM entirely -- neither is needed anywhere else in this env. Stored as the full,
        # un-curriculum'd target; `goal_scale` (1.0 = full difficulty, settable via
        # `set_goal_scale`) lets training start on an easy near-zero target and ramp up, rather
        # than the fixed, moderately-hard target every rollout has faced so far.
        self._goal_std_full = goal_std if goal_std is not None else compute_goal_std(
            ckpt_path, goal_clip, goal_embodiment)
        self.goal_scale = 1.0

        if self.reward_mode == "true_froude":
            # same clip's own recorded motion, in the dimensionless Froude space body_velocity/
            # yaw_rate already share across embodiments -- comparable to B1's true_froude directly.
            self._goal_true_full = load_clip(os.path.join(ROOT, goal_clip), HEXAPOD)["body_motion"].mean(0)

        self.horizon = horizon
        self.decimation = decimation

        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, model_path or MODEL))
        self.d = mujoco.MjData(self.m)
        self.dt = decimation * self.m.opt.timestep
        self.t = 0

        # body_velocity/yaw_rate smooth over one stride (BODY_WINDOW_S=1.0s) by convolving with a
        # ~50-sample kernel (at this dt) -- calling them with only the 2 samples spanning a single
        # control step (as an earlier version of this file did) convolves a 2-sample signal with a
        # 50-wide kernel, diluting the result by roughly the window size (measured: a real walking
        # gait's true_froude came back ~0.0025 through this bug, ~0.196 computed correctly over a
        # full trajectory). Fix: keep a rolling window of real position/quat history and recompute
        # over the full window every step, taking the most recent (rightmost) smoothed value.
        self.hist_len = max(3, round(BODY_WINDOW_S / self.dt)) + 5
        self.pos_hist = None
        self.quat_hist = None

        # Foot-contact sensors, present in b1_flat_real.xml already (collect_b1_cpg_babble.py's own
        # foot_contact recording uses these). feet_air_time rewards a foot for how long it stayed
        # airborne before its next touchdown -- the one mechanism in the reference Isaac Lab config
        # (velocity_env_cfg.py) not yet tried here, and the only one of its reward terms that
        # directly concerns lifting feet rather than tracking or survival.
        sensor_adr = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i): self.m.sensor_adr[i]
                     for i in range(self.m.nsensor)}
        self.foot_adr = sensor_adr["FR_touch"]
        self.air_time = np.zeros(4, dtype=np.float32)
        self.prev_contact = np.zeros(4, dtype=bool)

    @property
    def goal_std(self):
        return self._goal_std_full * self.goal_scale

    @property
    def goal_true(self):
        return self._goal_true_full * self.goal_scale

    def set_goal_scale(self, scale):
        self.goal_scale = float(scale)

    @torch.no_grad()
    def _reward(self, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.proj(a, "b1")
        pred = self.md.body(None, z).cpu().numpy()[0]
        distance = float(np.abs(pred - self.goal_std).sum())
        return float(np.exp(-TRACKING_SCALE * distance))   # bounded (0, 1], see TRACKING_SCALE above

    def _obs(self):
        return np.concatenate([
            self.d.qpos[7:19], self.d.qvel[6:18], self.d.qpos[3:7], self.d.qvel[0:6],
        ]).astype(np.float32)

    def _place_pose(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0, 0, 0.56]
        self.d.qpos[3:7] = [1, 0, 0, 0]
        # qpos must be in SDK order (matching the XML's own joint definition order) like ctrl is,
        # or the actuators snap hard to fix the mismatch on the next step.
        self.d.qpos[7:19] = il_to_sdk(DEFAULT_IL)
        self.d.ctrl[:] = il_to_sdk(DEFAULT_IL)
        mujoco.mj_forward(self.m, self.d)
        self.air_time[:] = 0.0
        self.prev_contact[:] = False
        # pre-fill the history with the resting pose repeated -- gives a full, valid window from
        # step 1 (near-zero velocity at rest is the correct value there, not an artifact)
        self.pos_hist = np.tile(self.d.qpos[0:3].copy(), (self.hist_len, 1))
        self.quat_hist = np.tile(self.d.qpos[3:7].copy(), (self.hist_len, 1))

    def reset(self, seed=None):
        self._place_pose()
        self.t = 0
        return self._obs()

    def _feet_air_time_reward(self):
        touch = np.asarray(self.d.sensordata[self.foot_adr:self.foot_adr + 4])
        contact = touch > 1.0
        first_contact = contact & ~self.prev_contact
        reward = float(np.sum(first_contact * (self.air_time - AIR_TIME_THRESHOLD)))
        self.air_time = np.where(contact, 0.0, self.air_time + self.dt)
        self.prev_contact = contact
        return reward

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        self.d.ctrl[:] = np.clip(target, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        pos0, quat0 = self.d.qpos[0:3].copy(), self.d.qpos[3:7].copy()
        positions, quats = [pos0], [quat0]
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)
        positions.append(self.d.qpos[0:3].copy())
        quats.append(self.d.qpos[3:7].copy())
        self.t += 1

        height = float(self.d.qpos[2])
        # up.z: the z-component of the body frame's own up vector, from the base quaternion
        w, x, y, z = self.d.qpos[3:7]
        upz = 1 - 2 * (x * x + y * y)
        tilt_deg = float(np.degrees(np.arccos(np.clip(upz, -1.0, 1.0))))
        fell = (tilt_deg > TILT_LIMIT_DEG) or (height < FALL_HEIGHT) or (upz < FALL_UPZ)
        # **A fall ends the episode again, as it does in both reference systems.** It was removed
        # here once, because the policy converged to standing still -- but that was caused by the
        # action-space corruption and the broken per-step velocity read (both since fixed), which
        # left the tracking reward flat and gave the policy nothing to climb. With a reward that
        # discriminates, termination is what stops a falling-forward lunge from out-earning a gait.
        done = fell or (self.t >= self.horizon)

        # roll the real history window forward one sample, recompute over the FULL window (not
        # just this step's 2 samples -- see hist_len's comment above), take the latest value
        self.pos_hist = np.roll(self.pos_hist, -1, axis=0)
        self.pos_hist[-1] = positions[-1]
        self.quat_hist = np.roll(self.quat_hist, -1, axis=0)
        self.quat_hist[-1] = quats[-1]
        true_fr = np.concatenate([
            body_velocity(self.pos_hist, self.quat_hist, self.dt, "b1"),
            yaw_rate(self.quat_hist, self.dt, "b1", height),
        ], axis=1)[-1]

        # `tracking_reward` itself stays the pure, bounded (0, 1] score -- interpretable, and what
        # gets logged. TRACKING_WEIGHT is applied only when composing the trained reward, so a
        # perfectly-tracking step now nets TRACKING_WEIGHT (5.0) against ALIVE_BONUS's 0.1 --
        # tracking dominates, survival alone no longer does.
        if self.reward_mode == "true_froude":
            distance = float(np.abs(true_fr - self.goal_true).sum())
            tracking_reward = float(np.exp(-TRACKING_SCALE_TRUE_FROUDE * distance))
        else:
            tracking_reward = self._reward(action)
        air_time_reward = self._feet_air_time_reward()
        # continuous cost for being off-level, in addition to the termination above: Isaac Lab
        # carries both, and the pair is what makes a crouched, pitched posture unprofitable rather
        # than merely risky.
        off_level = 1.0 - max(upz, 0.0)
        reward = (TRACKING_WEIGHT * tracking_reward + ALIVE_BONUS + AIR_TIME_WEIGHT * air_time_reward
                 - ORIENTATION_WEIGHT * off_level - (FALL_PENALTY if fell else 0.0))
        info = {"true_froude": true_fr, "height": height, "upz": upz, "fell": fell,
               "tilt_deg": tilt_deg, "tracking_reward": tracking_reward,
               "air_time_reward": air_time_reward}
        if fell:
            self._place_pose()
        return self._obs(), reward, done, info


def _worker(pipe, ckpt_path, kwargs):
    # One BLAS thread per worker. Torch defaults to one thread per core (16 here), so N workers ask
    # for N*16 threads on 16 cores and the machine spends its time context-switching instead of
    # stepping physics -- measured at load average 177 on a 32-core box with 16 workers. Every model
    # a worker runs is a small MLP that gains nothing from intra-op parallelism anyway.
    torch.set_num_threads(1)
    env = B1MuJoCoEnv(ckpt_path, **kwargs)
    while True:
        cmd, payload = pipe.recv()
        if cmd == "reset":
            pipe.send(env.reset())
        elif cmd == "step":
            obs, reward, done, info = env.step(payload)
            if done:
                obs = env.reset()  # auto-reset, standard vec-env convention
            pipe.send((obs, reward, done, info))
        elif cmd == "set_goal_scale":
            env.set_goal_scale(payload)
            pipe.send(None)
        elif cmd == "close":
            pipe.close()
            return


class VecB1MuJoCoEnv:
    """`n_envs` independent B1MuJoCoEnv instances in separate processes, stepped together --
    collecting PPO rollouts from one environment at a time is the reason training never accumulated
    enough real steps to discover a coordinated gait (the reference Isaac Lab policy used 4096
    parallel envs; a single MuJoCo env cannot match that throughput no matter how long it runs)."""

    def __init__(self, ckpt_path, n_envs, **kwargs):
        import multiprocessing as mp
        # spawn, not fork: forking after the parent has touched CUDA (e.g. torch.cuda.is_available())
        # crashes every worker with "Cannot re-initialize CUDA in forked subprocess" the moment it
        # loads its own VJEPA2 encoder.
        ctx = mp.get_context("spawn")
        self.n = n_envs
        if kwargs.get("goal_std") is None:
            print(f"computing the vision goal once (not {n_envs} times)...", flush=True)
            kwargs["goal_std"] = compute_goal_std(
                ckpt_path, kwargs.get("goal_clip", DEFAULT_GOAL_CLIP),
                kwargs.get("goal_embodiment", DEFAULT_GOAL_EMBODIMENT))
        self.pipes, worker_pipes = zip(*(ctx.Pipe() for _ in range(n_envs)))
        self.procs = [ctx.Process(target=_worker, args=(wp, ckpt_path, kwargs), daemon=True)
                     for wp in worker_pipes]
        for p in self.procs:
            p.start()

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return np.stack([pipe.recv() for pipe in self.pipes])

    def set_goal_scale(self, scale):
        for pipe in self.pipes:
            pipe.send(("set_goal_scale", scale))
        for pipe in self.pipes:
            pipe.recv()

    def step(self, actions):
        for pipe, a in zip(self.pipes, actions):
            pipe.send(("step", a))
        results = [pipe.recv() for pipe in self.pipes]
        obs, reward, done, info = zip(*results)
        return np.stack(obs), np.array(reward, dtype=np.float32), np.array(done), list(info)

    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for p in self.procs:
            p.join()


def main():
    """Smoke test: random actions, no policy, no training -- confirms the env runs, rewards are
    sane numbers, and episodes terminate for the right reasons before any RL code touches it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = B1MuJoCoEnv(args.ckpt)
    for ep in range(args.episodes):
        obs = env.reset()
        assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != expected {(OBS_DIM,)}"
        total_r, steps = 0.0, 0
        last_info = {}
        done = False
        while not done:
            action = rng.normal(0, 0.3, size=ACTION_DIM)  # random, matches babble's own noise scale
            obs, r, done, info = env.step(action)
            total_r += r
            steps += 1
            last_info = info
        print(f"episode {ep}: steps={steps}  total_reward={total_r:.2f}  "
             f"mean_reward={total_r / max(steps, 1):.3f}  fell={last_info.get('fell')}  "
             f"final_height={last_info.get('height', 0):.3f}  final_upz={last_info.get('upz', 0):.3f}  "
             f"true_froude={last_info.get('true_froude')}")


if __name__ == "__main__":
    main()
