"""B1 behavior cloning -- proprioceptive, the surviving control after F216's four mechanisms.

**Four other mechanisms were tried and removed from this file after failing** (fine local-
perturbation DAgger, coarse-candidate distillation, an auxiliary model-consistency loss, frame-
stacked history) -- see `doc/FINDINGS.md`'s F216 for the full comparison table and why each one
failed specifically. Only `bc`/`eval` (mechanism 0, the plain-clone control) survive here; the
removed code is recoverable from git history if it's ever needed again, but F216 already captured
everything about what it did and why it didn't work.

    .venv/bin/python3 sim/control/clone_b1.py bc   --forward_only --out wm/runs/students/b1_bc_forward.pt
    .venv/bin/python3 sim/control/clone_b1.py bc   --out wm/runs/students/b1_bc_all.pt
    .venv/bin/python3 sim/control/clone_b1.py eval --student wm/runs/students/b1_bc_forward.pt

**Why proprioceptive, unlike the insect version.** `beh12_b1_ego_flat` already carries `joint_pos`/
`joint_vel`/`base_quat`/`base_pos` every frame -- the exact quantities `B1MuJoCoEnv._obs()` assembles
live. Cloning on that needs no VJEPA2 encoding at train or eval time and needs no CoppeliaSim: the
whole loop is MuJoCo only, reusing `B1MuJoCoEnv`'s already-validated reset/step/fall logic instead of
a second physics harness. Base linear+angular velocity isn't recorded directly (only position/
quaternion), so it's rebuilt by finite-differencing -- validated directly against `B1MuJoCoEnv`'s own
live `qvel` on a re-simulated trajectory before trusting it as training data: central-difference
linear velocity is accurate to 2.1% of its own std, angular to 18.1% (a real, stated imprecision --
angular finite-differencing is intrinsically noisier at this control rate, not a bug).

**Reuses `Student`/`verdict` from `teacher_student_insect.py` unchanged** -- the goal is each clip's
OWN recorded body motion (self-supervised, same-embodiment), exactly the insect's methodology: this
is an engine test on B1 alone, no cross-embodiment claim (that path -- reading the goal from another
body's video -- is F210's, already validated, and deliberately not exercised here). `body_goal_trimmed`
below is this file's own addition: the whole-clip mean (`body_goal`'s own convention) sits below the
true cruising Froude, dragged down by every clip's accel/decel edges -- `--trim_goal` averages over
the steady-state window instead, `trim_goal=0` reproducing the original convention exactly.

**A real train/eval physics gap, stated because it could explain a failure.** `beh12_b1_ego_flat` was
collected on `sim/assets/b1_mujoco/b1_flat.xml` (uniform placeholder joint physics --
`rollout_b1_mujoco.py:21`). `B1MuJoCoEnv` (used here for both `D_real` and `eval`) defaults to
`b1_flat_real.xml` (system-identified damping/friction -- F209 measured this is what makes an
open-loop gait work at all). So the student is cloned on state/action pairs generated under one
physics model and evaluated in a dynamically different one. `D_real` is computed under the SAME
model `eval` uses (not read off the clip's own recorded `base_pos`, which reflects the OTHER model),
so the bar is at least fair to the physics the student is actually judged in -- but the gap between
training and evaluation dynamics remains a live, undismissed hypothesis for any failure below.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
from teacher_student_insect import Student, verdict  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import B1, heading, load  # noqa: E402
from wm.policy.b1_mujoco_env import ACTION_SCALE, DEFAULT_IL, B1MuJoCoEnv, il_to_sdk  # noqa: E402

DATA = "data/egocentric/beh12_b1_ego_flat"
STEPS = 66  # matches the clips' own length
DATA_DT = 0.05  # 20 Hz -- every B1 dataset this project has (old and new) is recorded at this rate


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def base_velocity_fd(base_pos, base_quat, dt):
    """Central-difference world-frame linear + body-frame angular velocity, matching MuJoCo's own
    free-joint `qvel[0:6]` convention. Validated against a live re-simulated trajectory before use
    (see module docstring): 2.1%/18.1% mean-abs-error against true `qvel`, central difference
    (t-1, t+1) -- a one-step forward difference measured WORSE (8.5%/39.8%), confirmed directly, not
    assumed, so central is what's used here despite the larger nominal step.
    """
    n = len(base_pos)
    vel = np.zeros((n, 6), dtype=np.float32)
    for t in range(n):
        t0, t1 = max(0, t - 1), min(n - 1, t + 1)
        denom = max(1, t1 - t0) * dt
        vel[t, 0:3] = (base_pos[t1] - base_pos[t0]) / denom
        dq = quat_mul(quat_conj(base_quat[t0]), base_quat[t1])
        dq = dq / np.linalg.norm(dq)
        vel[t, 3:6] = 2.0 * dq[1:4] / denom
    return vel


def body_state(clip_path, yaw_err_input=False):
    """34-d proprioceptive state per frame, in exactly `B1MuJoCoEnv._obs()`'s convention:
    12 joint pos + 12 joint vel + 4 base quat + 6 base lin/ang vel.

    `yaw_err_input=True` appends a 35th number: heading error against the clip's OWN frame-0
    heading (a fixed target, matching how the expert's own external heading-hold controller works
    -- `rollout_b1_mujoco.py`'s `heading_target` starts at the reset heading and the correction it
    feeds the network is `heading_target - current_heading`, computed fresh every step). Every
    recorded action already reflects that correction's effect, but the correction SIGNAL itself was
    never part of the state the student saw -- this is the fix, not a new mechanism: it's the same
    offset-correction input the expert's own network was given, just finally exposed."""
    with np.load(clip_path, allow_pickle=True) as d:
        joint_pos = d["joint_pos"].astype(np.float32)
        joint_vel = d["joint_vel"].astype(np.float32)
        base_pos = d["base_pos"].astype(np.float64)
        base_quat = d["base_quat"].astype(np.float64)
        dt = float(d["dt"])
    base_vel = base_velocity_fd(base_pos, base_quat, dt)
    state = np.concatenate([joint_pos, joint_vel, base_quat.astype(np.float32), base_vel], axis=1)
    if yaw_err_input:
        h = heading(base_quat, "b1")
        yaw_err = np.arctan2(np.sin(h[0] - h), np.cos(h[0] - h)).astype(np.float32)
        state = np.concatenate([state, yaw_err[:, None]], axis=1)
    return state


def body_goal_trimmed(clip_path, channels, trim):
    """Same as `body_goal`, but averaged over the steady-state window only, dropping `trim` frames
    from each end of the clip. `body_goal`'s whole-clip mean sits BELOW the real cruising Froude,
    dragged down by the accel/decel edges every clip has -- verified directly on a real clip
    (`b1_ep2002.npz`): first-10-frame mean 0.106, middle-20-frame (steady state) mean 0.174,
    whole-clip mean 0.154, a 12% understatement of the speed the student is actually meant to
    reach. `trim=0` reproduces `body_goal` exactly."""
    motion = np.asarray(load(clip_path, B1)["body_motion"])[:, channels]
    if trim > 0 and len(motion) > 2 * trim:
        motion = motion[trim:-trim]
    return motion.mean(0)


def clone_b1(args, device):
    """Pure imitation: fit the student on B1's own recorded frames and commands, no grading of any
    kind. Also the control -- if this alone clears the bar, nothing past it would have been needed."""
    ck = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    if args.forward_only:
        keep = []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                if str(z["behaviour"]) == "speed":
                    keep.append(p)
        paths = keep
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    val_n = max(1, int(0.2 * len(paths)))
    val_paths = {paths[i] for i in order[:val_n]}
    print(f"cloning on {len(paths) - val_n} clips, {val_n} held out, from {args.data}")

    X, G, Y, V = [], [], [], []
    for p in paths:
        clip = load(p, B1)
        s = body_state(p, yaw_err_input=args.yaw_err_input)
        a = np.asarray(clip["actions"], dtype=np.float32)
        n = min(len(s), len(a))
        g = body_goal_trimmed(p, channels, args.trim_goal)
        X.append(torch.from_numpy(s[:n])); Y.append(torch.from_numpy(a[:n]))
        G.append(torch.from_numpy(g).float().expand(n, -1))
        V.append(torch.full((n,), p in val_paths))

    X = torch.cat(X).to(device); G = torch.cat(G).to(device)
    Y = torch.cat(Y).to(device); V = torch.cat(V).to(device)
    student = Student(X.shape[-1], G.shape[-1], Y.shape[-1]).to(device)
    student.mean.copy_(Y[~V].mean(0)); student.std.copy_(Y[~V].std(0).clamp_min(1e-6))
    target = (Y - student.mean) / student.std

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    best = {"v": float("inf"), "epoch": 0, "state": None}
    for epoch in range(args.epochs):
        student.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(student(X[~V], G[~V]), target[~V])
        loss.backward(); opt.step()
        if (epoch + 1) % args.eval_every == 0:
            student.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(student(X[V], G[V]), target[V]).item()
            if v < best["v"]:
                best = {"v": v, "epoch": epoch + 1,
                        "state": {k: t.detach().clone() for k, t in student.state_dict().items()}}
            print(f"  epoch {epoch + 1:4d}  train {loss.item():.4f}  held out {v:.4f}"
                  + ("   <- best" if v == best["v"] else ""))
    if best["state"] is not None:
        student.load_state_dict(best["state"])
    print(f"\n  best held out {best['v']:.4f} at epoch {best['epoch']}  =  R2 {1 - best['v']:+.3f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "val_mse": best["v"], "val_epoch": best["epoch"],
               "forward_only": bool(args.forward_only), "state_dim": X.shape[-1],
               "goal_dim": G.shape[-1], "action_dim": Y.shape[-1], "channels": channels,
               "data": args.data, "val_paths": sorted(os.path.basename(p) for p in val_paths),
               "trim_goal": int(args.trim_goal), "yaw_err_input": bool(args.yaw_err_input)}, out)
    print(f"-> {args.out}")


def seed_from_clip(env, clip_path):
    """Set MuJoCo state to a clip's own recorded frame 0, not the env's static default pose.

    **Why this matters, found by measuring, not assumed.** `rollout_b1_mujoco.py` runs
    `--policy_warmup 45` steps of real walking BEFORE recording starts (`_i < policy_warmup:
    continue`), so a clip's own `actions[0]` was applied to an already-moving body, mid-gait --
    not a freshly reset, standing one. Replaying it from `env.reset()`'s static pose measured a
    net displacement in the WRONG direction (-0.23 m against the clip's own recorded +1.25 m,
    same action sequence, same model file) -- not a sign bug, a mismatched initial condition.
    Base linear/angular velocity isn't recorded directly, so it's rebuilt with the same
    finite-difference formula `body_state()` uses (validated to 2.1%/18.1% mean-abs-error against
    live MuJoCo `qvel` elsewhere in this file).
    """
    with np.load(clip_path, allow_pickle=True) as d:
        base_pos = d["base_pos"].astype(np.float64)
        base_quat = d["base_quat"].astype(np.float64)
        joint_pos = d["joint_pos"].astype(np.float64)
        joint_vel = d["joint_vel"].astype(np.float64)
        dt = float(d["dt"])
    import mujoco
    mujoco.mj_resetData(env.m, env.d)
    env.d.qpos[0:3] = base_pos[0]
    env.d.qpos[3:7] = base_quat[0]
    env.d.qpos[7:19] = joint_pos[0]
    env.d.qvel[6:18] = joint_vel[0]
    base_vel0 = base_velocity_fd(base_pos[:3], base_quat[:3], dt)[0]  # fwd-diff from frames 0,1
    env.d.qvel[0:6] = base_vel0
    mujoco.mj_forward(env.m, env.d)
    env.air_time[:] = 0.0
    env.prev_contact[:] = False
    env.pos_hist = np.tile(env.d.qpos[0:3].copy(), (env.hist_len, 1))
    env.quat_hist = np.tile(env.d.qpos[3:7].copy(), (env.hist_len, 1))
    env.t = 0


def apply_action_unclipped(env, action):
    """Step physics exactly as `rollout_b1_mujoco.py` does -- clamp only the FINAL scaled target
    at the physical actuator range, never the raw action at +-1 first.

    **Why not `env.step()`.** `B1MuJoCoEnv.step()` clips the raw action to [-1, 1] before scaling,
    correct for a tanh-squashed RL actor but wrong here: this dataset's own recorded expert
    actions are unbounded (up to 3.5, 32% already exceed |1|, matching F203's already-documented
    "action-space corruption" pattern), and the student is trained to reproduce that same
    distribution (plain linear output head, no tanh). Feeding either through `env.step()`'s
    pre-clip truncates a third of the commanded targets and silently breaks the gait -- measured
    directly: replaying the clip's own recorded actions through `env.step()` gave -0.17 m to
    -0.50 m net displacement (wrong direction) against the clip's own recorded +1.25 m, on the
    SAME model file. This function reproduces instead.
    """
    import mujoco
    target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * np.asarray(action, dtype=np.float64))
    env.d.ctrl[:] = np.clip(target, env.m.actuator_ctrlrange[:, 0], env.m.actuator_ctrlrange[:, 1])
    # **Hold each action for the recorded data's own real-time duration (DATA_DT, 20 Hz), not
    # env.decimation's 50 Hz.** The recorded `action` array is already a 20 Hz-subsampled view of a
    # policy that natively decides at 50 Hz (env.decimation=4 x 5 ms timestep = 0.02 s); replaying
    # it one array-row per `env.decimation`-length hold plays the same 66-row sequence in 1.32 s
    # instead of the 3.3 s (66 x 0.05 s) it was actually recorded over -- found by checking why
    # every eval video this session played so short. This affects `D_real` and the student's own
    # rollout identically (both call this function), so it was never an unfair asymmetry between
    # them, but it did mean every distance number was measured over a 2.5x-compressed replay.
    for _ in range(round(DATA_DT / env.m.opt.timestep)):
        mujoco.mj_step(env.m, env.d)
    env.t += 1


def rollout_open_loop(env, actions, clip_path=None, capture=None, frames=None):
    """Replay a fixed action sequence through `env`'s own physics, return per-step head xyz.

    `clip_path` given: seed from that clip's own recorded frame-0 state (for `D_real` -- see
    `seed_from_clip`). Omitted: fresh `env.reset()` (for evaluating a student from a standing
    start, matching F133's insect precedent). `capture`: see `rollout_student`'s own docstring."""
    if clip_path is not None:
        seed_from_clip(env, clip_path)
    else:
        env.reset()
    heads = [env.d.qpos[0:3].copy()]
    if capture is not None:
        frames.append(capture())
    for a in actions:
        apply_action_unclipped(env, a)
        heads.append(env.d.qpos[0:3].copy())
        if capture is not None:
            frames.append(capture())
    return np.asarray(heads)


def rollout_student(env, student, goal, device, steps, capture=None, frames=None,
                    yaw_err_input=False):
    """`yaw_err_input=True` appends the live heading-error term to every observation, matching
    `body_state`'s own convention exactly: heading target is fixed at whatever heading the episode
    STARTS at (mirroring `rollout_b1_mujoco.py`'s own `heading_target = cur_yaw` at reset, for the
    wz=0 "speed" conditions this is trained on), recomputed fresh from `env.d.qpos[3:7]` each step.

    `capture`, if given, is a zero-arg callable returning one rendered frame -- physics-engine
    agnostic, so the same rollout works with either a MuJoCo-native renderer or a live CoppeliaSim
    camera posed from this env's own state (see `evaluate_b1`'s `coppelia_render`)."""
    obs = env.reset()
    g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    heads = [env.d.qpos[0:3].copy()]
    heading_target = float(heading(env.d.qpos[3:7][None].copy(), "b1")[0]) if yaw_err_input else None
    if capture is not None:
        frames.append(capture())
    for _ in range(steps):
        with torch.no_grad():
            if yaw_err_input:
                cur = float(heading(env.d.qpos[3:7][None].copy(), "b1")[0])
                yaw_err = np.arctan2(np.sin(heading_target - cur), np.cos(heading_target - cur))
                obs_in = np.concatenate([obs, [yaw_err]]).astype(np.float32)
            else:
                obs_in = obs
            s = torch.tensor(obs_in, dtype=torch.float32, device=device).unsqueeze(0)
            action = student.act(s, g)[0].cpu().numpy()
        apply_action_unclipped(env, action)  # not env.step() -- see its docstring
        obs = env._obs()
        heads.append(env.d.qpos[0:3].copy())
        if capture is not None:
            frames.append(capture())
    return np.asarray(heads)


def coppelia_render_setup(env, args):
    """Third-person CoppeliaSim camera, posed live from `env`'s own MuJoCo state each frame --
    physics stays entirely in MuJoCo (this never touches `env.d` except to read it); CoppeliaSim is
    only ever the camera, the same split `close_loop_b1_physics.py` already established. Camera
    is pinned once, at the episode's own starting position, and never re-aimed (`close_loop_b1_
    physics.py`'s own finding: a camera that follows the body removes the one cue -- the sliding
    background -- a viewer uses to judge whether it's actually moving).

    `--cam_fov 24`/`--floor_scale 3`: the allocentric convention `render_b1_replay.py` and
    `close_loop_b1_physics.py` both already use, measured (F104) to keep the whole robot in frame
    in every clip -- MuJoCo's own default renderer, checkered floor and free camera, does not."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
    from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle

    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    sim.loadScene(os.path.abspath(os.path.join(ROOT, args.scene)))
    settle(sim)
    jm = {sim.getObjectAlias(h): h
         for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [jm[a] for a in JOINT_ALIASES_SDK]
    sm = {sim.getObjectAlias(h): h
         for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root, cam = sm[ROOT_ALIAS], sim.getObject("/" + SENSOR)
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    root0 = np.array(sim.getObjectPosition(root, sim.handle_world))
    off_xy, cam_z = cam0[:2] - root0[:2], cam0[2]

    p0 = env.d.qpos[0:3].copy()
    sim.setObjectPosition(cam, sim.handle_world,
                          [float(p0[0]) + off_xy[0], float(p0[1]) + off_xy[1], cam_z])
    if args.floor_scale > 0:
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                 if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            # **`sim.scaleObjects` grows a shape without moving its centre** -- scaling the floor
            # 3x lifts its walking SURFACE upward (a box's centre stays put, so its top face rises
            # by half the added thickness), burying the robot's feet below the new floor level.
            # `render_b1_replay.py` already found and fixed this (F104); `close_loop_b1_physics.py`,
            # which this setup was otherwise copied from, is missing the fix -- caught here because
            # the feet were invisible in the rendered result, not by inspection.
            top = sim.getObject("/Floor")

            def surface():
                q = sim.getObjectPosition(top, sim.handle_world)
                bb = sim.getShapeBB(top)
                return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
            before = surface()
            sim.scaleObjects(floors, float(args.floor_scale), False)
            drop = surface() - before
            for h in floors:                   # put the walking surface back where it was
                q = sim.getObjectPosition(h, sim.handle_world)
                sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - drop])
    if args.cam_fov > 0:
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                float(np.deg2rad(args.cam_fov)))

    def render():
        p = env.d.qpos[0:3]
        w, x, y, z = env.d.qpos[3:7]
        sim.setObjectPosition(root, sim.handle_world, [float(p[0]), float(p[1]), float(p[2])])
        sim.setObjectQuaternion(root, sim.handle_world, [float(x), float(y), float(z), float(w)])
        for h, a in zip(joints, env.d.qpos[7:19]):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)
    return render


def evaluate_b1(args, device):
    ck = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(ck["state_dim"], ck["goal_dim"], ck["action_dim"]).to(device).eval()
    student.load_state_dict(ck["student"])
    channels = list(ck["channels"])

    goal_path = os.path.join(ROOT, args.goal_clip)
    # matches whatever trim the student was actually trained with (0 = body_goal's own whole-clip
    # mean, unchanged) -- feeding a differently-scaled goal than training would use is its own bug
    goal = body_goal_trimmed(goal_path, channels, int(ck.get("trim_goal", 0)))
    goal_clip = load(goal_path, B1)
    goal_actions = np.asarray(goal_clip["actions"], dtype=np.float32)

    # D_real: replay the goal clip's OWN recorded actions open-loop, through the SAME model/reset
    # `eval` uses (b1_flat_real.xml, via B1MuJoCoEnv) -- not the clip's own stored base_pos, which
    # reflects the different (uniform-physics) model it was originally collected on. See module
    # docstring: this is a real, stated train/eval physics gap.
    env = B1MuJoCoEnv(args.base_ckpt, goal_std=np.zeros(3, dtype=np.float32), horizon=len(goal_actions) + 5,
                      model_path=args.model_path)
    heads_real = rollout_open_loop(env, goal_actions, clip_path=goal_path)
    d_real = float(np.linalg.norm(heads_real[-1, :2] - heads_real[0, :2]))
    print(f"D_real (replayed under eval physics) = {d_real:.4f} m, bar = {0.5 * d_real:.4f} m, "
         f"goal {np.round(goal, 4)} from {os.path.basename(args.goal_clip)}")

    capture_fn, frames = None, []
    if args.video:
        env.reset()  # establish the canonical starting pose the camera gets pinned to below --
                     # rollout_student's own internal reset() returns to this same pose (F: B1
                     # MuJoCo reset is deterministic), so the pin stays valid once it starts
        capture_fn = coppelia_render_setup(env, args)
    heads = rollout_student(env, student, goal, device, args.steps, capture_fn, frames,
                            yaw_err_input=bool(ck.get("yaw_err_input", False)))
    if frames:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.video)) or ".", exist_ok=True)
        imageio.mimsave(os.path.join(ROOT, args.video), frames, fps=int(round(1 / DATA_DT)))
        print(f"video: {args.video}")
    v = verdict(heads, d_real)
    print(f"\n  travelled {v['distance']:.4f} m = {v['fraction']:.0%} of D_real"
         f"   upright {v['upright']} (min head z {v['min_z']:.4f} against {v['z0']:.4f})")
    print(f"  **{'PASS' if v['pass'] else 'FAIL'}**")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, head=heads.astype(np.float32), d_real=np.float32(d_real),
                        goal=goal.astype(np.float32),
                        **{k: v[k] for k in ("distance", "fraction", "upright", "pass")})
    print(f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("bc", "eval"))
    ap.add_argument("--base_ckpt", default="wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/teacher_b1.pt",
                    help="Only `cfg.body_channels` is read (or, for eval, used to construct "
                         "B1MuJoCoEnv with goal_std zeroed out) -- body_head's own fit quality "
                         "never matters here, so `teacher_b1.pt` (no `body_head_fit` metadata, "
                         "i.e. an unfit body_head) is fine.")
    ap.add_argument("--student", default="wm/runs/students/b1_bc_forward.pt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--goal_clip", default=f"{DATA}/b1_ep100.npz")
    ap.add_argument("--forward_only", action="store_true")
    ap.add_argument("--trim_goal", type=int, default=0,
                    help="bc only: drop this many frames from each end of a clip before averaging "
                         "its body_motion into the goal, to exclude accel/decel edges and target the "
                         "real steady-state Froude instead of a whole-clip average that understates "
                         "it. Default 0 reproduces the original convention exactly. eval "
                         "auto-reads this from the checkpoint, so it never needs to be passed there.")
    ap.add_argument("--yaw_err_input", action="store_true",
                    help="bc only: append heading error against the clip's own frame-0 heading as "
                         "a 35th state number, live-computed the same way at eval time. The expert's "
                         "own recorded actions already reflect an external heading-hold correction "
                         "(rollout_b1_mujoco.py's head_kp/head_ki, on by default) that the student "
                         "was never shown the input for -- this exposes that same signal, not a new "
                         "mechanism. eval auto-reads this from the checkpoint too.")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="bc: checkpoint to write, default wm/runs/students/b1_bc_forward.pt. "
                         "eval: results .npz to write, default derived from --student's own name.")
    ap.add_argument("--video", default=None,
                    help="optional .mp4 of the eval rollout, rendered through a live CoppeliaSim "
                         "camera (physics stays entirely in MuJoCo) -- needs CoppeliaSim running "
                         "on --port. MuJoCo's own renderer was tried first and dropped: its free "
                         "camera and checkered floor didn't keep the robot clearly framed.")
    ap.add_argument("--port", type=int, default=23000, help="CoppeliaSim ZMQ port, for --video")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt", help="for --video")
    ap.add_argument("--cam_fov", type=float, default=24.0,
                    help="for --video -- 24, not the scene's authored 15: the allocentric "
                         "convention already measured (F104) to keep the whole robot in frame")
    ap.add_argument("--floor_scale", type=float, default=3.0,
                    help="for --video, matches render_b1_replay.py/close_loop_b1_physics.py")
    ap.add_argument("--model_path", default=None,
                    help="override B1MuJoCoEnv's default model (b1_flat_real.xml, system-"
                         "identified). Pass sim/assets/b1_mujoco/b1_flat.xml (uniform placeholder "
                         "physics, what beh12_b1_ego_flat was actually collected on) to check "
                         "whether that matches the student's cloned data better.")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("wm/runs/students/b1_bc_forward.pt" if args.stage != "eval" else
                   f"results/wm/{os.path.splitext(os.path.basename(args.student))[0]}_eval.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "bc":
        clone_b1(args, device)
    else:
        evaluate_b1(args, device)


if __name__ == "__main__":
    main()
