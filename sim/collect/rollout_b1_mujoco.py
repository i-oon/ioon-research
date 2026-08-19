"""Native-MuJoCo rollout of the B1 policy -> trajectory .npz (no ROS).

The policy walks cleanly only in its native MuJoCo physics; CoppeliaSim's engines
either freeze the imported base (MuJoCo engine) or can't reproduce the gait
(Newton/Bullet sim2sim gap). So we roll out here (correct motion + proprioception)
and render the trajectory in CoppeliaSim via render_b1_replay.py (same renderer as the
insect -> no render-style confound). Obs/control are ported verbatim from
b1_deployment/deploy_mujoco.py.

  python3 sim/collect/rollout_b1_mujoco.py --vx 0.4 --steps 300 --out data/b1_traj/walk_vx0.4.npz
  python3 sim/collect/rollout_b1_mujoco.py --vx 0.4 --schedule "1@0.4 0@0.2 1@0.4" --steps 300 ...
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import mujoco

MODEL = "/home/aria/Sim2Real-B1/b1_ws/src/b1_mujoco/model/b1_flat.xml"
CKPT = ("/home/aria/Sim2Real-B1/ppo_policy/logs/ppo_b1/sysid_real/"
        "with_ideal_gain/base_gait3/model_600.pt")

IL_TO_SDK = np.array([3, 0, 9, 6,  4, 1, 10, 7,  5, 2, 11, 8])
def sdk_to_il(v): return np.asarray(v)[IL_TO_SDK]
def il_to_sdk(v):
    out = np.empty(12); out[IL_TO_SDK] = np.asarray(v); return out

DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
DECIMATION = 4
GAIT_FREQ = 2.0        # base_gait3; base_1.7hz_sym was trained at 1.7, pass --gait_freq
FOOT_FORCE_THRESH = 1.0
SPAWN_Z = 0.50
_TOUCH_SDK_TO_IL_LEG = [1, 0, 3, 2]
HEAD_K = 0.5


def parse_schedule(spec, vx, vy, wz):
    """`"1@0.4 0@0.2 1@0.4"` -> [(1.0, 0.4), (0.0, 0.2), (1.0, 0.4)], scaling the base command.

    Identical syntax to `sim/collect/collect_ik.py`, so one schedule string can be handed to both
    robots and mean the same thing: rate is a multiplier on the commanded velocity, fraction is the
    share of the clip. Rate 0 commands a halt, which this policy holds standing rather than
    stopping dead.
    """
    segments = []
    for token in spec.split():
        rate, _, frac = token.partition("@")
        if not frac:
            raise SystemExit(f"schedule segment {token!r} needs rate@fraction, e.g. 1@0.4")
        segments.append((float(rate), float(frac)))
    if not segments:
        raise SystemExit("empty --schedule")
    total = sum(f for _, f in segments)
    return [(r * vx, r * vy, r * wz, f / total) for r, f in segments]


def command_plan(segments, n):
    """One (vx, vy, wz) row per policy step, so the loop reads a plan instead of branching."""
    plan = np.zeros((n, 3), np.float32)
    start = 0
    for i, (cvx, cvy, cwz, frac) in enumerate(segments):
        stop = n if i == len(segments) - 1 else min(n, start + int(round(frac * n)))
        plan[start:stop] = (cvx, cvy, cwz)
        start = stop
    return plan


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
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1 - 2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--gait_freq", type=float, default=GAIT_FREQ,
                    help="clock frequency in the observation. Must match the value the policy "
                         "was trained with (train_config.yaml: run.gait_freq) or the gait phase "
                         "the policy is tracking drifts against the one it is told about.")
    ap.add_argument("--schedule", type=str, default="",
                    help="piecewise pace as 'rate@fraction' segments, e.g. '1@0.4 0@0.2 1@0.4' to "
                         "walk, stand for a fifth of the clip, then walk. Rate multiplies "
                         "--vx/--vy/--wz. Same syntax as sim/collect/collect_ik.py.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=25, help="hold DEFAULT pose (settle) before policy")
    ap.add_argument("--policy_warmup", type=int, default=45,
                    help="run the policy UNLOGGED first, so the initial spawn->walk jump is cropped")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    actor = load_actor(args.checkpoint)
    use_clock = actor[0].in_features == 60
    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): m.sensor_adr[i]
           for i in range(m.nsensor)}

    d.qpos[0:3] = [0, 0, SPAWN_Z]; d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = il_to_sdk(DEFAULT_IL); d.ctrl[:] = il_to_sdk(DEFAULT_IL)
    mujoco.mj_forward(m, d)
    for _ in range(args.warmup * DECIMATION):
        mujoco.mj_step(m, d)

    plan = command_plan(parse_schedule(args.schedule, args.vx, args.vy, args.wz), args.steps) \
        if args.schedule else None
    last = np.zeros(12, np.float32); step_i = 0; heading_target = None
    L = {k: [] for k in ("base_pos", "base_quat", "joint_pos", "joint_vel",
                         "action", "command", "foot_contact")}
    for _i in range(args.policy_warmup + args.steps):
        # heading command (wz -> held heading), matches deploy
        _R = quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"]+4])
        cur_yaw = float(np.arctan2(_R[1, 0], _R[0, 0]))
        if heading_target is None:
            heading_target = cur_yaw
        # the plan covers the recorded steps only; the warmup holds the nominal command
        vx, vy, wz = (plan[min(_i - args.policy_warmup, len(plan) - 1)]
                      if plan is not None and _i >= args.policy_warmup
                      else (args.vx, args.vy, args.wz))
        heading_target += wz * (DECIMATION * m.opt.timestep)
        yaw_err = float(np.arctan2(np.sin(heading_target - cur_yaw), np.cos(heading_target - cur_yaw)))
        cmd = np.array([vx, vy, float(np.clip(HEAD_K * yaw_err, -1, 1))], np.float32)

        lin = d.sensordata[adr["base_linvel"]:adr["base_linvel"]+3]
        ang = d.sensordata[adr["base_angvel"]:adr["base_angvel"]+3]
        grav = quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"]+4]).T @ np.array([0., 0., -1.])
        jpos = sdk_to_il(d.qpos[7:19]) - DEFAULT_IL
        jvel = sdk_to_il(d.qvel[6:18])
        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"]+4])[_TOUCH_SDK_TO_IL_LEG]
        foot = (touch > FOOT_FORCE_THRESH).astype(np.float32)
        obs = np.concatenate([lin, ang, grav, cmd, jpos, jvel, last, foot]).astype(np.float32)
        if use_clock:
            t = step_i * DECIMATION * m.opt.timestep
            phi = 2 * np.pi * args.gait_freq * t + np.array([0., np.pi, np.pi, 0.])
            obs = np.concatenate([obs, np.sin(phi), np.cos(phi)]).astype(np.float32)

        with torch.no_grad():
            action = actor(torch.from_numpy(obs)).numpy()
        last = action
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        step_i += 1

        if _i < args.policy_warmup:                          # crop the spawn->walk transient
            continue
        L["base_pos"].append(d.qpos[0:3].copy())
        L["base_quat"].append(d.qpos[3:7].copy())          # (w, x, y, z)
        L["joint_pos"].append(d.qpos[7:19].copy())          # 12, SDK order
        L["joint_vel"].append(d.qvel[6:18].copy())
        L["action"].append(action.copy())                   # 12, IL order
        L["command"].append(cmd.copy())
        L["foot_contact"].append(foot.copy())

    out = {k: np.asarray(v, np.float32) for k, v in L.items()}
    bp = out["base_pos"]
    print(f"steps={len(bp)}  x_travel={bp[-1,0]-bp[0,0]:+.2f}m  y_drift={bp[-1,1]-bp[0,1]:+.2f}m  "
          f"z: start {bp[0,2]:.3f} min {bp[:,2].min():.3f} end {bp[-1,2]:.3f}  "
          f"({'WALKS' if bp[-1,2] > 0.35 and bp[-1,0]-bp[0,0] > 0.5 else 'FELL'})")
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(args.out, joint_order_sdk=np.array(
            [f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL") for s in ("hip", "thigh", "calf")]),
            dt=DECIMATION * m.opt.timestep, **out)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
