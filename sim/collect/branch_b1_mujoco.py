"""One saved MuJoCo state, several commands: the counterfactual branch for the B1.

    .venv/bin/python3 sim/collect/branch_b1_mujoco.py --prefix 60 --branch_steps 30 \\
        --out data/allocentric/cf_confirm --arms forward=0.4,0,0 turn=0.4,0,0.6 side=0,0.4,0

**The B1 can do what the insect cannot: return to a state exactly.** `mjSTATE_INTEGRATION` restores
the solver warmstart and applied forces alongside `qpos`/`qvel`, and two runs from it are
bit-identical. **`mjSTATE_FULLPHYSICS` is not enough** -- it is what "save positions and velocities"
means, it leaves 2.83e-2 rad of drift over five steps, and that is larger than the one-step
counterfactual signal. An experiment built on it measures its own reset error.

**The policy's state is more than the physics.** `last` action and the gait clock `step_i` enter the
observation, so both are saved and restored with the physics; branching without them is a different
experiment that looks like this one.

Every arm writes the **shared prefix followed by its own branch**, so the trajectories can be
replayed and merged side by side and the split is visible rather than asserted. Output matches
`rollout_b1_mujoco.py`, so `render_b1_replay.py` reads it unchanged.
"""
import argparse
import importlib.util as u
import os

import mujoco
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
spec = u.spec_from_file_location("r", os.path.join(HERE, "rollout_b1_mujoco.py"))
R = u.module_from_spec(spec)
try:
    spec.loader.exec_module(R)
except SystemExit:
    pass

STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def build_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    ws = [k for k in sd if k.startswith("actor.") and k.endswith("weight")]
    mods = []
    for i, k in enumerate(ws):
        mods.append(nn.Linear(sd[k].shape[1], sd[k].shape[0]))
        if i < len(ws) - 1:
            mods.append(nn.ELU())
    net = nn.Sequential(*mods)
    with torch.no_grad():
        j = 0
        for m_ in net:
            if isinstance(m_, nn.Linear):
                m_.weight.copy_(sd[ws[j]])
                m_.bias.copy_(sd[ws[j].replace("weight", "bias")])
                j += 1
    return net.eval(), sd[ws[0]].shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=R.MODEL)
    ap.add_argument("--ckpt", default=R.CKPT)
    ap.add_argument("--prefix", type=int, default=60, help="shared frames before the branch")
    ap.add_argument("--branch_steps", type=int, default=30)
    ap.add_argument("--arms", nargs="+", required=True, metavar="NAME=vx,vy,wz")
    ap.add_argument("--repeat_first", action="store_true", default=True,
                    help="also write <first>_repeat, the same commands twice; the noise floor, "
                         "which on MuJoCo should come out exactly zero and is worth proving rather "
                         "than assuming")
    ap.add_argument("--gait_freq", type=float, default=R.GAIT_FREQ)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): m.sensor_adr[i]
           for i in range(m.nsensor)}
    actor, n_obs = build_actor(args.ckpt)

    def obs_of(cmd, last, step_i):
        lin = d.sensordata[adr["base_linvel"]:adr["base_linvel"] + 3]
        ang = d.sensordata[adr["base_angvel"]:adr["base_angvel"] + 3]
        grav = R.quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"] + 4]).T @ \
            np.array([0., 0., -1.])
        jpos = R.sdk_to_il(d.qpos[7:19]) - R.DEFAULT_IL
        jvel = R.sdk_to_il(d.qvel[6:18])
        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"] + 4])[R._TOUCH_SDK_TO_IL_LEG]
        foot = (touch > R.FOOT_FORCE_THRESH).astype(np.float32)
        o = np.concatenate([lin, ang, grav, cmd, jpos, jvel, last, foot]).astype(np.float32)
        if len(o) < n_obs:
            t = step_i * R.DECIMATION * m.opt.timestep
            phi = 2 * np.pi * args.gait_freq * t + np.array([0., np.pi, np.pi, 0.])
            o = np.concatenate([o, np.sin(phi), np.cos(phi)]).astype(np.float32)
        return o

    def step(cmd, last, step_i, log):
        a = actor(torch.from_numpy(obs_of(cmd, last, step_i))).detach().numpy()
        tgt = R.il_to_sdk(R.DEFAULT_IL + R.ACTION_SCALE * a)
        d.ctrl[:] = np.clip(tgt, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(R.DECIMATION):
            mujoco.mj_step(m, d)
        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"] + 4])[R._TOUCH_SDK_TO_IL_LEG]
        log["base_pos"].append(d.qpos[0:3].copy())
        log["base_quat"].append(d.qpos[3:7].copy())
        log["joint_pos"].append(d.qpos[7:19].copy())
        log["joint_vel"].append(d.qvel[6:18].copy())
        log["action"].append(a.copy())
        log["command"].append(np.asarray(cmd, np.float32).copy())
        log["foot_contact"].append((touch > R.FOOT_FORCE_THRESH).astype(np.float32))
        return a

    d.qpos[0:3] = [0, 0, R.SPAWN_Z]; d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = R.il_to_sdk(R.DEFAULT_IL); d.ctrl[:] = R.il_to_sdk(R.DEFAULT_IL)
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)

    fwd = np.array([0.4, 0., 0.], np.float32)
    keys = ("base_pos", "base_quat", "joint_pos", "joint_vel", "action", "command", "foot_contact")
    shared = {k: [] for k in keys}
    last, si = np.zeros(12, np.float32), 0
    for _ in range(args.prefix):
        last = step(fwd, last, si, shared); si += 1

    n = mujoco.mj_stateSize(m, STATE)
    S = np.empty(n); mujoco.mj_getState(m, d, S, STATE)
    LAST, SI = last.copy(), si

    os.makedirs(os.path.join(os.path.dirname(HERE), "..", args.out), exist_ok=True) \
        if not os.path.isabs(args.out) else os.makedirs(args.out, exist_ok=True)
    specs = [(s.split("=", 1)[0], np.array([float(x) for x in s.split("=", 1)[1].split(",")],
                                           np.float32)) for s in args.arms]
    if args.repeat_first:
        specs.append((specs[0][0] + "_repeat", specs[0][1].copy()))

    for name, cmd in specs:
        mujoco.mj_setState(m, d, S, STATE); mujoco.mj_forward(m, d)
        log = {k: list(v) for k, v in shared.items()}
        last, si = LAST.copy(), SI
        for _ in range(args.branch_steps):
            last = step(cmd, last, si, log); si += 1
        out = {k: np.asarray(v, np.float32) for k, v in log.items()}
        path = os.path.join(args.out, f"b1_{name}.npz")
        np.savez_compressed(path, dt=R.DECIMATION * m.opt.timestep, branch=args.prefix,
                            joint_order_sdk=np.array(
                                [f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL")
                                 for s in ("hip", "thigh", "calf")]), **out)
        print(f"  {path}  {len(out['base_pos'])} frames, branch at {args.prefix}, cmd {cmd}")


if __name__ == "__main__":
    main()
