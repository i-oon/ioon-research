"""Reach a body-motion goal on a robot with **no recorded behaviours to choose from**.

    .venv/bin/python3 sim/control/plan_without_library.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0_bodyfit_proj.pt \\
        --projector wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0.pt

**Why this exists, and what every earlier result did not test.** F136 selects among twelve recorded
B1 clips and reaches 70% cross-embodiment. Those clips already contain working behaviours: **the
shared coordinate picks the right one and the clip supplies the "how".** Remove the library and the
coordinate alone cannot say which joint sequence produces a given body motion on a body it has never
controlled. **That gap is what a world model is for**, and F135's finding that the rollout adds
nothing was measured in the one setting where a library had already done the rollout's job.

So: a goal in the shared coordinate, read from an **insect's** frames; a bank of **sampled** action
sequences that no recording contains; and two ways to choose among them.

    condition 1   score = || body_head(ITM(e_t, FDM rolled h steps on proj(a))) - goal ||
                  planning over futures that do not exist as clips -- the world model's actual job

    condition 2   score = || body_head(proj(a)) - goal ||
                  the same goal and the same coordinate with no prediction of the future at all

**Both conditions see the identical sample bank**, so the comparison is the scoring rule and nothing
else. A third row picks uniformly at random from that bank, which is the floor any rule has to
clear.

**The winner of each rule is then executed in MuJoCo and the body motion it actually produced is
measured.** Nothing here is scored on a model's own prediction: the number reported is what the
robot did.

**Pass bar, fixed before running:** condition 1 lands nearer its goal than condition 2. If they tie,
the world model is not load-bearing even here, and the shared coordinate is the whole result.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
import mujoco  # noqa: E402
from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle  # noqa: E402
from rollout_b1_mujoco import MODEL  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, body_velocity, load, yaw_rate  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def channels_of(pos, quat, dt, embodiment):
    """Forward, lateral and yaw, dimensionless -- the three shared channels, in order."""
    v = body_velocity(pos, quat, dt, embodiment)
    w = yaw_rate(quat, dt, embodiment, float(np.median(pos[:, 2])))
    return np.concatenate([v, np.asarray(w).reshape(len(v), 1)], axis=1)


def sample_actions(rng, mean, std, k, h, scale, smooth):
    """A bank of action sequences no recording contains.

    **Smoothed, not white.** Independent noise per timestep is a joint command that reverses every
    20 ms; nothing walks under it and both conditions would be choosing among sequences that all
    fail identically. The noise is a random walk low-passed over `smooth` steps, which is motor
    babbling rather than a seizure, and it is the same bank for every condition.
    """
    raw = rng.normal(size=(k, h + smooth, len(mean)))
    kernel = np.ones(smooth) / smooth
    walk = np.stack([[np.convolve(raw[i, :, j], kernel, mode="valid")
                      for j in range(raw.shape[2])] for i in range(k)]).transpose(0, 2, 1)
    return mean[None, None, :] + scale * std[None, None, :] * walk[:, :h, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="the body-head checkpoint (F136's refit)")
    ap.add_argument("--projector", required=True, help="the stage-3 checkpoint carrying the projector")
    ap.add_argument("--data", default="data/beh12_b1_flat")
    ap.add_argument("--goal_dir", default="data/beh12_c08f09t09_flat")
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--seed_clip", default="b1_ep3.npz",
                    help="supplies the physical state the robot starts from, and nothing else")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0, help="noise in units of the joint's own sd")
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--floor_scale", type=float, default=3.0)
    ap.add_argument("--cam_fov", type=float, default=24.0)
    ap.add_argument("--out", default="results/wm/closed_loop/plan_without_library")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()
    std_s = np.asarray(ck["body_stats"][1]).ravel()

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"b1": 12}).to(device).eval(); md.load_state_dict(ck["md"], strict=False)
    saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
    proj.load_state_dict(saved["projector"])
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    # --- the goals: one per behaviour condition of the insect, read from its frames --------------
    goals, gcache_path = {}, os.path.join(ROOT, "results/wm/cache/bodycal_hexapod.pt")
    gcache = torch.load(gcache_path, map_location="cpu") if os.path.exists(gcache_path) else {}
    seen = set()
    with torch.no_grad():
        for path in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
            with np.load(path, allow_pickle=True) as raw:
                cond = str(raw["condition"])
            if cond in seen:
                continue
            seen.add(cond)
            if path not in gcache:
                clip = load(path, REGISTRY[args.goal_embodiment])
                gcache[path] = encode_clip(encoder, clip["frames"], 2).cpu().half()
            e = gcache[path].float().to(device)
            t0 = 5
            goals[cond] = md.body(None, itm(e[t0:t0 + 1],
                                            e[t0 + args.horizon:t0 + args.horizon + 1])).reshape(-1)
    torch.save(gcache, gcache_path)
    print(f"{len(goals)} goals read from {args.goal_dir} frames, horizon {args.horizon}")

    # --- the sample bank, identical for every condition and every goal ---------------------------
    acts = np.concatenate([load(p, REGISTRY["b1"])["actions"]
                           for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))])
    rng = np.random.default_rng(args.seed)
    bank = sample_actions(rng, acts.mean(0), acts.std(0), args.samples, args.horizon,
                          args.scale, args.smooth)
    print(f"{args.samples} sampled action sequences, {args.scale} sd, smoothed over {args.smooth}")

    # --- physics, seeded from one clip's state ---------------------------------------------------
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    with np.load(os.path.join(ROOT, args.data, args.seed_clip), allow_pickle=True) as raw0:
        qpos0 = (float(raw0["base_pos"][0][2]), np.asarray(raw0["base_quat"][0], np.float64),
                 np.asarray(raw0["joint_pos"][0], np.float64))

    def reset():
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, qpos0[0]]
        d.qpos[3:7] = qpos0[1]
        d.qpos[7:19] = qpos0[2]
        d.ctrl[:] = qpos0[2]
        mujoco.mj_forward(m, d)
        for _ in range(args.settle):
            mujoco.mj_step(m, d)

    sub = int(round(0.05 / m.opt.timestep))

    def execute(sequence):
        """Run one sampled sequence and return the body motion it actually produced."""
        reset()
        pos, quat = [d.qpos[0:3].copy()], [d.qpos[3:7].copy()]
        for a in sequence:
            d.ctrl[:] = np.asarray(a, np.float64)
            for _ in range(sub):
                mujoco.mj_step(m, d)
            pos.append(d.qpos[0:3].copy()); quat.append(d.qpos[3:7].copy())
        pos, quat = np.array(pos), np.array(quat)
        return channels_of(pos, quat, 0.05, "b1")[:, channels].mean(0), float(pos[-1][2])

    # --- the frame the planner stands in ---------------------------------------------------------
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
    if args.floor_scale > 0:
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                  if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            sim.scaleObjects(floors, float(args.floor_scale), False)
    if args.cam_fov > 0:
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                float(np.deg2rad(args.cam_fov)))
    reset()
    p0 = d.qpos[0:3].copy()
    sim.setObjectPosition(cam, sim.handle_world,
                          [float(p0[0]) + off_xy[0], float(p0[1]) + off_xy[1], cam_z])
    sim.setObjectPosition(root, sim.handle_world, [float(p0[0]), float(p0[1]), float(p0[2])])
    w, x, y, z = d.qpos[3:7]
    sim.setObjectQuaternion(root, sim.handle_world, [float(x), float(y), float(z), float(w)])
    for hh, a in zip(joints, d.qpos[7:19]):
        sim.setJointPosition(hh, float(a))
    frame = capture(sim, cam)
    e_t = encode_clip(encoder, np.asarray(frame)[None], 1).float().to(device)
    del encoder
    torch.cuda.empty_cache()

    # --- score the same bank two ways ------------------------------------------------------------
    a_t = torch.tensor(bank, dtype=torch.float32, device=device)          # K x h x 12
    K, h = a_t.shape[0], a_t.shape[1]
    with torch.no_grad():
        z = proj(a_t.reshape(K * h, -1), "b1").reshape(K, h, -1)
        pred_no_wm = md.body(None, z.reshape(K * h, -1)).reshape(K, h, -1).mean(1)
        e = e_t.expand(K, -1, -1)
        for i in range(h):
            e = ftm(e, z[:, i])
        pred_wm = md.body(None, itm(e_t.expand(K, -1, -1), e))
        if pred_wm.dim() == 1:
            pred_wm = pred_wm.unsqueeze(-1)

    print(f"\n{'goal':<16}{'channel':<10}{'wanted':>9}{'world model':>13}{'no rollout':>12}"
          f"{'random':>9}")
    rows = []
    for cond, g in sorted(goals.items()):
        picks = {"world model": int((pred_wm - g).pow(2).mean(-1).argmin()),
                 "no rollout": int((pred_no_wm - g).pow(2).mean(-1).argmin()),
                 "random": int(rng.integers(K))}
        got = {k: execute(bank[i]) for k, i in picks.items()}
        want = (g.cpu().numpy() * std_s[:len(channels)] + mean_s[:len(channels)])
        for j, ch in enumerate(("forward", "lateral", "yaw")[:len(channels)]):
            print(f"{cond if j == 0 else '':<16}{ch:<10}{want[j]:>9.3f}"
                  + "".join(f"{got[k][0][j]:>13.3f}" if k == "world model"
                            else f"{got[k][0][j]:>12.3f}" if k == "no rollout"
                            else f"{got[k][0][j]:>9.3f}" for k in
                            ("world model", "no rollout", "random")))
        rows.append({"condition": cond, "want": want,
                     **{k: got[k][0] for k in got},
                     **{f"{k}_z": got[k][1] for k in got},
                     **{f"{k}_pick": i for k, i in picks.items()}})

    print(f"\n{'':<16}{'distance to the goal, mean over goals':<40}")
    for k in ("world model", "no rollout", "random"):
        dist = np.mean([np.linalg.norm(r[k] - r["want"]) for r in rows])
        per = np.mean([np.abs(r[k] - r["want"]) for r in rows], axis=0)
        print(f"  {k:<14}{dist:>8.4f}   per channel " +
              "  ".join(f"{c} {v:.3f}" for c, v in
                        zip(("forward", "lateral", "yaw")[:len(channels)], per)))
    print("\n  **Lower is better and the pass bar is `world model` under `no rollout`.** `random`")
    print("  is the floor: a rule that does not beat it is not choosing, and both rules see the")
    print("  identical bank, so anything separating them is the scoring rule alone.")

    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "plan_without_library.npz"),
             **{f"{r['condition']}_{k}": np.asarray(v) for r in rows for k, v in r.items()
                if k != "condition"})
    print(f"\n-> {args.out}/plan_without_library.npz")


if __name__ == "__main__":
    main()
