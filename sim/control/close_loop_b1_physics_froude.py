"""Physics closed loop on the B1: the planner chooses a behaviour, the B1's own policy executes it.

MuJoCo holds the physics and the B1's walking policy (`b1_policy_runtime.B1Walker`, the controller
that produced every B1 clip); CoppeliaSim renders the 90-deg egocentric view from MuJoCo's state
(`counterfactual_truth_check.build_scene`, checked against the training clips). Each decision
(every --replan_every frames at 20 Hz) the planner picks a library candidate; the body then runs
that candidate's recorded **command** (vx, vy, yaw_cmd at frame t) through the policy, from its
current state, for 2.5 policy steps per frame. What the planner scores (the candidate's actions,
projected) and what is executed (its command, by the policy) are the same behaviour at two levels.

The outcome depends on the state (momentum, gait phase, heading): the achieved Froude is measured
from the simulated body, not read from the candidate's recording, so a state-blind selector can
pick the right behaviour and still miss the goal. The body can fall; a fall ends the episode.

    .venv/bin/python3 sim/control/close_loop_b1_physics_froude.py --mechanism rollout --window 21 \\
        --ckpt wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt \\
        --goal data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/hexapod_ep302.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "sim/control", "sim/scene", "sim/render", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from b1_policy_runtime import B1Walker  # noqa: E402
from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
from counterfactual_truth_check import build_scene  # noqa: E402
from ego_camera import check_ego_view  # noqa: E402
from final_2x2x2_test import build_planner  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import RolloutFroudePlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=("direct", "rollout", "random"), default="rollout",
                    help="random: a uniformly random library candidate at every decision (seeded by "
                         "the goal), the physics reference a selector has to beat")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--replan_every", type=int, default=2)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--policy_warmup", type=int, default=45, help="policy steps walking the first "
                    "candidate's command before recording, as the collector crops the spawn transient")
    ap.add_argument("--policy", choices=("gait3", "sym"), default="gait3",
                    help="the B1 body's walking policy (the controller executing every candidate's recorded "
                         "command). gait3 = the walker's long-standing default, now explicit and saved in the "
                         "output; the candidates' recorded yaw command is replayed, so no heading gains apply")
    ap.add_argument("--steps", type=int, default=65)
    ap.add_argument("--fall_ratio", type=float, default=0.6)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--out", default="results/wm/closed_loop/b1_physics_live")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    ckpt = os.path.join(ROOT, args.ckpt)
    cdir = os.path.join(ROOT, args.candidates_dir)
    planner = build_planner(ckpt, cdir, "b1", args.horizon, free_offset=False, device=dev, per_condition=999)
    planner.window = args.window
    rp, encoder, off = None, None, None
    if args.mechanism == "rollout":
        rp = RolloutFroudePlanner.from_checkpoint(ckpt, cdir, embodiment="b1", horizon=args.horizon,
                                                  per_condition=999, device=dev)
        rp.window = args.window
        from vjepa2_encoder import VJEPA2FrameEncoder
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        off = offset_for(torch.load(ckpt, map_location="cpu", weights_only=False), "b1")
    cands = planner.candidates
    cmds = [np.asarray(np.load(c["path"], allow_pickle=True)["command"], np.float32) for c in cands]
    goal_path = os.path.join(ROOT, args.goal)
    goal_bm = np.asarray(load(goal_path, REGISTRY[args.goal_embodiment])["body_motion"])[:, :3]
    steps = min(args.steps, len(goal_bm), min(len(c) for c in cmds))

    walker = B1Walker(policy=args.policy)
    print(f"  B1 body settings: {walker.settings()}")
    for _ in range(args.policy_warmup):
        walker.policy_step(cmds[0][0])
    settled_z = float(walker.d.qpos[2])
    # re-centre the walked-in warmup at the origin so the room (built around 0,0) surrounds it
    walker.d.qpos[0:2] -= walker.d.qpos[0:2].copy()

    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    pose = build_scene(sim, args.scene)
    ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]

    def render():
        p, q, j, _ = walker.state()
        return pose(p, q, j)

    frame = render()
    print(f"  ego view check: row-profile corr {check_ego_view(frame, ref, what='B1 physics frame 0'):.3f}")
    rec = {k: [] for k in ("frames", "base_pos", "base_quat", "joint_pos", "joint_vel", "action",
                           "command", "foot_contact")}
    chosen, held, fell_at, acc = [], 0, None, 0.0
    rnd = np.random.default_rng(__import__("zlib").crc32(os.path.basename(goal_path).encode()))
    for t in range(steps):
        if t % args.replan_every == 0:
            g = planner.standardize(goal_bm[min(t, len(goal_bm) - 1)])
            with torch.no_grad():
                if args.mechanism == "random":
                    held = int(rnd.integers(len(cands)))
                elif rp is None:
                    _, held, _, _ = planner.act(g, t)
                else:
                    e_t = encode_clip(encoder, np.asarray(frame)[None], 1).float()
                    if off is not None:
                        e_t = e_t - off.to(e_t.device)
                    _, held, _ = rp.act(e_t, torch.as_tensor(g, dtype=torch.float32), t)
        cmd = cmds[held][min(t, len(cmds[held]) - 1)]
        acc += 2.5                                   # 20 Hz frames, 50 Hz policy
        n = int(acc)
        acc -= n
        for _ in range(n):
            action = walker.policy_step(cmd)
        frame = render()
        p, q, j, jv = walker.state()
        for k, v in (("frames", frame), ("base_pos", p), ("base_quat", q), ("joint_pos", j),
                     ("joint_vel", jv), ("action", action), ("command", cmd), ("foot_contact", walker.foot)):
            rec[k].append(np.asarray(v))
        chosen.append(cands[held]["condition"])
        if float(p[2]) < args.fall_ratio * settled_z or walker.upright() < 0.5:
            fell_at = t
            print(f"  FELL at step {t}")
            break

    out_dir = os.path.join(ROOT, args.out, f"{args.mechanism}_w{args.window}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"b1_{os.path.splitext(os.path.basename(goal_path))[0]}.npz")
    arrays = {k: np.asarray(v, np.uint8 if k == "frames" else np.float32) for k, v in rec.items()}
    np.savez_compressed(out, **arrays, dt=np.float32(0.05), fps=np.float32(20), embodiment="b1",
                        condition="closed_loop", behaviour="closed_loop", level=-1, expert_episode=-1,
                        chosen=np.asarray(chosen), goal=os.path.basename(goal_path),
                        mechanism=args.mechanism, window=args.window,
                        fell_at=-1 if fell_at is None else fell_at, settled_z=settled_z,
                        **{f"body_{k}": np.array(v) for k, v in walker.settings().items()})
    achieved = np.asarray(load(out, REGISTRY["b1"])["body_motion"])[:, :3]
    n = min(len(achieved), len(goal_bm))
    dec = np.arange(0, n, args.replan_every)
    err = np.linalg.norm(achieved[dec] - goal_bm[dec], axis=1)
    with np.load(out, allow_pickle=True) as d:
        saved = {k: d[k] for k in d.files}
    np.savez_compressed(out, **saved, achieved_froude=achieved[:n].astype(np.float32),
                        goal_froude_t=goal_bm[:n].astype(np.float32))
    print(f"B1 physics closed loop: mean L2 error {err.mean():.4f} at {len(dec)} decision steps"
          f"{'' if fell_at is None else f' (fell at {fell_at})'} -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
