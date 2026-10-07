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
    ap.add_argument("--rr_room", default="", help="an rr B1 clip (data/counterfactual_walks/rr_b1_clips_*): build "
                    "its random room live (render_shift_heldout.b1_setup, the rr data's room / camera mount / FOV 90), "
                    "loop start at the clip start's offset from the room centre and turned to its start heading; "
                    "'' = the old fixed room (counterfactual_truth_check.build_scene)")
    ap.add_argument("--thirdperson", action="store_true", help="after the run, re-render the saved MuJoCo states "
                    "with a chase camera in the same room (render_allo_selection.chase_poses) -> "
                    "<out>/thirdperson[_<mech>]_b1_<goal>.npz (with --rr_room)")
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
    if args.rr_room:
        sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
        import render_shift_heldout as RS
        from render_b1_replay import pose_and_capture
        from wm.data.embodiment import heading
        rc = RS.load(os.path.join(ROOT, args.rr_room))
        # turn the walked-in state to the clip's start heading (policy observations are yaw-invariant)
        p0, q0, _, _ = walker.state()
        dpsi = float(heading(np.asarray(rc["base_quat"][:1], float), "b1")[0] - heading(q0[None], "b1")[0])
        cz, sz = np.cos(dpsi / 2), np.sin(dpsi / 2)
        w, x, y, z = q0
        walker.d.qpos[3:7] = [cz * w - sz * z, cz * x - sz * y, cz * y + sz * x, cz * z + sz * w]
        Rz = np.array([[np.cos(dpsi), -np.sin(dpsi), 0], [np.sin(dpsi), np.cos(dpsi), 0], [0, 0, 1]])
        walker.d.qvel[0:3] = Rz @ walker.d.qvel[0:3]
        __import__("mujoco").mj_forward(walker.m, walker.d)
        off_rr = np.asarray(rc["rr_room_offset"], float)
        root_h, joints_h, cam_h, _ = RS.b1_setup(sim, int(rc["room_seed"]), -off_rr,
                                                 RS.rs_room(float(rc["rr_room_size"])))
        s0 = np.asarray(rc["base_pos"][0], float) * [1, 1, 0]   # the clip, shifted so its start is at (0, 0)
        f0, _ = pose_and_capture(sim, root_h, joints_h, cam_h, rc["base_pos"][0] - s0, rc["base_quat"][0],
                                 rc["joint_pos"][0])
        mae = float(np.abs(f0.astype(int) - rc["frames"][0].astype(int)).mean())
        print(f"  rr room: seed {int(rc['room_seed'])} size {float(rc['rr_room_size']):.2f} m offset "
              f"{off_rr.round(2).tolist()} (from {args.rr_room}); start heading turned by {np.rad2deg(dpsi):+.1f} deg; "
              f"stored clip frame 0 re-rendered through the live path: pixel MAE {mae:.3f} / 255")

        def pose(pos, quat, jangles):
            return pose_and_capture(sim, root_h, joints_h, cam_h, pos, quat, jangles)[0]
        ref = [rc["frames"][0]]
    else:
        pose = build_scene(sim, args.scene)
        ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]

    def render():
        p, q, j, _ = walker.state()
        return pose(p, q, j)

    frame = render()
    print(f"  ego view check: row-profile corr {check_ego_view(frame, ref, what='B1 physics frame 0'):.3f}")
    if args.rr_room:
        from PIL import Image
        os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
        Image.fromarray(np.hstack([frame, ref[0]])).save(os.path.join(
            ROOT, args.out, f"framecheck_{args.mechanism}_{os.path.splitext(os.path.basename(args.goal))[0]}.png"))
        print(f"  loop frame 0 vs stored rr clip frame 0 (same room, start pose): pixel MAE "
              f"{np.abs(frame.astype(int) - ref[0].astype(int)).mean():.2f} / 255")
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
    np.savez_compressed(out[:-4] + ".tmp.npz", **saved, achieved_froude=achieved[:n].astype(np.float32),
                        goal_froude_t=goal_bm[:n].astype(np.float32))
    os.replace(out[:-4] + ".tmp.npz", out)
    if args.thirdperson and args.rr_room:
        sys.path.insert(0, os.path.join(ROOT, "scripts/figures"))
        from render_allo_selection import chase_poses, BODY_LEN, FOV_DEG
        from render_b1_replay import capture
        cp = np.asarray([pose_and_capture(sim, root_h, joints_h, cam_h, p_, q_, j_, render=False)[1]
                         for p_, q_, j_ in zip(arrays["base_pos"], arrays["base_quat"], arrays["joint_pos"])])
        tc = sim.copyPasteObjects([cam_h], 0)[0]
        sim.setObjectParent(tc, -1, True)
        sim.setObjectFloatParam(tc, sim.visionfloatparam_perspective_angle, float(np.deg2rad(FOV_DEG)))
        sim.setObjectFloatParam(tc, sim.visionfloatparam_near_clipping, 0.02)
        tf = []
        for p_, q_, j_, M in zip(arrays["base_pos"], arrays["base_quat"], arrays["joint_pos"],
                                 chase_poses(arrays["base_pos"].astype(float), cp, BODY_LEN["b1"])):
            pose_and_capture(sim, root_h, joints_h, cam_h, p_, q_, j_, render=False)
            sim.setObjectMatrix(tc, sim.handle_world, [float(v) for v in M])
            tf.append(capture(sim, tc))
        sim.removeObjects([tc])
        tag = "thirdperson" if args.mechanism == "direct" else f"thirdperson_{args.mechanism}"
        tp = os.path.join(ROOT, args.out, f"{tag}_{os.path.basename(out)}")
        np.savez_compressed(tp[:-4] + ".tmp.npz", frames=np.asarray(tf, np.uint8), source=os.path.relpath(out, ROOT),
                            fov=FOV_DEG, body_len=BODY_LEN["b1"])
        os.replace(tp[:-4] + ".tmp.npz", tp)
        print(f"  third-person -> {os.path.relpath(tp, ROOT)}")
    print(f"B1 physics closed loop: mean L2 error {err.mean():.4f} at {len(dec)} decision steps"
          f"{'' if fell_at is None else f' (fell at {fell_at})'} -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
