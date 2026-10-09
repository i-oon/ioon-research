"""Physics closed loop on a hexapod body (e.g. the held-out c08f09t09), Froude-goal selection.

The hexapod twin of `close_loop_direct_froude.py`, with one difference that makes it stricter: the
insect is **simulated**, not posed. `collect_ik.drive_and_record` steps CoppeliaSim with joint
targets and calls `policy(frame, t)` for each command, so the planner picks a candidate from the
body's own library, the chosen candidate's recorded joint command at t is sent to the joints, and
the achieved Froude is measured from how the body actually moved (it can slip, drift or fall).

  every --replan_every steps   direct: planner.act(goal_t, t)
                               rollout: planner.act(encode(ego frame), goal_t, t)
  every step                   command = chosen candidate's actions[t]
  grading                      L2(achieved body_motion[t], goal body_motion[t]) at decision steps

Egocentric camera exactly as the collector's (`--view egocentric`, 90 deg set after
`startSimulation`, room around the spawn, `ego_seed`); the first frame must pass
`ego_camera.check_ego_view` against the candidate library's frames (F256).

    .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism rollout --window 21 \\
        --ckpt wm/runs/beh24_stride5_cleansplit/c08_zeroshot/ckpt_lib_zeroshot.pt \\
        --goal data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/hexapod_ep302.npz \\
        --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat --morph c08f09t09=medauroidea_c08f09t09.ttt
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "sim/collect", "sim/scene", "sim/render", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
from collect_ik import drive_and_record  # noqa: E402
from ego_camera import EGO_FOV_DEG, check_ego_view  # noqa: E402
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
    ap.add_argument("--goal", required=True, help="goal clip (hexapod), read as recorded body_motion per step")
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--morph", required=True, help="NAME=SCENE of the controlled body")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--replan_every", type=int, default=2)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--phase_match", action="store_true",
                    help="execution only: on a switch, continue the new candidate from the frame whose "
                         "joint command is nearest the last one sent (its gait phase), not from index t. "
                         "Separates 'chose the right behaviour' from 'the switch broke the gait'.")
    ap.add_argument("--replay_clip", default="", help="verification: send this clip's recorded joint commands "
                    "open loop (no planner) through the same measurement path; the goal still sets the grading")
    ap.add_argument("--ego_seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop/hexapod_live")
    ap.add_argument("--rr_room", default="", help="an rr hexapod clip of the controlled body "
                    "(data/counterfactual_walks/rr_c08_clips_*): build its random room live (render_shift_heldout rs path: "
                    "room seed / size / offset, floor tiles, wall skirts, visible floor at the source level, far clip; "
                    "same ego camera mount + FOV 90), body turned to the clip's start heading, head spawned at the clip "
                    "start (0, 0); frame 0 checked against the clip's frame 0. '' = the old 8 m room (unchanged)")
    ap.add_argument("--orig_room", default="", help="an ORIGINAL-room clip of the controlled body "
                    "(data/counterfactual_walks/{c10,c08}_clips_*): the collector's own room (8 m, sized to the body) with "
                    "the clip's room seed, body turned to the clip's start heading; frame 0 checked against the clip's")
    ap.add_argument("--cpg_clock", action="store_true",
                    help="F323 fix: drive the body from the chosen candidate's CPG recipe (plan_*) on ONE continuous gait "
                         "clock, cross-fading 4 frames at a switch exactly as the branches were collected "
                         "(sim/control/cpg_clock.py), instead of playing the candidate's recorded actions[t]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    emb, dev = "hexapod", args.device
    ckpt = os.path.join(ROOT, args.ckpt)
    cdir = os.path.join(ROOT, args.candidates_dir)
    planner = build_planner(ckpt, cdir, emb, args.horizon, free_offset=False, device=dev, per_condition=999)
    planner.window = args.window
    rp, encoder, off = None, None, None
    if args.mechanism == "rollout":
        rp = RolloutFroudePlanner.from_checkpoint(ckpt, cdir, embodiment=emb, horizon=args.horizon,
                                                  per_condition=999, device=dev)
        rp.window = args.window
        from vjepa2_encoder import VJEPA2FrameEncoder
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        off = offset_for(torch.load(ckpt, map_location="cpu", weights_only=False), emb)
    cands = planner.candidates
    goal_path = os.path.join(ROOT, args.goal)
    goal_bm = np.asarray(load(goal_path, REGISTRY[emb])["body_motion"])[:, :3]
    ref_frames = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    name, scene = args.morph.split("=", 1)
    rr_kw, rc = {}, None
    if args.rr_room:
        sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
        import render_shift_heldout as RS
        from collect_ik import ENV, settle, ROBOT_ROOT
        from wm.data.embodiment import heading
        rc = RS.load(os.path.join(ROOT, args.rr_room))
        seed, S, o = int(rc["room_seed"]), float(rc["rr_room_size"]), np.asarray(rc["rr_room_offset"], float)
        # probe the unscaled scene: source visible-floor top (8 m rule) and the authored body heading
        settle(sim); sim.loadScene(os.path.join(ENV, scene)); settle(sim)
        top_src = RS.src_floor_top(sim, 8.0)
        q_auth = np.asarray(sim.getObjectQuaternion(sim.getObject(ROBOT_ROOT), sim.handle_world), float)
        dpsi = float(heading(np.asarray(rc["body_quat"][:1], float), "hexapod")[0] - heading(q_auth[None], "hexapod")[0])
        dpsi = float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))

        def build_room(sim_, cam_, R_, here):
            RS.rs_walls(sim_, R_, seed, (float(here[0] - o[0]), float(here[1] - o[1])), top_src)
            # hold_floor moves /Floor, the respondable floor; in physics only the visual /Floor/box is moved
            bx = sim_.getObject("/Floor/box")
            q = sim_.getObjectPosition(bx, sim_.handle_world)
            sim_.setObjectPosition(bx, sim_.handle_world, [q[0], q[1], q[2] + (top_src - RS.visible_floor_top(sim_))])

        def after_start(sim_, cam_):
            RS.rs_far(sim_, cam_, S)
        rr_kw = dict(ego_room=RS.rs_room(S), build_room=build_room, after_start=after_start,
                     yaw=float(np.rad2deg(dpsi)))
        args.ego_seed = seed
        ref_frames = [rc["frames"][0]]
        print(f"  rr room: seed {seed} size {S:.2f} m offset {o.round(2).tolist()} (from {args.rr_room}); "
              f"start heading turned by {np.rad2deg(dpsi):+.1f} deg", flush=True)
    if args.orig_room:
        from collect_ik import ENV, settle, ROBOT_ROOT
        from wm.data.embodiment import heading
        sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
        import render_shift_heldout as RS
        rc = RS.load(os.path.join(ROOT, args.orig_room))
        settle(sim); sim.loadScene(os.path.join(ENV, scene)); settle(sim)
        q_auth = np.asarray(sim.getObjectQuaternion(sim.getObject(ROBOT_ROOT), sim.handle_world), float)
        dpsi = float(heading(np.asarray(rc["body_quat"][:1], float), "hexapod")[0] - heading(q_auth[None], "hexapod")[0])
        dpsi = float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
        rr_kw = dict(yaw=float(np.rad2deg(dpsi)))
        args.ego_seed = int(rc["room_seed"])
        ref_frames = [rc["frames"][0]]
        print(f"  original room: seed {args.ego_seed} (from {args.orig_room}); start heading turned by "
              f"{np.rad2deg(dpsi):+.1f} deg", flush=True)
    clock = {"k": None}
    if args.cpg_clock:
        from cpg_clock import Clock, centre_of, recipe
        centre = centre_of(name)
        cand_np = [np.load(c["path"], allow_pickle=True) for c in cands]
    steps = min(len(goal_bm), min(len(c["actions"]) for c in cands))
    seed_cmds = np.asarray(cands[0]["actions"], np.float32)[:steps]      # warmup pose + clip length

    held = {"i": 0, "off": 0, "last": None}
    rnd = np.random.default_rng(__import__("zlib").crc32(os.path.basename(goal_path).encode()))
    chosen, checked = [], {"done": False}

    def policy(frame, t):
        if not checked["done"]:
            if rc is not None:
                from PIL import Image
                os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
                Image.fromarray(np.hstack([frame, ref_frames[0]])).save(os.path.join(
                    ROOT, args.out, f"framecheck_{args.mechanism}_{os.path.splitext(os.path.basename(args.goal))[0]}.png"))
                print(f"  loop frame 0 vs stored rr clip frame 0 (same room, start pose): pixel MAE "
                      f"{np.abs(frame.astype(int) - ref_frames[0].astype(int)).mean():.2f} / 255", flush=True)
            # rr: reference = every frame of the stored clip in this room (one mid-gait frame carries the head's
            # roll / pitch sway; the training-view check is about the room / mount / FOV, not the gait phase)
            r = check_ego_view(frame, list(rc["frames"]) if rc is not None else ref_frames,
                               what="hexapod closed-loop ego frame 0")
            print(f"  ego view check: row-profile corr {r:.3f}", flush=True)
            checked["done"] = True
        if t % args.replan_every == 0:
            g = torch.as_tensor(planner.standardize(goal_bm[min(t, len(goal_bm) - 1)]), dtype=torch.float32)
            with torch.no_grad():
                if args.mechanism == "random":
                    i = int(rnd.integers(len(cands)))
                elif rp is None:
                    _, i, _, _ = planner.act(g.numpy(), t)
                else:
                    e_t = encode_clip(encoder, np.asarray(frame)[None], 1).float()
                    if off is not None:
                        e_t = e_t - off.to(e_t.device)
                    if args.cpg_clock:
                        # F330: imagine each candidate from the frame of its clip whose gait phase matches the robot's
                        # current one (the clock's phase), not from index t (another phase: a pairing training never shows)
                        k_ = clock["k"]
                        ph = (k_.c % 1.0) if k_ is not None else float(cand_np[held["i"]]["cpg_phase"][0])
                        st = []
                        for cn in cand_np:
                            cp = np.asarray(cn["cpg_phase"], float)[:len(cn["actions"]) - 26]
                            d_ = np.abs(cp - ph); d_ = np.minimum(d_, 1 - d_)
                            st.append(int(np.argmin(d_ + 1e-3 * np.abs(np.arange(len(cp)) - t))))
                        rp.cand_start = st
                    _, i, _ = rp.act(e_t, g, t)
            if i != held["i"] and args.phase_match and held["last"] is not None:
                acts = np.asarray(cands[i]["actions"])
                tau = int(np.argmin(np.linalg.norm(acts[:-1] - held["last"], axis=1)))
                held["off"] = tau - t                  # continue the new clip from its matching phase
            elif i != held["i"]:
                held["off"] = 0
            held["i"] = i
        if args.cpg_clock and not args.replay_clip:
            if clock["k"] is None:     # start on the first choice's own phase and recipe (= its actions[0])
                clock["k"] = Clock(cand_np[held["i"]]["cpg_cycles_total"][0], recipe(cand_np[held["i"]]), centre)
            else:
                clock["k"].choose(recipe(cand_np[held["i"]]))
            chosen.append(cands[held["i"]]["condition"])
            cmd = clock["k"].step()
            held["last"] = cmd
            return cmd
        if args.replay_clip:
            a = np.load(os.path.join(ROOT, args.replay_clip), allow_pickle=True)["actions"]
            chosen.append("replay")
            return a[min(t, len(a) - 1)]
        chosen.append(cands[held["i"]]["condition"])
        a = cands[held["i"]]["actions"]
        cmd = a[int(np.clip(t + held["off"], 0, len(a) - 1))]
        held["last"] = np.asarray(cmd)
        return cmd

    st = {}
    frames, actions, forces, heads, oris = drive_and_record(
        sim, scene, seed_cmds, 0.0, args.warmup, spawn=(0.0, 0.0), ego=True, ego_seed=args.ego_seed,
        cam_fov=EGO_FOV_DEG, policy=policy, state_out=st, **rr_kw)
    # labels at the CoM like every goal / candidate clip (wm.data.com, F301); without com_pos the loader labels
    # at the head, 0.246 m ahead of the CoM, and a turn's yaw reads as lateral (yaw x 0.246 / h)
    from wm.data.com import hex_com
    com_pos = np.asarray(hex_com(st["state_link_names"], np.asarray(st["state_link_pose"], float),
                                 morph=name), np.float64)
    assert len(com_pos) == len(heads), (len(com_pos), len(heads))
    if rc is not None:
        # the stored clip's frame 0 posed (links mode, as render_shift_heldout.hex_render) in the live-built scene
        from render_hex_replay import capture
        cam = sim.getObject("/vjepa_cam")
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(EGO_FOV_DEG)))
        if not args.orig_room:
            RS.rs_far(sim, cam, float(rc["rr_room_size"]))
        for h, p in zip([sim.getObject(str(n)) for n in rc["state_link_names"]], np.asarray(rc["state_link_pose"][0], float)):
            sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        f0 = capture(sim, cam)
        print(f"  stored clip frame 0 re-rendered through the live scene: pixel MAE "
              f"{np.abs(f0.astype(int) - rc['frames'][0].astype(int)).mean():.3f} / 255", flush=True)

    out_dir = os.path.join(ROOT, args.out, ("replay" if args.replay_clip else f"{args.mechanism}_w{args.window}")
                           + ("_phase" if args.phase_match else "") + ("_cpg" if args.cpg_clock else ""))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(goal_path))[0]
    out = os.path.join(out_dir, f"{name}_{stem}.npz")
    np.savez_compressed(out, frames=frames, actions=actions, forces=forces, head=heads, body_quat=oris, com_pos=com_pos,
                        state_link_names=st["state_link_names"], state_link_pose=np.asarray(st["state_link_pose"]),
                        embodiment=emb, morph=name, chosen=np.asarray(chosen), goal=os.path.basename(goal_path),
                        mechanism=args.mechanism, window=args.window, dt=np.float32(0.05),
                        expert_episode=-1, condition="closed_loop", behaviour="closed_loop", level=-1)
    achieved = np.asarray(load(out, REGISTRY[emb])["body_motion"])[:, :3]
    n = min(len(achieved), len(goal_bm))
    dec = np.arange(0, n, args.replan_every)
    err = np.linalg.norm(achieved[dec] - goal_bm[dec], axis=1)
    with np.load(out, allow_pickle=True) as d:
        saved = {k: d[k] for k in d.files}
    np.savez_compressed(out[:-4] + ".tmp.npz", **saved, achieved_froude=achieved[:n].astype(np.float32),
                        goal_froude_t=goal_bm[:n].astype(np.float32))
    os.replace(out[:-4] + ".tmp.npz", out)
    if clock["k"] is not None:
        print(f"  cpg clock: {clock['k'].switches} recipe switches (4-frame cross-fade each)")
    print(f"physics closed loop: mean L2 error {err.mean():.4f} at {len(dec)} decision steps -> "
          f"{os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
