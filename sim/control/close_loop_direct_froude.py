"""Kinematic closed loop, two mechanisms compared side by side: direct-Froude vs. rollout-Froude.

**Why this file exists.** `close_loop_kinematic.py`'s `LatentPlanner` rolls the FTM and scores raw
embedding distance to a goal frame -- the mechanism F125/F127 found failing under every estimator
tried. `score_by_body_motion.py` measures two better-founded mechanisms offline against a fixed
clip library: mode A (`body_head(proj(a))` vs. a recorded goal number, no rollout) and mode C
(the goal read from the SOURCE robot's frames via the ITM, candidates scored by rolling the FTM
forward from the CURRENT observation and reading the transition, also via the ITM -- "frames and
rollout only... the condition the project's claim actually needs"). F202 measured mode A clearing
chance (38-46% vs 28%) and mode C failing at every horizon (22-27%) on the correctly-adapted B1
checkpoint. This file runs BOTH live, step-by-step, rendered -- `--mechanism direct` is mode A's
planner (`DirectFroudePlanner`), `--mechanism rollout` is mode C's (`RolloutFroudePlanner`), same
control loop, same candidates, same goal clip, so the two are directly comparable.

**Still kinematic, not physics** -- the body is posed frame by frame from whichever candidate's
recorded motion won, so it cannot fall. This tests whether selection stays right when the frames it
next sees are produced by what it chose, not whether the sequence is physically executable.

  .venv/bin/python3 sim/control/close_loop_direct_froude.py --mechanism direct \\
      --ckpt wm/runs/b1_adapt/body_head_b1.pt \\
      --demo data/egocentric/beh12_b1_ego_flat/b1_ep2.npz \\
      --goal data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep0.npz --goal_embodiment hexapod \\
      --candidates_dir data/egocentric/beh12_b1_ego_flat

  .venv/bin/python3 sim/control/close_loop_direct_froude.py --mechanism rollout \\
      --ckpt wm/runs/b1_adapt/body_head_b1.pt \\
      --demo data/egocentric/beh12_b1_ego_flat/b1_ep2.npz \\
      --goal data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep0.npz --goal_embodiment hexapod \\
      --candidates_dir data/egocentric/beh12_b1_ego_flat
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle  # noqa: E402

from wm.data.embodiment import REGISTRY, heading, load  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import B1_DT, GECKO_DT, HEXAPOD_DT  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from wm.policy.planner import (DirectFroudePlanner, RolloutFroudePlanner,  # noqa: E402
                               condition_of)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def load_motion(path):
    """Per-step body-frame deltas and joint angles for one clip -- identical to
    `close_loop_kinematic.py`'s version, duplicated rather than imported because that file also
    imports `LatentPlanner` at module scope and this one deliberately never touches it."""
    with np.load(path, allow_pickle=True) as d:
        pos = d["base_pos"].astype("float64")
        quat = d["base_quat"].astype("float64")     # (w, x, y, z)
        jpos = d["joint_pos"].astype("float64")
    n = len(pos) - 1
    dpos = np.einsum("nij,nj->ni", np.array([quat_wxyz_to_R(quat[t]).T for t in range(n)]),
                     pos[1:] - pos[:-1])
    dquat = np.array([quat_mul(quat_conj(quat[t]), quat[t + 1]) for t in range(n)])
    return {"dpos": dpos, "dquat": dquat, "jpos": jpos, "height": float(np.median(pos[:, 2]))}


def goal_froude(path, embodiment, channels):
    """Mean (forward, lateral, yaw) over the whole clip -- the recorded-number goal (mode A/B)."""
    clip = load(path, REGISTRY[embodiment])
    return np.asarray(clip["body_motion"])[:, channels].mean(0)


def vision_goal(itm, md, encoder, offset, goal_path, horizon):
    """The vision-only goal (mode C/D): averaged over every valid (t, t+horizon) frame-pair in
    the clip, read via ITM + body_head, no recorded number involved anywhere. Independent of
    which planner scores candidates -- this is the goal-reading half only."""
    with np.load(goal_path, allow_pickle=True) as gd:
        gframes = gd["frames"]
    n_pairs = max(1, len(gframes) - horizon)
    idx0 = np.arange(n_pairs)
    idx1 = idx0 + horizon
    ge = encode_clip(encoder, gframes[np.concatenate([idx0, idx1])], 2).float()
    if offset is not None:
        ge = ge - offset.to(ge.device)
    g0, g1 = ge[:n_pairs], ge[n_pairs:]
    dev = next(itm.parameters()).device
    with torch.no_grad():
        z = itm(g0.to(dev), g1.to(dev))
        goals_per_pair = md.body(None, z)
    goal = goals_per_pair.mean(0)
    return goal, goals_per_pair[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=("direct", "rollout"), default="direct",
                    help="**candidate-scoring mechanism ONLY** -- direct = body_head(proj(a)), no "
                         "rollout, no FTM. rollout = roll the FTM from the current observation, "
                         "read the transition via the ITM. Independent of --goal_source: crossing "
                         "them gives the original A/B/C/D framework (A=direct+physics, "
                         "B=rollout+physics, C=rollout+vision, D=direct+vision). Conflating goal "
                         "source with candidate mechanism (always physics with direct, always "
                         "vision with rollout) confounds two variables in one comparison -- this "
                         "flag and --goal_source are deliberately orthogonal so that mistake isn't "
                         "possible here.")
    ap.add_argument("--goal_source", choices=("physics", "vision"), default=None,
                    help="how the goal is read. physics = mode A/B, a recorded number (mean body "
                         "motion telemetry, no vision at all). vision = mode C/D, read from the "
                         "goal clip's own FRAMES via ITM+body_head, averaged over every valid "
                         "frame-pair in the clip (never a single noisy pair). Defaults to "
                         "matching --mechanism (direct->physics, rollout->vision) for "
                         "backward compatibility, but pass this explicitly to decouple them.")
    ap.add_argument("--ckpt", required=True, help="the fully-merged checkpoint (itm/ftm/md/"
                    "projector/body_stats all needed for --mechanism rollout; direct needs only "
                    "md.body_head, the projector and body_stats)")
    ap.add_argument("--projector", default="", help="defaults to --ckpt, which carries one for "
                    "the merged b1_adapt/teacher_*.pt or body_head_*.pt checkpoints")
    ap.add_argument("--demo", required=True, help="a B1 clip supplying the spawn pose and the "
                    "warm-start motion -- candidates must be B1 (executable), only the goal crosses")
    ap.add_argument("--goal", required=True, help="clip supplying the goal, possibly a different "
                    "robot -- this is what makes it cross-embodiment. Read as a recorded Froude "
                    "number for --mechanism direct, or as a pair of frames for --mechanism rollout")
    ap.add_argument("--goal_embodiment", default="", help="defaults to --embodiment")
    ap.add_argument("--goal_frame0", type=int, default=0, help="rollout only: first goal frame index")
    ap.add_argument("--goal_frame1", type=int, default=-1, help="rollout only: second goal frame "
                    "index; -1 means --horizon steps after --goal_frame0")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizon", type=int, default=5,
                    help="the CANDIDATE-SCORING horizon: how far the rollout planner rolls the "
                         "forward model. Nothing to do with how the goal is read -- see "
                         "--goal_horizon, which used to share this flag and should not have.")
    ap.add_argument("--goal_horizon", type=int, default=None,
                    help="frame spacing used to READ the goal from the source clip's video "
                         "(t, t+goal_horizon). Defaults to --horizon for backwards comparability, "
                         "but **1 is the value that matches training**: every stage that fits the "
                         "reading path (pretrain ITM, stage-1 adapt, the projector, the body head) "
                         "is built on adjacent frame pairs, so reading at 5 deploys the head on a "
                         "spacing it never saw. Measured across 48 source clips, reading at 1 "
                         "instead of 5 cuts the median goal-read error 0.063 -> 0.040 and raises "
                         "the share of clips under the candidate-spacing threshold from 15% to 42%.")
    ap.add_argument("--free_offset", action="store_true", help="--mechanism direct only: let "
                    "DirectFroudePlanner pick ANY offset within a candidate, not just the one "
                    "matching the live step -- see wm/policy/planner.py's DirectFroudePlanner "
                    "docstring. F80 measured this ~15pts more accurate and rejected it for the "
                    "discontinuous joint command it can cause; off by default so the original, "
                    "already-validated mechanism is the one you get unless you ask for this.")
    ap.add_argument("--steps", type=int, default=66)
    ap.add_argument("--warm_start", type=int, default=10)
    ap.add_argument("--travel", type=float, default=2.0)
    ap.add_argument("--cam_dx", type=float, default=0.0)
    ap.add_argument("--cam_dy", type=float, default=0.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop/direct_froude")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    goal_embodiment = args.goal_embodiment or args.embodiment
    ckpt_path = os.path.join(ROOT, args.ckpt)
    goal_path = args.goal if os.path.isabs(args.goal) else os.path.join(ROOT, args.goal)
    proj_path = os.path.join(ROOT, args.projector) if args.projector else ""

    goal_source = args.goal_source or ("physics" if args.mechanism == "direct" else "vision")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    obs_offset = offset_for(checkpoint, args.embodiment)
    # Needed whenever candidates are scored by rollout (e_t every step) OR the goal is read from
    # vision -- independent conditions, so this can't live inside either branch alone (mode B
    # needs it for e_t with a physics goal; mode D needs it for the goal with direct candidates).
    encoder = VJEPA2FrameEncoder(dtype=torch.float32) if (
        args.mechanism == "rollout" or goal_source == "vision") else None

    # **Candidate-scoring planner, decoupled from goal source.** Both planner classes' `.act()`
    # take a plain goal vector -- neither cares how it was computed -- so building "the wrong"
    # planner for a given goal source was never actually required; it was just how the two got
    # bundled originally (A always with direct, C always with rollout).
    if args.mechanism == "direct":
        planner = DirectFroudePlanner.from_checkpoint(
            ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment, proj_path,
            horizon=args.horizon, device=str(device), free_offset=args.free_offset)
    else:
        planner = RolloutFroudePlanner.from_checkpoint(
            ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment, proj_path,
            horizon=args.horizon, device=str(device))

    # **The recorded number is read either way, but only USED when goal_source=physics.** Under a
    # vision goal it is the reference the read-from-video estimate is scored against -- without it
    # saved, nothing downstream can say how accurate the vision reading was.
    goal_reference = goal_froude(goal_path, goal_embodiment, planner.channels)
    if goal_source == "physics":
        goal_raw = goal_reference
        goal = torch.as_tensor(planner.standardize(goal_raw), dtype=torch.float32, device=device)
        print(f"goal_source=physics (recorded number): {os.path.basename(goal_path)} "
             f"({goal_embodiment})  raw={np.round(goal_raw, 4)}  "
             f"standardised={np.round(goal.cpu().numpy(), 3)}")
    else:
        # vision goal needs its own itm+body_head -- reuse the planner's if it already has them
        # (mechanism=rollout), otherwise load a throwaway pair just for reading the goal
        goal_offset = offset_for(checkpoint, goal_embodiment)
        if args.mechanism == "rollout":
            itm_for_goal, md_for_goal = planner.itm, planner.md
        else:
            cfg = from_checkpoint(checkpoint["config"])
            itm_for_goal = InverseTransitionModel(cfg).to(device).eval()
            itm_for_goal.load_state_dict(checkpoint["itm"])
            md_for_goal = MotionDecoder(cfg, {args.embodiment: 12}).to(device).eval()
            md_for_goal.load_state_dict(checkpoint["md"], strict=False)
            for p in list(itm_for_goal.parameters()) + list(md_for_goal.body_head.parameters()):
                p.requires_grad_(False)
        gh = args.goal_horizon if args.goal_horizon is not None else args.horizon
        goal, goal_single = vision_goal(itm_for_goal, md_for_goal, encoder, goal_offset,
                                        goal_path, gh)
        print(f"goal_source=vision ({len(np.load(goal_path)['frames']) - gh} "
             f"frame-pairs, goal_horizon {gh}, averaged via ITM): "
             f"{os.path.basename(goal_path)} ({goal_embodiment})  "
             f"body_head units={np.round(goal.cpu().numpy(), 3)}  "
             f"(single-pair estimate was {np.round(goal_single.cpu().numpy(), 3)})")

    motion = [load_motion(c["path"]) for c in planner.candidates]
    demo_path = args.demo if os.path.isabs(args.demo) else os.path.join(ROOT, args.demo)
    want = condition_of(demo_path)
    demo_motion = load_motion(demo_path)

    steps = min(args.steps, min(len(m["dpos"]) for m in motion))
    print(f"mechanism={args.mechanism}  demonstration {os.path.basename(demo_path)} "
         f"(its own condition {want})")
    print(f"{len(planner.candidates)} B1 candidates, horizon {args.horizon}, {steps} steps")
    print("KINEMATIC: the body is posed, not simulated. It cannot fall.\n")

    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.getObject("sim")
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

    pos = np.array([args.spawn[0], args.spawn[1], demo_motion["height"]])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    sim.setObjectPosition(cam, sim.handle_world,
                          [pos[0] + off_xy[0] + args.cam_dx, pos[1] + off_xy[1] + args.cam_dy,
                           cam_z])

    # **Allocentric `cam` above is for YOU to watch, never for the model.** `--mechanism rollout`
    # feeds `e_t` into VJEPA2/ITM/FTM every step, and those were calibrated on the egocentric
    # convention (`data/egocentric/beh12_b1_ego_flat`) -- a third-person frame here would be
    # out-of-distribution and confound the result. A second camera, mounted egocentrically and
    # PARENTED to `root`, rides along automatically as `pose()` moves `root` each step, matching
    # `render_b1_replay.py --ego`'s own setup (same room-building call, same pitch compensation).
    # **Built for every mode, not just rollout.** Rollout NEEDS it (it is `e_t`); direct does not
    # read sim vision at all, so mounting it there changes the pictures and nothing else -- and the
    # three-panel comparison video needs the new body's egocentric view under every mode, including
    # the ones that never consume it. Verified harmless for direct: its score comes from
    # `body_head(proj(a))`, which never touches a camera.
    cam_ego = None
    if True:
        sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
        from ego_camera import attach_ego, build_texture_box, randomise_ground, room_for, WALK_PITCH
        cam_ego = sim.createVisionSensor(
            1 | 2 | 4, [256, 256, 0, 0],
            [0.01, 20.0, np.deg2rad(24.0), 0.05, 0, 0, 0, 0, 0, 0, 0])
        sim.setObjectAlias(cam_ego, "vjepa_cam_ego_loop")
        R = room_for(root0[2])
        build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=0,
                          centre=(float(pos[0]), float(pos[1])))
        # **Forward comes from the body's own heading, not a hardcoded +x.** `render_b1_replay.py`
        # derives it via `heading()` because a crabbing robot travels along -y while FACING +x, and
        # a camera aimed down the travel direction is a different experiment from the one the
        # training clips recorded (its own F62/F108 note).
        psi = float(heading(quat[None], args.embodiment)[0])
        attach_ego(sim, cam_ego, root, [float(np.cos(psi)), float(np.sin(psi)), 0.0], (0, 0, 0),
                   offset_frac=R["offset_frac"], pitch_comp=WALK_PITCH["b1"])
        # **`--floor_scale 3.0`, matching how the training clips were recorded**
        # (`scripts/dataset/recollect_b1_more.py` and `render_b1_replay.py` both default to 3.0).
        # Without it the floor is a third of the size the wall box was built for, so the ego view
        # shows a BLACK BAND of background through the gap between floor edge and wall -- measured
        # 4.7% near-black pixels against the training clips' 0.000, with the horizon at 72% of frame
        # height instead of 55%. That band is fed straight to VJEPA2 as `e_t` under --mechanism
        # rollout, so this is a distribution mismatch in the model's input, not a cosmetic one.
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                  if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            top = sim.getObject("/Floor")

            def _surface():
                q = sim.getObjectPosition(top, sim.handle_world)
                bb = sim.getShapeBB(top)
                return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
            _before = _surface()
            sim.scaleObjects(floors, 3.0, False)
            _drop = _surface() - _before
            for h in floors:                    # keep the walking surface where it was
                q = sim.getObjectPosition(h, sim.handle_world)
                sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - _drop])
        randomise_ground(sim, seed=0, uv=R["ground_uv"])   # after the scaling, as in the collector
        print("  egocentric camera mounted for rollout's e_t (separate from the viewing camera)")

    def pose(jangles):
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in pos])
        sim.setObjectQuaternion(root, sim.handle_world,
                                [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
        for h, a in zip(joints, jangles):
            sim.setJointPosition(h, float(a))
        view = capture(sim, cam)
        ego_view = capture(sim, cam_ego) if cam_ego is not None else None
        return view, ego_view

    demo_index = next((i for i, c in enumerate(planner.candidates) if c["path"] == demo_path), None)

    frames, chosen, heads, quats, all_scores, ego_frames = [], [], [], [], [], []
    observation, ego_observation = pose(demo_motion["jpos"][0])
    replan_t = replan_i = replan_tau0 = replan_sc = None
    for t in range(steps):
        motion_idx = t   # overridden below only for --mechanism direct with free_offset=True
        if t < args.warm_start:
            i, label = demo_index, f"warm:{want}"
            src = demo_motion
            all_scores.append(np.full(len(planner.candidates), np.nan, np.float32))
        else:
            if args.mechanism == "direct":
                # **`tau` is the candidate-internal index the chosen ACTION actually came from --
                # under free_offset=True it is NOT `t`.** Reading this candidate's MOTION at `t`
                # while its ACTION was scored/executed from `tau` would pose the body along one
                # candidate's timeline while believing it is executing another point of it -- a
                # real bug caught while wiring this through, not a hypothetical.
                if args.free_offset:
                    # `score_offsets` (and therefore `act`) does not depend on `t` at all, so
                    # calling it fresh on any fixed schedule with an UNCHANGING goal returns the
                    # IDENTICAL (candidate, tau0) every single time -- confirmed directly by
                    # calling it twice back to back. **Re-searching on a periodic schedule (every
                    # `horizon` steps) therefore does not advance anything -- it snaps back to the
                    # same tau0 each time**, so the body replays the SAME short window (here, 5
                    # frames) over and over for the whole episode. That window's own net dpos/dquat
                    # over its 5 frames is not zero (measured: net dpos [0.032, 0.018, -0.057], a
                    # real per-cycle sink and pitch) -- replaying it ~11 times compounds into a
                    # catastrophic, smooth tip-over into the floor (measured: up.z 1.0 -> 0.40,
                    # height +0.43 -> -0.09, below ground). Found from a video, not a table.
                    #
                    # Fix: replan ONCE (right after warm start), then let `tau` advance
                    # CONTINUOUSLY for the rest of the episode -- never re-search on a schedule,
                    # which is what caused the snap-back. Only re-search when the CURRENT window
                    # actually runs out of the candidate's own recorded length, which is the one
                    # principled reason to abandon progress and pick a new (candidate, tau0).
                    if replan_t is None:
                        _, replan_i, replan_sc, replan_tau0 = planner.act(goal, t)
                        replan_t = t
                    tau = replan_tau0 + (t - replan_t)
                    if tau >= len(motion[replan_i]["dpos"]) - 1:
                        _, replan_i, replan_sc, replan_tau0 = planner.act(goal, t)
                        replan_t = t
                        tau = replan_tau0
                    i, sc = replan_i, replan_sc
                else:
                    _, i, sc, tau = planner.act(goal, t)
                motion_idx = tau
            else:
                # egocentric only -- the allocentric `observation` is for the saved video alone
                e_t = encode_clip(encoder, np.asarray(ego_observation)[None], 1).float()
                if obs_offset is not None:
                    e_t = e_t - obs_offset.to(e_t.device)
                _, i, sc = planner.act(e_t, goal, t)
            all_scores.append(np.asarray(sc, np.float32))
            label = planner.candidates[i]["condition"]
            src = motion[i]
        motion_idx = min(motion_idx, len(src["dpos"]) - 1, len(src["jpos"]) - 2)
        chosen.append(label)
        pos = pos + quat_wxyz_to_R(quat) @ src["dpos"][motion_idx]
        quat = quat_mul(quat, src["dquat"][motion_idx])
        quat = quat / np.linalg.norm(quat)
        observation, ego_observation = pose(src["jpos"][motion_idx + 1])
        frames.append(observation)
        if ego_observation is not None:
            ego_frames.append(ego_observation)
        heads.append(pos.copy())
        quats.append(quat.copy())
        if t % 10 == 0:
            print(f"  step {t:3d}  -> {label}", flush=True)
        if args.travel > 0 and float(np.linalg.norm(pos[:2] - np.array(args.spawn))) >= args.travel:
            break

    # **This is a KINEMATIC loop: each decision consumes exactly one frame of whichever candidate
    # won, at that body's own recorded rate.** Not the physics loop's "20Hz planner holding a 50Hz
    # policy's command" convention (`close_loop_b1_physics.py`) -- there's no separate decision
    # rate here to hold, so `dt` has to be the body's own native recording rate or every downstream
    # Froude computation on the saved file is wrong by whatever ratio the two differ by. This was
    # hardcoded to 0.05 (B1/hexapod's old convention) and would have silently corrupted gecko's
    # numbers by 2.5x (0.05 vs gecko's real 0.02) the first time this script was pointed at it.
    real_dt = {"hexapod": HEXAPOD_DT, "b1": B1_DT, "gecko": GECKO_DT}[args.embodiment]

    # The goal the loop actually consumed, in BOTH forms: standardised (what the planner compares
    # against) and Froude (what a human can read). Saved because the video and every later analysis
    # otherwise has to guess which of the two goal sources produced the number.
    _g = goal.detach().cpu().numpy().astype(np.float32)
    _mean = np.asarray(checkpoint["body_stats"][0]).ravel()[:len(_g)]
    _std = np.asarray(checkpoint["body_stats"][1]).ravel()[:len(_g)]

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{args.mechanism}-{goal_source}_"
                                f"{os.path.splitext(os.path.basename(goal_path))[0]}"
                                f"_{os.path.splitext(os.path.basename(demo_path))[0]}.npz")
    np.savez_compressed(
        out, frames=np.asarray(frames, np.uint8), head=np.asarray(heads, np.float32),
        ego_frames=np.asarray(ego_frames, np.uint8),
        body_quat=np.asarray(quats, np.float32), dt=np.float32(real_dt),
        goal_used_std=_g, goal_used_froude=(_g * _std + _mean).astype(np.float32),
        goal_reference_froude=np.asarray(goal_reference, np.float32),
        embodiment=args.embodiment, demo_condition=want, chosen=np.asarray(chosen),
        scores=np.asarray(all_scores, np.float32),
        goal=os.path.basename(goal_path), goal_embodiment=goal_embodiment,
        demo=os.path.basename(demo_path), kinematic=np.array(True),
        mechanism=np.array(args.mechanism), goal_source=np.array(goal_source),
        horizon=np.int32(planner.horizon), warm_start=np.int32(args.warm_start),
        ckpt=os.path.relpath(args.ckpt, ROOT),
        candidates=np.asarray([c["condition"] for c in planner.candidates]))
    planned = [c for c in chosen if not c.startswith("warm:")]
    print(f"\nchosen condition counts (planned steps only), mechanism={args.mechanism}:")
    for cond, n in Counter(planned).most_common():
        print(f"  {cond:<12} {n:>3} ({n/max(len(planned),1):.0%})")
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
