"""Closed loop on the B1 with real physics: MuJoCo is the world, CoppeliaSim is the camera.

**Why two simulators, which is not a workaround.** The B1's policy walks only in its native MuJoCo
physics -- CoppeliaSim's engines either freeze the imported base or cannot reproduce the gait -- so
the B1 dataset was rolled out in MuJoCo and rendered in CoppeliaSim. That split has to be kept
here: rendering the B1 from MuJoCo while the insect comes from CoppeliaSim would let V-JEPA2
separate the two robots by **render style** rather than by morphology, and every cross-embodiment
number in this project would be measuring the wrong thing.

    MuJoCo      holds the physics -- weight, contacts, falling
    CoppeliaSim poses a body from MuJoCo's state and returns the camera image
    planner     reads that image and picks the next behaviour

**This replaces `close_loop_kinematic.py`'s central compromise.** That file poses the body from a
recorded clip, so the robot cannot fall and `S.R. survival` passes by construction. Here the robot
carries its own weight and the survival column means something.

**What it can and cannot reach, measured before building it.** Replaying single clips at the rate
the planner runs them -- 50 ms per decision, not the policy's native 20 ms -- the forward clip
survives all 66 steps and the turning and sideways clips fall at 28 and 27. So a full-length
physics episode is available for forward travel and half an episode for the others, and **falling
is a result to report rather than a reason not to run.** F93's "0 of 8" was measured over 300
steps at 50 Hz, six seconds; this loop is three.

    .venv/bin/python3 sim/control/close_loop_b1_physics.py \\
        --ckpt wm/runs/beh12_hexonly/stage3_b1_nce.pt \\
        --projector wm/runs/beh12_hexonly/projector_stage3_nce.pt \\
        --demo data/beh12_b1_flat/b1_ep2.npz --out results/wm/closed_loop/b1_physics
"""
import argparse
import os
import sys

import numpy as np
import torch
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
import mujoco  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle  # noqa: E402
from rollout_b1_mujoco import (ACTION_SCALE, DEFAULT_IL, MODEL, SPAWN_Z,  # noqa: E402
                               il_to_sdk, sdk_to_il)

from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import LatentPlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--goal", default="",
                    help="clip supplying the **goal frames**, which may be a different robot. The "
                         "candidates stay B1 clips because only those are executable, and only the "
                         "goal crosses -- which is the form a cross-embodiment control result has "
                         "to take. Defaults to `--demo`, i.e. same-robot goals")
    ap.add_argument("--goal_embodiment", default="",
                    help="embodiment of `--goal`, for the centring offset. Defaults to --embodiment")
    ap.add_argument("--candidates_dir", default="data/beh12_b1_flat")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--warm_start", type=int, default=10,
                    help="steps driven by the demonstration's own commands before the planner "
                         "takes over. From a standstill there is no motion in the frame to read")
    ap.add_argument("--commit", type=int, default=1,
                    help="hold the chosen behaviour for this many steps before deciding again. "
                         "**1 is re-deciding every step, which is what every run so far used and "
                         "what nothing has justified.** Switching clips was measured to cost half "
                         "the turning and four fifths of the lateral travel (F102), so committing "
                         "is the cheapest thing that could recover them")
    ap.add_argument("--steps", type=int, default=66)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--fall_ratio", type=float, default=0.6,
                    help="body height below this fraction of its settled height counts as fallen")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop/b1_physics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_path = os.path.join(ROOT, args.demo)
    demo = load(demo_path, REGISTRY[args.embodiment])
    with np.load(demo_path, allow_pickle=True) as raw:
        want = str(raw["condition"])

    planner = LatentPlanner.from_checkpoint(
        os.path.join(ROOT, args.ckpt), os.path.join(ROOT, args.candidates_dir),
        embodiment=args.embodiment, projector_path=os.path.join(ROOT, args.projector),
        horizon=args.horizon, device=str(device))
    checkpoint = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    offset = offset_for(checkpoint, args.embodiment)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    # **The goal may come from another robot; the warm start and the seeding never can.** Those
    # need executable commands and a physical state, so they stay with `--demo`.
    goal_path = os.path.join(ROOT, args.goal) if args.goal else demo_path
    goal_emb = args.goal_embodiment or args.embodiment
    goal_clip = load(goal_path, REGISTRY[goal_emb]) if goal_path != demo_path else demo
    demo_e = encode_clip(encoder, goal_clip["frames"], 2).float()
    goal_off = offset_for(checkpoint, goal_emb)
    if goal_off is not None:
        demo_e = demo_e - goal_off
    demo_e = demo_e.to(device)
    if goal_path != demo_path:
        with np.load(goal_path, allow_pickle=True) as raw:
            print(f"goal from {os.path.basename(goal_path)} ({goal_emb}, {str(raw['condition'])}) "
                  f"-- body driven is {args.embodiment}")

    steps = min(args.steps, len(demo["actions"]) - 1,
                min(len(c["actions"]) for c in planner.candidates) - 1)
    print(f"demonstration {os.path.basename(demo_path)}  condition {want}")
    print(f"{len(planner.candidates)} candidates, horizon {args.horizon}, {steps} steps")
    print("PHYSICS: MuJoCo carries the weight. The robot can fall.\n")

    # --- MuJoCo: the world -------------------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    # **Start where the demonstration starts, not from a standing pose.** The clips were recorded
    # with the spawn-to-walk transient cropped, so their first action is a command for a robot
    # already mid-stride. Applying it to a robot standing still asks a leg to continue a swing it
    # never began: the body leapt from 0.435 to 0.665 in six steps, a third above its own stance
    # height, before the gait recovered. Seeding the state from the demonstration's first frame
    # removes the discontinuity at the only moment we can remove it.
    with np.load(demo_path, allow_pickle=True) as raw0:
        d.qpos[0:3] = [0.0, 0.0, float(raw0["base_pos"][0][2])]
        d.qpos[3:7] = np.asarray(raw0["base_quat"][0], np.float64)
        d.qpos[7:19] = np.asarray(raw0["joint_pos"][0], np.float64)
        d.qvel[6:18] = np.asarray(raw0["joint_vel"][0], np.float64)
        d.ctrl[:] = np.asarray(raw0["joint_pos"][0], np.float64)
    mujoco.mj_forward(m, d)
    for _ in range(args.settle):
        mujoco.mj_step(m, d)
    settled_z = float(d.qpos[2])
    # **One decision covers 50 ms of physics, not the policy's 20 ms.** The clips are 20 Hz, so a
    # planner step holds its command for two and a half policy steps; stepping only 20 ms would
    # simulate 40% of the episode and report survival the robot never earned.
    sub = int(round(0.05 / m.opt.timestep))

    # --- CoppeliaSim: the camera -------------------------------------------------------------
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

    # **The camera is placed once and never moves again.** `render_b1_replay.py` sets it from the
    # trajectory's *first* frame and leaves it, so in every clip the model was trained on the robot
    # travels across a fixed view. An earlier version of this file moved the camera with the body,
    # which keeps the robot centred and means the background never slides -- the one cue the
    # forward model uses to predict travel. Its one-step error on those frames was **2.9x** the
    # error on recorded clips while the frames themselves scored as barely novel, because a pooled
    # embedding hardly notices the difference and the forward model entirely does.
    p0 = d.qpos[0:3].copy()
    sim.setObjectPosition(cam, sim.handle_world,
                          [float(p0[0]) + off_xy[0], float(p0[1]) + off_xy[1], cam_z])

    def render():
        """Pose the CoppeliaSim body from MuJoCo's state and take the picture."""
        p = d.qpos[0:3]
        w, x, y, z = d.qpos[3:7]
        sim.setObjectPosition(root, sim.handle_world, [float(p[0]), float(p[1]), float(p[2])])
        sim.setObjectQuaternion(root, sim.handle_world, [float(x), float(y), float(z), float(w)])
        for h, a in zip(joints, d.qpos[7:19]):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)

    # **Every candidate's score, not just the winner's.** Which candidate won says nothing about
    # whether the runner-up was a hair behind or nowhere near, and the open question about speed is
    # exactly that: the loop picks the right behaviour family and the wrong rate, so the scores
    # within a family are what decide it. Stored per step, `nan` during warm start.
    frames, chosen, heads, quats, uprights, all_scores = [], [], [], [], [], []
    fell_at = None
    held_i, held_since = None, 0
    observation = render()
    for t in range(steps):
        if t < args.warm_start:
            action, label = demo["actions"][t], f"warm:{want}"
            all_scores.append(np.full(len(planner.candidates), np.nan, np.float32))
        else:
            e_t = encode_clip(encoder, np.asarray(observation)[None], 1).float()
            if offset is not None:
                e_t = e_t - offset.to(e_t.device)
            h = planner.horizon_at(t)
            if held_i is not None and t - held_since < args.commit:
                # inside the commitment window nothing is decided, so nothing is scored
                i = held_i
                all_scores.append(all_scores[-1])
                action = planner.candidates[i]["actions"][
                    min(t, len(planner.candidates[i]["actions"]) - 1)]
            else:
                action, i, sc = planner.act(e_t[0:1].to(device),
                                            demo_e[min(t + h, len(demo_e) - 1)], t)
                all_scores.append(np.asarray(sc, np.float32))
                if i != held_i:
                    held_since = t
                held_i = i
            label = planner.candidates[i]["condition"]
        chosen.append(label)

        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * np.asarray(action, np.float32))
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(sub):
            mujoco.mj_step(m, d)

        observation = render()
        frames.append(observation)
        heads.append(np.array(d.qpos[0:3], np.float32))
        quats.append(np.array(d.qpos[3:7], np.float32))
        w, x, y, z = d.qpos[3:7]
        uprights.append(float(1 - 2 * (x * x + y * y)))
        if t % 10 == 0:
            print(f"  step {t:3d}  -> {label}   z {float(d.qpos[2]):.3f}", flush=True)
        if fell_at is None and (float(d.qpos[2]) < args.fall_ratio * settled_z
                                or uprights[-1] < 0.5):
            fell_at = t
            print(f"  FELL at step {t}", flush=True)
            break

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    # **The goal belongs in the name.** Two runs differing only in which robot supplied the goal
    # wrote the same file and the first was silently lost.
    stem = os.path.splitext(os.path.basename(demo_path))[0]
    if goal_path != demo_path:
        stem += "__goal_" + os.path.splitext(os.path.basename(goal_path))[0]
    out = os.path.join(out_dir, f"phys_{stem}.npz")
    np.savez_compressed(
        out, frames=np.asarray(frames, np.uint8), head=np.asarray(heads, np.float32),
        body_quat=np.asarray(quats, np.float32), upright=np.asarray(uprights, np.float32),
        dt=np.float32(0.05), embodiment=args.embodiment, condition=want,
        chosen=np.asarray(chosen), scores=np.asarray(all_scores, np.float32),
        demo=os.path.basename(demo_path),
        # **What the planner was actually looking at.** When the goal comes from another robot the
        # demonstration only supplies the warm start and the starting state; a video that shows the
        # B1 beside a B1 clip hides the entire point of the run.
        goal=os.path.basename(goal_path), goal_embodiment=goal_emb,
        kinematic=np.array(False), horizon=np.int32(planner.horizon),
        warm_start=np.int32(args.warm_start), commit=np.int32(args.commit),
        settled_z=np.float32(settled_z),
        fell_at=np.int32(-1 if fell_at is None else fell_at),
        candidates=np.asarray([c["condition"] for c in planner.candidates]))
    planned = [c for c in chosen if not c.startswith("warm:")]
    hit = sum(c == want for c in planned) / max(len(planned), 1)
    print(f"\n{hit:.0%} of {len(planned)} planned steps chose {want}")
    print(f"survived {len(chosen)} of {steps} steps"
          f"{'' if fell_at is None else f' -- fell at {fell_at}'}")
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
