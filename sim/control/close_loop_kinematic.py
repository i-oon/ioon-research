"""Kinematic closed loop: the planner chooses, the body is posed rather than simulated.

**This is not the loop `close_loop_ik.py` runs and must not be reported as if it were.** There is no
physics here. The robot is placed frame by frame, so it cannot fall, and `S.R. survival` is passing
by construction rather than by result.

It exists because the dynamic loop is impossible on the B1 and the reason is the robot: its gait is
a PPO policy reading state at 50 Hz, so a recorded action sequence is a *response* and re-issuing it
open loop drops the robot in 72 to 289 steps (F93). That closes the execution half of the question
and leaves the selection half untested across embodiments -- which is the half this measures.

    what it tests    does the planner keep choosing the right behaviour when the frames it sees
                     next are produced by what it chose now. That is the covariate shift F92 found
                     on the hexapod, and it has never been measured across robots.

    what it cannot   whether the chosen sequence is executable. Splicing two clips' motion gives a
                     smooth trajectory whether or not a real B1 could produce it.

**Motion composes as per-step deltas in the body frame, never as absolute pose.** Each clip stores
where *that* rollout was; taking position directly from whichever clip won this step teleports the
robot every time the choice changes. The delta is rotated out of the source clip's heading and into
the current one, so a turn chosen after a walk continues from where the walk left off.

  .venv/bin/python3 sim/control/close_loop_kinematic.py --ckpt wm/runs/beh12_hexonly/best.pt \\
      --demo data/beh12_b1_flat/b1_ep1.npz
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
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import LatentPlanner, condition_of  # noqa: E402


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
    """Per-step body-frame deltas and joint angles for one clip.

    The translation delta is expressed in the *source* clip's body frame at that step, so applying
    it under a different current heading rotates with the robot instead of dragging it along the
    direction the recording happened to face.
    """
    with np.load(path, allow_pickle=True) as d:
        pos = d["base_pos"].astype("float64")
        quat = d["base_quat"].astype("float64")     # (w, x, y, z)
        jpos = d["joint_pos"].astype("float64")
    n = len(pos) - 1
    dpos = np.einsum("nij,nj->ni", np.array([quat_wxyz_to_R(quat[t]).T for t in range(n)]),
                     pos[1:] - pos[:-1])
    dquat = np.array([quat_mul(quat_conj(quat[t]), quat[t + 1]) for t in range(n)])
    return {"dpos": dpos, "dquat": dquat, "jpos": jpos, "height": float(np.median(pos[:, 2]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="")
    ap.add_argument("--demo", required=True)
    ap.add_argument("--candidates_dir", default="data/beh12_b1_flat")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--steps", type=int, default=66)
    ap.add_argument("--warm_start", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1,
                    help="**Deterministic here.** No physics and a deterministic planner means "
                         "repeats are identical; kept only so the output layout matches the "
                         "dynamic loop's.")
    ap.add_argument("--travel", type=float, default=2.0)
    ap.add_argument("--cam_dx", type=float, default=0.0)
    ap.add_argument("--cam_dy", type=float, default=0.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop/kinematic")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    planner = LatentPlanner.from_checkpoint(
        ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment,
        os.path.join(ROOT, args.projector) if args.projector else "",
        horizon=args.horizon, device=str(device))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    offset = offset_for(checkpoint, args.embodiment)

    motion = [load_motion(c["path"]) for c in planner.candidates]
    demo_path = args.demo if os.path.isabs(args.demo) else os.path.join(ROOT, args.demo)
    want = condition_of(demo_path)
    demo_motion = load_motion(demo_path)
    with np.load(demo_path, allow_pickle=True) as d:
        demo_frames = d["frames"]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    demo_e = encode_clip(encoder, demo_frames, 2).float()
    if offset is not None:
        demo_e = demo_e - offset.to(demo_e.device)

    steps = min(args.steps, len(demo_e) - args.horizon - planner.action_lag - 1,
                min(len(m["dpos"]) for m in motion))
    print(f"demonstration {os.path.basename(demo_path)}  condition {want}")
    print(f"{len(planner.candidates)} candidates, horizon {args.horizon}, {steps} steps")
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

    def pose(jangles):
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in pos])
        sim.setObjectQuaternion(root, sim.handle_world,
                                [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
        for h, a in zip(joints, jangles):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)

    demo_index = planner.candidates.index(
        next((c for c in planner.candidates if c["path"] == demo_path), planner.candidates[0])) \
        if any(c["path"] == demo_path for c in planner.candidates) else None

    frames, chosen, heads, quats = [], [], [], []
    observation = pose(demo_motion["jpos"][0])
    for t in range(steps):
        if t < args.warm_start:
            i, label = demo_index, f"warm:{want}"
            src = demo_motion
        else:
            e_t = encode_clip(encoder, np.asarray(observation)[None], 1).float()
            if offset is not None:
                e_t = e_t - offset.to(e_t.device)
            h = planner.horizon_at(t)
            _, i, _ = planner.act(e_t[0:1].to(device),
                                  demo_e[min(t + h, len(demo_e) - 1)], t)
            label = planner.candidates[i]["condition"]
            src = motion[i]
        chosen.append(label)
        # compose in the body frame: rotate the source clip's local step into the current heading
        pos = pos + quat_wxyz_to_R(quat) @ src["dpos"][t]
        quat = quat_mul(quat, src["dquat"][t])
        quat = quat / np.linalg.norm(quat)
        observation = pose(src["jpos"][t + 1])
        frames.append(observation)
        heads.append(pos.copy())
        quats.append(quat.copy())
        if t % 10 == 0:
            print(f"  step {t:3d}  -> {label}", flush=True)
        if args.travel > 0 and float(np.linalg.norm(pos[:2] - np.array(args.spawn))) >= args.travel:
            break

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"kine_{os.path.splitext(os.path.basename(demo_path))[0]}.npz")
    np.savez_compressed(
        out, frames=np.asarray(frames, np.uint8), head=np.asarray(heads, np.float32),
        body_quat=np.asarray(quats, np.float32), dt=np.float32(0.05),
        embodiment=args.embodiment, condition=want, chosen=np.asarray(chosen),
        demo=os.path.basename(demo_path), kinematic=np.array(True),
        candidates=np.asarray([c["condition"] for c in planner.candidates]))
    planned = [c for c in chosen if not c.startswith("warm:")]
    hit = sum(c == want for c in planned) / max(len(planned), 1)
    print(f"\n{hit:.0%} of {len(planned)} planned steps chose {want}")
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
