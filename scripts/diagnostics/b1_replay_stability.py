"""Does a recorded B1 action sequence keep the robot upright when replayed open loop?

**This decides whether a closed loop is possible on the B1 at all**, before anything is built for
it. The planner's candidates are recorded action sequences -- that is the design choice that lets it
work on a robot whose kinematics are unknown -- and replaying one requires the sequence to be a
*plan* rather than the output of a feedback law.

    hexapod   IK and a CPG write joint targets from a clock. No state is read. A recorded
              sequence replays exactly, which is why the closed loop on it works at all.

    B1        a PPO policy reads base orientation, joint state and a phase clock at 50 Hz and
              emits joint targets. Its output is a *response*, not a plan. Replayed without the
              state it was responding to, there is no reason for it to stay up.

If the B1 falls, candidates-as-recorded-sequences do not transfer to robots whose gait is closed
loop, and that is a finding about the method rather than a missing feature.

Uses `data/b1_traj/*.npz` at the native 50 Hz, not the 20 Hz clips in `data/allocentric/beh12_b1_flat`:
resampling a control signal to 40% of its rate would be a second reason to fall and would confound
the answer.

  .venv/bin/python3 scripts/diagnostics/b1_replay_stability.py
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))

import mujoco  # noqa: E402

from rollout_b1_mujoco import (ACTION_SCALE, DECIMATION, DEFAULT_IL, MODEL,  # noqa: E402
                               SPAWN_Z, il_to_sdk)


def replay(model_path, actions, warmup, fall_ratio, frames=None, every=2):
    """`frames` is an optional list to append rendered images to -- **the numbers say the robot
    fell and this project's rule is to look.** Two clips in `ik_walk_8body` passed every summary
    statistic they were checked against and were tumbling; a height ratio is the same kind of
    evidence."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 240, 320) if frames is not None else None
    d.qpos[0:3] = [0, 0, SPAWN_Z]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = il_to_sdk(DEFAULT_IL)
    d.ctrl[:] = il_to_sdk(DEFAULT_IL)
    for _ in range(warmup * DECIMATION):
        mujoco.mj_step(m, d)

    start_z = float(d.qpos[2])
    heights, upright = [], []
    for t in range(len(actions)):
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * actions[t])
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        if renderer is not None and t % every == 0:
            renderer.update_scene(d, camera=-1)
            frames.append(renderer.render())
        heights.append(float(d.qpos[2]))
        # world z of the body's own up axis: 1 standing, 0 on its side, negative upside down
        w, x, y, z = d.qpos[3:7]
        upright.append(float(1 - 2 * (x * x + y * y)))
        if heights[-1] < fall_ratio * start_z or upright[-1] < 0.5:
            # keep rendering a little past the fall, or the video stops before it is visible
            if renderer is not None:
                for _ in range(20):
                    for _ in range(DECIMATION):
                        mujoco.mj_step(m, d)
                    renderer.update_scene(d, camera=-1)
                    frames.append(renderer.render())
            return t + 1, heights, upright, start_z
    return len(actions), heights, upright, start_z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="data/b1_traj", help="native-rate rollouts, 50 Hz")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--fall_ratio", type=float, default=0.7,
                    help="fraction of the settled height below which the robot has fallen")
    ap.add_argument("--limit", type=int, default=8, help="trajectories to test")
    ap.add_argument("--video", default="",
                    help="directory to write one mp4 per trajectory. The verdict here is a fall, "
                         "and a fall is the kind of claim that has to be looked at.")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"MuJoCo model not found: {args.model}")
    paths = sorted(glob.glob(os.path.join(ROOT, args.traj, "*.npz")))[:args.limit]
    if not paths:
        raise SystemExit(f"no trajectories in {args.traj}")

    print(f"{'trajectory':<22}{'steps':>8}{'survived':>10}{'end z / z0':>12}{'end upright':>13}")
    survived = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            actions = data["action"].astype(np.float64)
        frames = [] if args.video else None
        n, heights, upright, z0 = replay(args.model, actions, args.warmup, args.fall_ratio, frames)
        if args.video:
            import sys as _sys
            _sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
            from npz_to_video import write_mp4
            out_dir = os.path.join(ROOT, args.video)
            os.makedirs(out_dir, exist_ok=True)
            write_mp4(os.path.join(out_dir, os.path.basename(path).replace(".npz", ".mp4")),
                      iter(frames), 25)
        ok = n == len(actions)
        survived += ok
        print(f"{os.path.basename(path):<22}{len(actions):>8}{n:>10}"
              f"{heights[-1] / z0:>12.2f}{upright[-1]:>13.2f}{'' if ok else '   FELL'}")

    print(f"\n{survived} of {len(paths)} replayed to the end")
    if survived == len(paths):
        print("Open-loop replay holds the B1 up. Candidates-as-recorded-sequences are viable on it,")
        print("and a closed loop can be built the same way it was for the hexapod.")
    else:
        print("Open-loop replay does not hold the B1 up. Its gait is a feedback response, not a")
        print("plan, so a planner choosing among recorded sequences cannot drive it. A closed loop")
        print("on this robot needs a different action representation -- which is a finding, not a")
        print("missing feature.")


if __name__ == "__main__":
    main()
