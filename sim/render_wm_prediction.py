"""Drive the robot with world-model predicted joint commands, next to the IK ground truth.

Takes the npz written by wm.predict_actions and runs each command sequence open-loop through
the same physics and the same fixed camera used to collect the training data, so the two
videos differ only in where the joint targets came from. Foot force sensors are recorded on
both passes, which is what the gait diagram is built from.

Open-loop means nothing corrects the joint angles as they drift, so this shows the physical
consequence of the reconstruction error rather than a controller tracking a reference.

Needs a CoppeliaSim with the ZMQ remote API listening, e.g.
  cd /home/aria/CoppeliaSim && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23000

Run from the repository root:
  .venv/bin/python3 sim/render_wm_prediction.py --port 23000 \
      --pred results/wm/predictions/<run>_epoch020_medium.npz --clip 0
"""
import argparse
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from collect_ik import SCENES, drive_and_record  # noqa: E402

SCENE_BY_MORPH = dict(SCENES)


def scene_for(morph):
    """The three leg-scale bodies are named in collect_ik; the segment-scale bodies are not,
    because they were recorded by passing --morphs NAME=SCENE explicitly. They follow one
    convention, so derive it rather than duplicating the table."""
    if morph in SCENE_BY_MORPH:
        return SCENE_BY_MORPH[morph]
    derived = f"medauroidea_{morph}.ttt"
    if os.path.exists(os.path.join(ROOT, "sim", "env", derived)):
        return derived
    raise SystemExit(f"no scene for body {morph!r}; looked for sim/env/{derived}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--pred", required=True, help="npz from wm.predict_actions")
    ap.add_argument("--clip", type=int, default=0, help="which clip inside the npz")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--travel", type=float, default=0.0,
                    help="0 replays every step; a positive value stops once the robot has "
                         "covered that many metres, matching the collector's framing gate")
    ap.add_argument("--cam_dx", type=float, default=0.0)
    ap.add_argument("--cam_dy", type=float, default=0.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
    ap.add_argument("--scene", default="", help="override the .ttt; defaults to the body's own")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = np.load(args.pred, allow_pickle=True)
    lengths = data["lengths"]
    if not 0 <= args.clip < len(lengths):
        raise SystemExit(f"--clip must be in 0..{len(lengths) - 1}")
    start = int(np.sum(lengths[:args.clip]))
    stop = start + int(lengths[args.clip])
    morph = str(data["morph"])
    scene = args.scene or scene_for(morph)

    sequences = {"predicted": data["pred"][start:stop], "ground_truth": data["gt"][start:stop]}
    print(f"body '{morph}' ({'held out' if bool(data['held_out']) else 'seen in training'}), "
          f"clip {str(data['clips'][args.clip])}, {stop - start} steps, scene {scene}")

    sim = RemoteAPIClient(port=args.port).require("sim")
    recorded = {}
    for name, cmds in sequences.items():
        frames, actions, forces, heads = drive_and_record(
            sim, scene, np.asarray(cmds, np.float32), args.travel, args.warmup,
            cam_dx=args.cam_dx, cam_dy=args.cam_dy, spawn=tuple(args.spawn),
        )
        step = heads[-1] - heads[0]
        # x is the walking direction and y is sideways; the magnitude alone would score a
        # robot that veers off course as if it had walked that far forward
        forward, lateral = float(step[0]), float(step[1])
        heading = float(np.degrees(np.arctan2(step[1], step[0])))
        recorded[name] = {"frames": frames, "forces": forces, "heads": heads, "actions": actions}
        print(f"  {name:<13} {len(frames)} steps | forward {forward:+.3f} m | "
              f"lateral {lateral:+.3f} m | heading {heading:+.1f} deg | "
              f"body height {heads[-1][2]:.3f} m | mean feet down "
              f"{(forces > 0.5).sum(axis=1).mean():.2f} of 6")

    out = args.out or os.path.join(
        ROOT, "results", "wm", "replay",
        f"{os.path.splitext(os.path.basename(args.pred))[0]}_clip{args.clip}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        pred_frames=recorded["predicted"]["frames"],
        pred_forces=recorded["predicted"]["forces"],
        pred_heads=recorded["predicted"]["heads"],
        pred_actions=recorded["predicted"]["actions"],
        gt_frames=recorded["ground_truth"]["frames"],
        gt_forces=recorded["ground_truth"]["forces"],
        gt_heads=recorded["ground_truth"]["heads"],
        gt_actions=recorded["ground_truth"]["actions"],
        morph=morph, held_out=bool(data["held_out"]), epoch=int(data["epoch"]),
        clip=str(data["clips"][args.clip]), source=os.path.basename(args.pred),
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
