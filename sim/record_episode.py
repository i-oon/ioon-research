"""Record an episode: RGB frames + time-aligned joint commands (a_t).

This is the piece that was missing entirely — the link between the CoppeliaSim
stick insect and the V-JEPA2 pipeline. Until now `scripts/` ran on pre-recorded
B1 quadruped video from other renderers, and `sim/` emitted joint state only.

Alignment is by construction, which is the whole point: within one stepped
simulation tick we (a) advance physics, (b) render the vision sensor, (c) read
the joint targets. Frame k and a_t[k] therefore describe the same instant. A
screen recording cannot give this — it captures at wall-clock rate while the sim
steps at its own, so frames drop/duplicate and the pairing drifts. The Motion
Decoder's loss (L_motion = ||MD(x_t, z_t) - a_t||^2) is only meaningful if the
pairing is exact.

Requires the scene to already have a camera + proper floor:
  python sim/set_floor_texture.py --scene <scene>
  python sim/add_camera.py        --scene <scene>

Usage (CoppeliaSim must be running):
  python sim/record_episode.py --scene sim/env/medauroidea_stick_insect.ttt \\
      --steps 300 --out data/episodes/long_walk_000
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SENSOR_NAME = "vjepa_cam"
TRACK_OBJ = "/head"        # camera follows this in x/y
LEG_SUFFIXES = ["_FL", "_ML", "_HL", "_FR", "_MR", "_HR"]
JOINT_NAMES = ["/m1", "/m2", "/m3"]   # ThC, CTr, FTi


def get_joint_handles(sim):
    """18 joint handles, ordered leg-major: [FL_m1, FL_m2, FL_m3, ML_m1, ...].
    Matches the a_t ordering used by CoppeliaSimEnv / expert.py."""
    handles = []
    for leg in LEG_SUFFIXES:
        for j in JOINT_NAMES:
            handles.append(sim.getObject(f"{j}{leg}"))
    return handles


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)
    return np.flipud(img).copy()   # CoppeliaSim returns bottom-up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--warmup", type=int, default=20, help="discard N settling steps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")
    sim.loadScene(os.path.abspath(args.scene))

    cam = sim.getObject("/" + SENSOR_NAME)
    track = sim.getObject(TRACK_OBJ)
    joints = get_joint_handles(sim)

    # camera offset relative to the tracked object, captured from the authored
    # pose. Held fixed for the whole episode -> constant apparent size, and
    # identical relative framing across every morphology.
    cam_pos0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    trk_pos0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    offset_xy = cam_pos0[:2] - trk_pos0[:2]
    cam_z = cam_pos0[2]

    sim.setStepping(True)
    sim.startSimulation()

    frames, actions, track_xy = [], [], []
    total = args.warmup + args.steps
    for k in range(total):
        sim.step()

        # camera tracks x/y only: fixed height keeps body-bob as visible signal,
        # fixed orientation makes a TURN read as a heading change in-frame
        p = np.array(sim.getObjectPosition(track, sim.handle_world))
        sim.setObjectPosition(cam, sim.handle_world,
                              [p[0] + offset_xy[0], p[1] + offset_xy[1], cam_z])

        if k < args.warmup:
            continue

        frames.append(capture(sim, cam))
        actions.append([sim.getJointTargetPosition(h) for h in joints])
        track_xy.append(p[:2])

    sim.stopSimulation()

    frames = np.asarray(frames, dtype=np.uint8)      # (N, 256, 256, 3)
    actions = np.asarray(actions, dtype=np.float32)  # (N, 18)

    np.save(os.path.join(args.out, "frames.npy"), frames)
    np.save(os.path.join(args.out, "actions.npy"), actions)

    moved = float(np.linalg.norm(np.array(track_xy)[-1] - np.array(track_xy)[0]))
    print(f"scene   : {os.path.basename(args.scene)}")
    print(f"frames  : {frames.shape}  dtype={frames.dtype}")
    print(f"actions : {actions.shape}  dtype={actions.dtype}")
    print(f"a_t range: [{actions.min():.3f}, {actions.max():.3f}] rad")
    print(f"a_t std/joint (first 6): {np.round(actions.std(axis=0)[:6], 4)}")
    print(f"robot moved: {moved:.3f} m over {args.steps} steps")
    print(f"frame brightness: mean={frames.mean():.1f} std={frames.std():.1f}")
    assert len(frames) == len(actions), "frame/action count mismatch"
    print(f"\nsaved -> {args.out}/{{frames,actions}}.npy")


if __name__ == "__main__":
    main()
