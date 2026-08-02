"""Add a vision sensor (camera) to a Medauroidea scene variant.

Creates the camera PROGRAMMATICALLY so it is byte-identical across every
morphology variant. This matters: per the render-style confound documented in
PROGRESS.md §5-6 and direction_plan.md, if the camera differs at all between
morphology recording sessions, Step 1.5 measures "which session was this" rather
than morphology-vs-behavior. A scripted camera makes that impossible by
construction; a hand-dragged viewport does not.

Outputs 256x256 directly (V-JEPA2's native input size), which also avoids the
resize/crop guesswork that scripts/temporal_similarity_quantified.py had to
reverse-engineer (RESIZE=292/CROP=256).

The camera TRACKS the robot's x/y at a fixed offset, with fixed world
orientation and fixed height. Rationale:
  - the robot walks 3.5-4.8 m per episode, so a world-fixed camera loses it
  - constant apparent size keeps legs resolvable vs. the 16px patch size
  - fixed orientation (not parented to the body) means a TURN reads as a visible
    heading change in frame, rather than the whole world rotating
  - height not tracked, so body bob is preserved as a visual signal

Usage (CoppeliaSim must already be running, see sim/SOURCES.md):
  python sim/add_camera.py --scene sim/env/medauroidea_stick_insect.ttt
  python sim/add_camera.py --scene sim/env/medauroidea_stick_insect_short.ttt --preview
  python sim/add_camera.py --all          # do all 3 variants
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SENSOR_NAME = "vjepa_cam"
RESOLUTION = 256
ENV_DIR = "/home/aria/ioon-research/sim/env"
VARIANTS = [
    "medauroidea_stick_insect.ttt",           # long / base 1.0x
    "medauroidea_stick_insect_medium.ttt",    # 0.75x
    "medauroidea_stick_insect_short.ttt",     # 0.5x
]

# --- camera framing (identical for every morphology, FIXED to the world) ---
# The camera no longer follows the robot; it is bolted in place so the robot
# visibly travels left->right through a static frame (that world-frame travel is
# exactly what an internal joint encoder cannot report). Far distance + a narrow
# FOV ("telephoto") compresses perspective so apparent size stays ~constant
# across the run, without a true orthographic projection (which V-JEPA2 never saw
# in training). The frame is aimed RUNWAY_AIM metres ahead of the body's start so
# the robot enters near one edge and crosses the centre.
# Constraint unchanged: the frame must contain NO sky/void (ViT register-token
# failure) and NO floor edge. Verify both with --preview before recording.
DISTANCE = 8.0      # metres. Far => perspective compressed => apparent size ~constant.
ELEVATION = 40.0    # degrees above horizontal. >30 keeps the horizon/void out.
AZIMUTH = 90.0      # degrees around +x (robot's forward). 90 = pure side view
VIEW_ANGLE = 15.0   # degrees FOV. Telephoto; balances robot size vs runway (tune with --preview).
TARGET_Z = 0.10     # nominal robot centre height to aim at
RUNWAY_AIM = 0.75   # metres. Robot walks in-frame LEFT (world +x); positive aim starts it on
                    # the RIGHT side but fully inside (whole body clears the edge), then it
                    # walks left. Gate --travel so it never exits the left edge either.


def robot_centre(sim):
    """Nominal robot centre from head + all 6 feet (world frame)."""
    names = ["/head"] + [f"/foot_{s}" for s in ["FL", "ML", "HL", "FR", "MR", "HR"]]
    pts = []
    for n in names:
        try:
            pts.append(sim.getObjectPosition(sim.getObject(n), sim.handle_world))
        except Exception:
            pass
    return np.array(pts).mean(axis=0)


def camera_offset():
    """Fixed offset from robot centre to camera, in world axes."""
    el = np.deg2rad(ELEVATION)
    az = np.deg2rad(AZIMUTH)
    horiz = DISTANCE * np.cos(el)
    return np.array([horiz * np.cos(az), horiz * np.sin(az), DISTANCE * np.sin(el)])


def look_at_matrix(cam_pos, target):
    """3x4 row-major pose matrix for a CoppeliaSim vision sensor at cam_pos
    looking at target.

    NOTE: CoppeliaSim vision sensors view along their own **+Z** axis.
    (Verified empirically: pointing +Z away from the target gives min_depth=1.0
    i.e. an all-black frame; pointing +Z toward it renders correctly.)
    """
    z = target - cam_pos                      # sensor +Z points TOWARD the target
    z = z / np.linalg.norm(z)
    world_up = np.array([0.0, 0.0, 1.0])
    x = np.cross(world_up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = []
    for r in range(3):
        m += [x[r], y[r], z[r], cam_pos[r]]
    return m


def add_camera(sim, scene_path, preview=False):
    scene_path = os.path.abspath(scene_path)  # CoppeliaSim requires absolute paths
    sim.loadScene(scene_path)

    # remove a previous camera so this script is idempotent
    try:
        old = sim.getObject("/" + SENSOR_NAME)
        sim.removeObjects([old])
        print("  removed existing camera")
    except Exception:
        pass

    # Aim at a fixed world point RUNWAY_AIM metres ahead of the body's start,
    # using the head's x/y (consistent across morphologies) rather than the
    # feet-averaged centre (which shifts with leg length). Same offset for all
    # three bodies -> framing identical by construction.
    centre = robot_centre(sim)  # kept for the log line below
    head = np.array(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world))
    target = np.array([head[0] + RUNWAY_AIM, head[1], TARGET_Z])
    cam_pos = target + camera_offset()

    # options: bit0(1)=explicitly handled, bit1(2)=perspective, bit2(4)=hide frustum
    options = 1 | 2 | 4
    int_params = [RESOLUTION, RESOLUTION, 0, 0]
    float_params = [
        0.01,                       # near clip
        20.0,                       # far clip
        np.deg2rad(VIEW_ANGLE),     # view angle
        0.05,                       # sensor display size (cosmetic)
        0.0, 0.0,                   # reserved
        0.0, 0.0, 0.0,              # null pixel rgb
        0.0, 0.0,                   # reserved
    ]
    h = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectAlias(h, SENSOR_NAME)
    sim.setObjectMatrix(h, look_at_matrix(cam_pos, target))

    # createVisionSensor defaults to visibility layer 8, but the robot is on
    # layer 1 and the floor on 32768 -> no overlap -> the sensor renders nothing.
    # Put it on all layers so it sees the whole scene.
    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)

    print(f"  robot centre : {np.round(centre, 3)}")
    print(f"  camera pos   : {np.round(cam_pos, 3)}")
    print(f"  offset       : {np.round(camera_offset(), 3)}  (dist={DISTANCE}m, elev={ELEVATION}deg, az={AZIMUTH}deg)")

    img = None
    if preview:
        img = capture(sim, h)

    sim.saveScene(scene_path)
    print(f"  saved: {scene_path}")
    return h, img


def capture(sim, h):
    """Grab one frame. Returns HxWx3 uint8 RGB."""
    sim.handleVisionSensor(h)
    buf, res = sim.getVisionSensorImg(h)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)
    return np.flipud(img)  # CoppeliaSim returns bottom-up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preview", action="store_true", help="save a preview PNG")
    ap.add_argument("--preview-dir", type=str, default="/tmp")
    args = ap.parse_args()

    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")

    scenes = [os.path.join(ENV_DIR, v) for v in VARIANTS] if args.all else [args.scene]
    if scenes == [None]:
        ap.error("give --scene or --all")

    for s in scenes:
        print(f"\n=== {os.path.basename(s)} ===")
        _, img = add_camera(sim, s, preview=args.preview)
        if img is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            out = os.path.join(args.preview_dir, f"cam_{os.path.basename(s).replace('.ttt', '.png')}")
            plt.imsave(out, img)
            print(f"  preview: {out}")


if __name__ == "__main__":
    main()
