"""Render walking previews for 4-leg stick-insect leg-loss variants.

This is intentionally a rough visual diagnostic. The original scene controller
is still the six-leg open-loop gait, so the missing-leg bodies are expected to
limp, drift, or fall. To avoid breaking the scene script, removed legs are kept
as handles but their visible/respondable shapes are disabled ("ghost removal").

WHAT THE PREVIEW SHOWED, AND WHAT A DATASET WOULD NEED
======================================================

`middle_loss` is the usable variant, and by more than expected: driven by the unchanged six-leg
gait it stays upright and walks away much as the six-leg body does. `front_loss` tips and is lying
diagonal by frame 55; `hind_loss` rears vertical at frame 27 and collapses. Strips in
`results/wm/gait/leg_loss_strips.png`.

Two things this settles about building a 4-leg dataset, both cheaper than the roadmap assumed:

  - **No scene file is needed.** Legs are ghost-removed at runtime from
    `medauroidea_stick_insect.ttt`, which is why `sim/env/` contains nothing 4-legged.
  - **No policy is needed.** The four remaining legs are geometrically unchanged, so the IK
    commands already computed for FL, HL, FR and HR still apply. Dropping the six middle columns
    turns the 18-D command into a 12-D one.

What is missing is the framing. This preview uses a head camera and the body leaves the frame
around step 83 of 139. A dataset render has to match `data/ik_walk_8body` exactly or the model
sees a different scene, not a different body:

    --scale 0.5 --travel 0.8 --warmup 20 --cam_dx -0.6 --cam_dy 0.0 --spawn 0 0

`--cam_dx -0.6 --spawn 0 0` is the pair that keeps the body fully in frame for all 66 frames and
the floor edge out of view; without `--spawn` the robot starts 0.95 m from the edge of a 5x5 m
floor. See direction_plan.md.

Why it is worth building: it is the only body that separates *leg count* from *appearance*. The
hexapod is 6-legged and insect-shaped, the B1 4-legged and quadruped-shaped, so nothing currently
tells us which of the two an embodiment probe reading 0.99 is responding to. A 4-legged insect
does.
"""
import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


BASE_SCENE = "/home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt"
SENSOR_NAME = "leg_loss_walk_cam"
LEG_PAIRS = {
    "six_leg_base": (),
    "front_loss": ("FL", "FR"),
    "middle_loss": ("ML", "MR"),
    "hind_loss": ("HL", "HR"),
}
CHAIN_NAMES = ("m1", "coxa", "m2", "femur", "m3", "tibia", "tibial", "forceSensor", "foot")
RESOLUTION = 360


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def get_optional(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def look_at_matrix(cam_pos, target):
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def add_camera(sim):
    old = get_optional(sim, "/" + SENSOR_NAME)
    if old is not None:
        sim.removeObjects([old])
    head = np.array(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world))
    target = np.array([head[0] + 0.35, head[1], 0.06])
    cam_pos = target + np.array([-1.45, -2.35, 1.45])
    options = 1 | 2 | 4
    int_params = [RESOLUTION, RESOLUTION, 0, 0]
    float_params = [0.01, 20.0, np.deg2rad(30.0), 0.05, 0, 0, 0, 0, 0, 0, 0]
    cam = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectAlias(cam, SENSOR_NAME)
    sim.setObjectMatrix(cam, look_at_matrix(cam_pos, target))
    sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0xFFFF)
    return cam


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def label(frame, text):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def leg_subtree(sim, leg):
    handles = []
    root = get_optional(sim, f"/m1_{leg}")
    if root is not None:
        handles.extend(sim.getObjectsInTree(root, sim.handle_all, 1) + [root])
    for name in CHAIN_NAMES:
        h = get_optional(sim, f"/{name}_{leg}")
        if h is not None and h not in handles:
            handles.append(h)
    return handles


def ghost_remove_legs(sim, legs):
    disabled_shapes = 0
    disabled_joints = 0
    for leg in legs:
        for h in leg_subtree(sim, leg):
            typ = sim.getObjectType(h)
            if typ == sim.object_shape_type:
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
                try:
                    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
                except Exception:
                    pass
                disabled_shapes += 1
            elif typ == sim.object_joint_type:
                try:
                    sim.setJointTargetForce(h, 0.0)
                except Exception:
                    pass
                disabled_joints += 1
    return disabled_shapes, disabled_joints


def render_variant(sim, name, legs, steps, warmup, out_dir, fps):
    sim.loadScene(BASE_SCENE)
    settle(sim)
    disabled_shapes, disabled_joints = ghost_remove_legs(sim, legs)
    cam = add_camera(sim)
    head = sim.getObject("/head")

    sim.setStepping(True)
    sim.startSimulation()
    frames = []
    heads = []
    for k in range(warmup + steps):
        for leg in legs:
            for jn in ("m1", "m2", "m3"):
                h = get_optional(sim, f"/{jn}_{leg}")
                if h is not None:
                    try:
                        sim.setJointTargetForce(h, 0.0)
                    except Exception:
                        pass
        sim.step()
        if k < warmup:
            continue
        p = np.array(sim.getObjectPosition(head, sim.handle_world))
        heads.append(p)
        frames.append(label(capture(sim, cam), f"{name}  t={k - warmup:03d}"))
    sim.stopSimulation()
    settle(sim)

    out = os.path.join(out_dir, f"{name}.mp4")
    imageio.mimsave(out, frames, fps=fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    heads = np.asarray(heads)
    dx = float(heads[-1, 0] - heads[0, 0]) if len(heads) else 0.0
    dy = float(heads[-1, 1] - heads[0, 1]) if len(heads) else 0.0
    zf = float(heads[-1, 2]) if len(heads) else 0.0
    print(f"{name:12s} shapes_off={disabled_shapes:2d} joints_zero={disabled_joints:2d} "
          f"frames={len(frames):3d} dx={dx:+.3f} dy={dy:+.3f} final_z={zf:.3f} -> {out}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out-dir", default="results/leg_loss_walk")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)

    all_frames = []
    for name, legs in LEG_PAIRS.items():
        all_frames.append(render_variant(sim, name, legs, args.steps, args.warmup, args.out_dir, args.fps))

    n = min(len(frames) for frames in all_frames)
    grid = [np.hstack([frames[i] for frames in all_frames]) for i in range(n)]
    grid_path = os.path.join(args.out_dir, "grid_leg_loss_walk.mp4")
    imageio.mimsave(grid_path, grid, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
