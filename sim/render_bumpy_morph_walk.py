"""Render three stick-insect morphologies on a bumpy/uneven floor.

This uses the current migrated morphology scenes/controller handles, then adds
temporary static bump shapes at render time. It is meant as a qualitative
preview of the old "uneven/bumpy" terrain idea without depending on the old
airl-insect-walking scene scripts, which no longer drive the robot correctly.
"""
import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


ROOT = "/home/aria/ioon-research"
ENV = os.path.join(ROOT, "sim/env")
CSV = os.path.join(ENV, "expert_66k_aug3c_fcontact.csv")
SCENES = [
    ("long", "medauroidea_stick_insect.ttt"),
    ("medium", "medauroidea_stick_insect_medium.ttt"),
    ("short", "medauroidea_stick_insect_short.ttt"),
]
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
JN = ["m1", "m2", "m3"]
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}
EP = 66
SENSOR = "vjepa_cam"
TRACK = "/head"


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def label(frame, text):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return np.asarray(img)


def expert_cmds(df, ep):
    rows = df.iloc[ep * EP:(ep + 1) * EP]
    cols = [f"motor_cmd_{leg}_{SEG[jn]}" for leg in LEGS for jn in JN]
    return rows[cols].to_numpy(np.float32)


def add_bumpy_floor(sim, center_xy, x_span, y_span, amplitude, spacing):
    """Add small static ellipsoid bumps around the walking corridor."""
    cx, cy = center_xy
    xs = np.arange(cx - 0.20, cx + x_span, spacing)
    ys = np.arange(cy - y_span / 2.0, cy + y_span / 2.0 + 1e-6, spacing)
    rng = np.random.default_rng(20260806)
    made = []

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            # Leave a gentle central corridor, but not perfectly flat.
            lane = abs(y - cy)
            base = 0.45 + 0.55 * np.sin(2.7 * x + 1.4 * iy) ** 2
            height = amplitude * base * (0.55 if lane < 0.12 else 1.0)
            if rng.random() < 0.12:
                height *= 1.35
            sx = spacing * rng.uniform(0.45, 0.70)
            sy = spacing * rng.uniform(0.45, 0.80)
            sz = max(0.008, height)
            h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, [sx, sy, sz], 0)
            sim.setObjectAlias(h, f"uneven_bump_{ix:02d}_{iy:02d}")
            sim.setObjectPosition(h, sim.handle_world, [float(x), float(y), float(sz / 2.0 - 0.004)])
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
            sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.27, 0.23, 0.18])
            made.append(h)

    # Add a thin dark base plane so the bumps read visually from the side camera.
    base = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [x_span + 0.45, y_span + 0.25, 0.012], 0)
    sim.setObjectAlias(base, "uneven_bumpy_base")
    sim.setObjectPosition(base, sim.handle_world, [float(cx + x_span / 2.0 - 0.08), float(cy), -0.006])
    sim.setObjectInt32Param(base, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(base, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(base, None, sim.colorcomponent_ambient_diffuse, [0.20, 0.18, 0.15])
    made.append(base)
    return made


def render_one(sim, morph, scene, cmds, args):
    sim.loadScene(os.path.join(ENV, scene))
    settle(sim)
    cam = sim.getObject("/" + SENSOR)
    head = sim.getObject(TRACK)
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in JN]
    hp = np.asarray(sim.getObjectPosition(head, sim.handle_world), dtype=float)
    add_bumpy_floor(sim, (float(hp[0]), float(hp[1])), args.x_span, args.y_span, args.amplitude, args.spacing)

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(args.warmup):
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()

    frames, heads = [], []
    for t in range(args.steps):
        cmd = cmds[t % len(cmds)]
        for h, v in zip(joints, cmd):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
        p = np.asarray(sim.getObjectPosition(head, sim.handle_world), dtype=float)
        heads.append(p)
        frames.append(label(capture(sim, cam), f"{morph} bumpy uneven  t={t:03d}"))

    sim.stopSimulation()
    settle(sim)

    heads = np.asarray(heads)
    dx = float(heads[-1, 0] - heads[0, 0]) if len(heads) else 0.0
    dy = float(heads[-1, 1] - heads[0, 1]) if len(heads) else 0.0
    zf = float(heads[-1, 2]) if len(heads) else 0.0
    out = os.path.join(args.out_dir, f"{morph}_bumpy_walk.mp4")
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"{morph:6s} frames={len(frames):3d} dx={dx:+.3f} dy={dy:+.3f} final_z={zf:.3f} -> {out}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--episode", type=int, default=926)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--amplitude", type=float, default=0.035)
    ap.add_argument("--spacing", type=float, default=0.22)
    ap.add_argument("--x-span", type=float, default=4.2)
    ap.add_argument("--y-span", type=float, default=1.25)
    ap.add_argument("--out-dir", default="results/bumpy_morph_walk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(CSV)
    cmds = expert_cmds(df, args.episode)
    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)

    all_frames = [render_one(sim, morph, scene, cmds, args) for morph, scene in SCENES]
    n = min(len(frames) for frames in all_frames)
    grid = [np.hstack([frames[i] for frames in all_frames]) for i in range(n)]
    grid_path = os.path.join(args.out_dir, "grid_bumpy_walk.mp4")
    imageio.mimsave(grid_path, grid, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
