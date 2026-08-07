"""Render the three stick-insect morphologies walking on terrain scenes.

This is a qualitative preview: all bodies replay the same expert joint command
cycle on their corresponding terrain scene. It is not IK-retargeted and not a
trained terrain policy.
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
    ("long", "medauroidea_stick_insect_terrain.ttt"),
    ("medium", "medauroidea_stick_insect_medium_terrain.ttt"),
    ("short", "medauroidea_stick_insect_short_terrain.ttt"),
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


def render_one(sim, morph, scene, cmds, steps, warmup, fps, out_dir):
    sim.loadScene(os.path.join(ENV, scene))
    settle(sim)
    cam = sim.getObject("/" + SENSOR)
    head = sim.getObject(TRACK)
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in JN]

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(warmup):
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        sim.step()

    frames, heads = [], []
    for t in range(steps):
        cmd = cmds[t % len(cmds)]
        for h, v in zip(joints, cmd):
            sim.setJointTargetPosition(h, float(v))
        sim.step()
        p = np.asarray(sim.getObjectPosition(head, sim.handle_world), dtype=float)
        heads.append(p)
        frames.append(label(capture(sim, cam), f"{morph} terrain  t={t:03d}"))
    sim.stopSimulation()
    settle(sim)

    heads = np.asarray(heads)
    dx = float(heads[-1, 0] - heads[0, 0]) if len(heads) else 0.0
    dy = float(heads[-1, 1] - heads[0, 1]) if len(heads) else 0.0
    zf = float(heads[-1, 2]) if len(heads) else 0.0
    out = os.path.join(out_dir, f"{morph}_terrain_walk.mp4")
    imageio.mimsave(out, frames, fps=fps, codec="libx264", macro_block_size=1,
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
    ap.add_argument("--out-dir", default="results/terrain_morph_walk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(CSV)
    cmds = expert_cmds(df, args.episode)
    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)

    all_frames = [render_one(sim, morph, scene, cmds, args.steps, args.warmup, args.fps, args.out_dir)
                  for morph, scene in SCENES]
    n = min(len(frames) for frames in all_frames)
    grid = [np.hstack([frames[i] for frames in all_frames]) for i in range(n)]
    grid_path = os.path.join(args.out_dir, "grid_terrain_walk.mp4")
    imageio.mimsave(grid_path, grid, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
