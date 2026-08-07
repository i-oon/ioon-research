"""Render walking previews on the original airl-insect-walking uneven scenes."""
import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


SCENES = [
    ("uneven", "/home/aria/ioon-research/airl-insect-walking/env/medauroidea_stick_insect_uneven.ttt"),
    ("uneven_flat", "/home/aria/ioon-research/airl-insect-walking/env/medauroidea_stick_insect_uneven_flat.ttt"),
]
SENSOR = "uneven_preview_cam"
RES = 480


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def get_optional(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def look_at(cam_pos, target):
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def add_cam(sim):
    old = get_optional(sim, "/" + SENSOR)
    if old is not None:
        sim.removeObjects([old])
    head = np.asarray(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world), dtype=float)
    target = np.array([head[0] + 0.8, head[1], 0.12])
    cam_pos = target + np.array([0.0, 4.2, 2.0])
    h = sim.createVisionSensor(1 | 2 | 4, [RES, RES, 0, 0],
                               [0.01, 30.0, np.deg2rad(38.0), 0.05, 0, 0, 0, 0, 0, 0, 0])
    sim.setObjectAlias(h, SENSOR)
    sim.setObjectMatrix(h, look_at(cam_pos, target))
    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)
    return h


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def label(frame, text):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def render(sim, name, scene, steps, warmup, out_dir, fps):
    sim.loadScene(scene)
    settle(sim)
    cam = add_cam(sim)
    head = sim.getObject("/head")
    sim.setStepping(True)
    sim.startSimulation()
    frames, heads = [], []
    for k in range(warmup + steps):
        sim.step()
        if k < warmup:
            continue
        p = np.asarray(sim.getObjectPosition(head, sim.handle_world), dtype=float)
        heads.append(p)
        frames.append(label(capture(sim, cam), f"{name} t={k-warmup:03d}"))
    sim.stopSimulation()
    settle(sim)
    heads = np.asarray(heads)
    dx = float(heads[-1, 0] - heads[0, 0]) if len(heads) else 0.0
    dy = float(heads[-1, 1] - heads[0, 1]) if len(heads) else 0.0
    zf = float(heads[-1, 2]) if len(heads) else 0.0
    out = os.path.join(out_dir, f"{name}_walk.mp4")
    imageio.mimsave(out, frames, fps=fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"{name:12s} frames={len(frames):3d} dx={dx:+.3f} dy={dy:+.3f} final_z={zf:.3f} -> {out}")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", default="results/uneven_original_walk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    settle(sim)
    all_frames = [render(sim, name, scene, args.steps, args.warmup, args.out_dir, args.fps)
                  for name, scene in SCENES]
    n = min(len(x) for x in all_frames)
    grid = [np.hstack([x[i] for x in all_frames]) for i in range(n)]
    grid_path = os.path.join(args.out_dir, "grid_uneven_original.mp4")
    imageio.mimsave(grid_path, grid, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
