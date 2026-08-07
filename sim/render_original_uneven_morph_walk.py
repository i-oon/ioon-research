"""Render current morphologies walking on the original uneven Terrain model.

The old airl-insect-walking uneven scene contains a model `/Terrain` with a
heightfield-like `/Terrain/shape`. Loading that whole old scene no longer makes
the robot move in our migrated pipeline, so this script extracts `/Terrain` as a
temporary model and imports it into each current morphology scene before replaying
the expert joint cycle.
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
OLD_UNEVEN = os.path.join(ROOT, "airl-insect-walking/env/medauroidea_stick_insect_uneven.ttt")
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


def export_original_terrain(sim, model_path):
    sim.loadScene(os.path.abspath(OLD_UNEVEN))
    settle(sim)
    terrain = sim.getObject("/Terrain")
    sim.saveModel(terrain, os.path.abspath(model_path))
    print(f"exported original /Terrain -> {model_path}")


def hide_flat_floor(sim):
    for name in ["/Floor", "/Floor/box"]:
        try:
            h = sim.getObject(name)
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        except Exception:
            pass


def import_terrain(sim, model_path, head_pos, x_offset, y_offset):
    h = sim.loadModel(os.path.abspath(model_path))
    sim.setObjectAlias(h, "original_uneven_Terrain")
    sim.setObjectPosition(
        h,
        sim.handle_world,
        [float(head_pos[0] + x_offset), float(head_pos[1] + y_offset), 0.0],
    )
    return h


def render_one(sim, morph, scene, terrain_model, cmds, args):
    sim.loadScene(os.path.join(ENV, scene))
    settle(sim)
    hide_flat_floor(sim)
    head = sim.getObject(TRACK)
    hp = np.asarray(sim.getObjectPosition(head, sim.handle_world), dtype=float)
    import_terrain(sim, terrain_model, hp, args.terrain_x_offset, args.terrain_y_offset)

    cam = sim.getObject("/" + SENSOR)
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in JN]

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
        frames.append(label(capture(sim, cam), f"{morph} original uneven  t={t:03d}"))

    sim.stopSimulation()
    settle(sim)

    heads = np.asarray(heads)
    dx = float(heads[-1, 0] - heads[0, 0]) if len(heads) else 0.0
    dy = float(heads[-1, 1] - heads[0, 1]) if len(heads) else 0.0
    zf = float(heads[-1, 2]) if len(heads) else 0.0
    out = os.path.join(args.out_dir, f"{morph}_original_uneven_walk.mp4")
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
    ap.add_argument("--out-dir", default="results/original_uneven_morph_walk")
    ap.add_argument("--terrain-model", default="/tmp/original_uneven_terrain.ttm")
    ap.add_argument("--terrain-x-offset", type=float, default=0.0,
                    help="place original terrain root this far ahead of /head")
    ap.add_argument("--terrain-y-offset", type=float, default=-0.45,
                    help="place original terrain root this far sideways from /head")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)
    export_original_terrain(sim, args.terrain_model)

    df = pd.read_csv(CSV)
    cmds = expert_cmds(df, args.episode)
    all_frames = [render_one(sim, morph, scene, args.terrain_model, cmds, args) for morph, scene in SCENES]
    n = min(len(frames) for frames in all_frames)
    grid = [np.hstack([frames[i] for frames in all_frames]) for i in range(n)]
    grid_path = os.path.join(args.out_dir, "grid_original_uneven_walk.mp4")
    imageio.mimsave(grid_path, grid, fps=args.fps, codec="libx264", macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"grid -> {grid_path}")


if __name__ == "__main__":
    main()
