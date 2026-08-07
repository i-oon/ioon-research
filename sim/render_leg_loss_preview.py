"""Render preview images of 4-leg stick-insect variants.

This is a visual design check for the Stage-2 held-out body, not a walking
controller. The current Medauroidea scene scripts assume six legs, so removing
legs and then running physics needs a separate controller/data path. Here we
only load the scene, remove a symmetric leg pair, and capture the fixed camera.
"""
import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


BASE_SCENE = "/home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt"
SENSOR_NAME = "vjepa_cam"
PREVIEW_CAM = "leg_loss_preview_cam"
RESOLUTION = 360
LEG_PAIRS = {
    "front_loss": ("FL", "FR"),
    "middle_loss": ("ML", "MR"),
    "hind_loss": ("HL", "HR"),
}
CHAIN_NAMES = ("m1", "coxa", "m2", "femur", "m3", "tibia", "tibial", "forceSensor", "foot")


def look_at_matrix(cam_pos, target):
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def get_optional(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def leg_objects(sim, leg):
    handles = []
    for name in CHAIN_NAMES:
        h = get_optional(sim, f"/{name}_{leg}")
        if h is not None:
            handles.append(h)
    return handles


def remove_leg_pair(sim, legs):
    handles = []
    for leg in legs:
        h = get_optional(sim, f"/m1_{leg}")
        if h is not None:
            handles.extend(sim.getObjectsInTree(h, sim.handle_all, 1) + [h])
        else:
            handles.extend(leg_objects(sim, leg))
    # Remove children before parents and deduplicate while preserving order.
    unique = []
    for h in handles:
        if h not in unique:
            unique.append(h)
    if unique:
        sim.removeObjects(unique)
    return unique


def add_preview_camera(sim):
    old = get_optional(sim, "/" + PREVIEW_CAM)
    if old is not None:
        sim.removeObjects([old])
    head = np.array(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world))
    target = np.array([head[0], head[1], 0.05])
    cam_pos = target + np.array([-1.25, -2.35, 1.55])
    options = 1 | 2 | 4
    int_params = [RESOLUTION, RESOLUTION, 0, 0]
    float_params = [0.01, 20.0, np.deg2rad(28.0), 0.05, 0, 0, 0, 0, 0, 0, 0]
    cam = sim.createVisionSensor(options, int_params, float_params)
    sim.setObjectAlias(cam, PREVIEW_CAM)
    sim.setObjectMatrix(cam, look_at_matrix(cam_pos, target))
    sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0xFFFF)
    return cam


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def label_image(frame, text):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return img


def render_variant(sim, name, legs, out_dir, save_scene):
    sim.loadScene(BASE_SCENE)
    settle(sim)
    removed = remove_leg_pair(sim, legs)
    cam = add_preview_camera(sim)
    frame = capture(sim, cam)
    label = f"{name}: removed {','.join(legs)}"
    img = label_image(frame, label)
    png = os.path.join(out_dir, f"{name}.png")
    img.save(png)
    scene_path = None
    if save_scene:
        scene_path = os.path.join(out_dir, f"medauroidea_stick_insect_{name}.ttt")
        sim.saveScene(scene_path)
    print(f"{name:12s} removed={len(removed):2d} -> {png}")
    if scene_path:
        print(f"{'':12s} scene -> {scene_path}")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out-dir", default="results/leg_loss_preview")
    ap.add_argument("--save-scene", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)

    images = []
    sim.loadScene(BASE_SCENE)
    settle(sim)
    cam = add_preview_camera(sim)
    base = label_image(capture(sim, cam), "six_leg_base")
    base_path = os.path.join(args.out_dir, "six_leg_base.png")
    base.save(base_path)
    print(f"{'six_leg_base':12s} -> {base_path}")
    images.append(base)

    for name, legs in LEG_PAIRS.items():
        images.append(render_variant(sim, name, legs, args.out_dir, args.save_scene))

    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h))
    for i, img in enumerate(images):
        sheet.paste(img, (i * w, 0))
    sheet_path = os.path.join(args.out_dir, "leg_loss_contact_sheet.png")
    sheet.save(sheet_path)
    print(f"\ncontact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
