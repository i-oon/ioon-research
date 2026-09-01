"""One contact sheet of egocentric camera variants, so "it looks bad" becomes "that one".

    .venv/bin/python3 sim/scene/ego_preview.py --scene medauroidea_c10f10t10.ttt \\
        --embodiment hexapod --out results/ego_preview.png

**Written because a single wrong preview says nothing about which knob is wrong.** A head view can
look bad for at least five unrelated reasons -- the walls too close or too far, the field of view too
narrow, the camera inside the body, the near clipping plane cutting everything, or the wall texture
tiled so coarsely that a wall is a flat colour -- and they are not distinguishable by staring at one
frame. This renders the same standing robot under a sweep and labels each tile with its settings.

**It also prints the numbers a picture cannot show**: the sensor's field of view, near and far
clipping planes, and the camera's height and distance to the nearest wall. A frame that is black
because the near plane is 10 cm looks exactly like a frame that is black because the camera is
inside the head.
"""
import argparse
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ego_camera import (attach_ego, build_texture_box, clear_box,  # noqa: E402
                        insect_forward, randomise_ground)

ENV = os.path.join(os.path.dirname(HERE), "env")
SENSOR = "vjepa_cam"


def shot(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    img = np.frombuffer(buf, np.uint8).reshape(res[1], res[0], 3)
    return np.flipud(img).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="medauroidea_c10f10t10.ttt")
    ap.add_argument("--embodiment", choices=("hexapod", "b1"), default="hexapod")
    ap.add_argument("--parent", default="/head", help="/head for the insect")
    ap.add_argument("--boxes", type=float, nargs="+", default=[8.0])
    ap.add_argument("--fovs", type=float, nargs="+", default=[75.0, 90.0, 110.0])
    ap.add_argument("--up", type=float, nargs="+", default=[0.02, 0.06])
    ap.add_argument("--pitch", type=float, nargs="+", default=[0.0, -0.25],
                    help="radians; negative looks down. The floor is the surface with "
                         "the most texture and the most optical flow, so tilting into "
                         "it may beat looking level at a distant wall")
    ap.add_argument("--forward", type=float, default=0.03)
    ap.add_argument("--tile", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0, help="appearance seed")
    ap.add_argument("--wall_height", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/ego_preview.png")
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    sim.loadScene(os.path.join(ENV, args.scene))
    cam = sim.getObject("/" + SENSOR)
    parent = sim.getObject(args.parent)

    near = sim.getObjectFloatParam(cam, sim.visionfloatparam_near_clipping)
    far = sim.getObjectFloatParam(cam, sim.visionfloatparam_far_clipping)
    fov0 = np.degrees(sim.getObjectFloatParam(cam, sim.visionfloatparam_perspective_angle))
    print(f"sensor as authored: fov {fov0:.1f} deg, near clip {near:.3f} m, far clip {far:.1f} m")
    print(f"parent {args.parent} at height "
          f"{sim.getObjectPosition(parent, sim.handle_world)[2]:.3f} m\n")
    if near > 0.02:
        print(f"  **near clipping is {near:.3f} m.** At insect scale anything closer than that is "
              f"invisible,\n  which can empty the frame on its own -- lower it if the tiles below "
              f"are blank.\n")

    tiles, labels = [], []
    for box in args.boxes:
        clear_box(sim)
        build_texture_box(sim, size=box, tile=args.tile, height=args.wall_height,
                          seed=args.seed)
        randomise_ground(sim, seed=args.seed)
        for fov in args.fovs:
            sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                    float(np.deg2rad(fov)))
            for up in args.up:
                for pitch in args.pitch:
                    fwd = insect_forward(sim) if args.embodiment == "hexapod" else None
                    if fwd is None:
                        raise SystemExit("--embodiment b1 needs a trajectory; preview the insect "
                                         "first")
                    f = np.asarray(fwd, float)
                    f = f / max(np.linalg.norm(f), 1e-9)
                    f = f * np.cos(pitch) + np.array([0.0, 0.0, np.sin(pitch)])
                    info = attach_ego(sim, cam, parent, f, (0.0, up, args.forward))
                    tiles.append(shot(sim, cam))
                    labels.append(f"box {box:g} fov {fov:g} up {up:g} pitch {pitch:g}")
                    print(f"  {labels[-1]:<34} camera z={info['position'][2]:.3f} m")

    h, w = tiles[0].shape[:2]
    cols = len(args.fovs) * len(args.up)
    cols = min(cols, 6)
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * (w + 4), rows * (h + 18)), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * (w + 4), (i // cols) * (h + 18)
        sheet.paste(Image.fromarray(t), (x + 2, y + 16))
        d.text((x + 4, y + 3), lab, fill=(235, 235, 235))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"\n{args.out}  {len(tiles)} variants, {cols} per row")
    print("**Point at the tile that looks like a head view.** If every tile is blank the fault is "
          "the\nnear clipping plane or the camera sitting inside the body, not the room.")


if __name__ == "__main__":
    main()
