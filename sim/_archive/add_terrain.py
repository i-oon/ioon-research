"""Add a STAIRCASE to a scene (vision-vs-proprioception sub-experiment), CLI-tunable.

Staircase up, placed ahead of the robot along +x. Step height/tread/count/width are
CLI flags so we can sweep for a setup the open-loop gait can actually climb. Honest
caveat: only very low steps (well under the foot lift) are climbable open-loop, and
those are gentle -- a real climb needs an adaptive (CPG) controller.

  python3 sim/add_terrain.py --scene sim/env/medauroidea_stick_insect_terrain.ttt \
      --steps 3 --height 0.03 --tread 0.5 --width 1.4 --preview
  python3 sim/add_terrain.py --scene ... --remove
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ENV = "/home/aria/ioon-research/sim/env"
TAG = "terrain_"
SENSOR = "vjepa_cam"
X0 = 0.40          # start distance ahead of the robot's head [m]


def remove_terrain(sim):
    removed = 0
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type):
        try:
            if sim.getObjectAlias(h).startswith("terrain"):
                sim.removeObjects([h]); removed += 1
        except Exception:
            pass
    return removed


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3))


def add_terrain(sim, scene, n, height, tread, width, preview, preview_dir):
    scene = os.path.abspath(scene)
    sim.loadScene(scene)
    r = remove_terrain(sim)
    if r:
        print(f"  removed {r} existing terrain shapes")

    head = np.array(sim.getObjectPosition(sim.getObject("/head"), sim.handle_world))
    hx, hy = float(head[0]), float(head[1])
    for i in range(n):
        top = (i + 1) * height                       # step i: block from floor to top
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [tread, width, top], 0)
        sim.setObjectAlias(h, f"{TAG}stair{i}")
        sim.setObjectPosition(h, sim.handle_world, [hx + X0 + i * tread + tread / 2.0, hy, top / 2.0])
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
        sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, [0.32, 0.28, 0.23])
    print(f"  + staircase: {n} x {height} m = {n*height:.2f} m rise, tread {tread} m, width {width} m, ahead of x={hx:.2f}")

    img = None
    if preview:
        try:
            img = np.array(capture(sim, sim.getObject("/" + SENSOR)))
        except Exception as e:
            print("  (preview skipped)", e)
    sim.saveScene(scene)
    print(f"  saved: {scene}")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--height", type=float, default=0.03)
    ap.add_argument("--tread", type=float, default=0.50)
    ap.add_argument("--width", type=float, default=1.40)
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-dir", default="/tmp")
    args = ap.parse_args()

    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim")

    if args.remove:
        sim.loadScene(os.path.abspath(args.scene))
        n = remove_terrain(sim)
        sim.saveScene(os.path.abspath(args.scene))
        print(f"removed {n} terrain shapes -> {args.scene}")
        return

    img = add_terrain(sim, args.scene, args.steps, args.height, args.tread, args.width,
                      args.preview, args.preview_dir)
    if img is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = os.path.join(args.preview_dir, "terrain_" + os.path.basename(args.scene).replace(".ttt", ".png"))
        plt.imsave(out, img)
        print(f"  preview: {out}")


if __name__ == "__main__":
    main()
