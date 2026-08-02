"""Build sim/env/b1_flat.ttt: fresh scene + floor + imported B1 + fixed vjepa_cam.

Idempotent: closes to a blank scene and rebuilds from scratch each run, so re-running
after tweaking the camera constants just replaces the scene. Writes a preview PNG.

  python3 sim/build_b1_scene.py            # build + preview + save
  python3 sim/build_b1_scene.py --no-save  # build + preview only (tune camera)
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

URDF = "/home/aria/ioon-research/sim/assets/b1_description/b1_coppelia.urdf"
# Build B1's scene FROM an insect scene so it inherits the identical floor + lighting
# (render-style consistency for the cross-embodiment vision comparison); strip the insect.
INSECT_SCENE = "/home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt"
KEEP = {"Floor", "DefaultLights", "DefaultCamera", "XYZCameraProxy", "ResizableFloor_5_25"}
OUT = "/home/aria/ioon-research/sim/env/b1_flat.ttt"
SENSOR_NAME = "vjepa_cam"
RESOLUTION = 256
SPAWN_Z = 0.60

# --- camera framing: same rendering params as the insect (add_camera.py), re-scaled
# for B1 (~1.1 m long, ~0.5 m tall vs the insect's ~0.1 m). Side telephoto view. ---
# Match the insect camera's VIEWPOINT (add_camera.py: elevation 40, azimuth 90 side view)
# so the two embodiments share render style; only distance/FOV differ to frame the 10x
# larger body at comparable apparent size.
DISTANCE = 7.0
ELEVATION = 40.0
AZIMUTH = 90.0
VIEW_ANGLE = 24.0
TARGET_Z = 0.35
RUNWAY_AIM = 1.0
FLOOR_SCALE = 4.0    # enlarge the floor so its far edge never enters frame (no void)


def camera_offset():
    el, az = np.deg2rad(ELEVATION), np.deg2rad(AZIMUTH)
    horiz = DISTANCE * np.cos(el)
    return np.array([horiz * np.cos(az), horiz * np.sin(az), DISTANCE * np.sin(el)])


def look_at_matrix(cam_pos, target):
    z = target - cam_pos; z = z / np.linalg.norm(z)          # sensor +Z toward target
    x = np.cross([0, 0, 1.0], z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def add_camera(sim, target):
    cam_pos = target + camera_offset()
    options = 1 | 2 | 4                                       # explicit | perspective | hide frustum
    float_params = [0.01, 30.0, np.deg2rad(VIEW_ANGLE), 0.05, 0, 0, 0, 0, 0, 0, 0]
    h = sim.createVisionSensor(options, [RESOLUTION, RESOLUTION, 0, 0], float_params)
    sim.setObjectAlias(h, SENSOR_NAME)
    sim.setObjectMatrix(h, look_at_matrix(cam_pos, target))
    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)
    return h


def capture(sim, h):
    sim.handleVisionSensor(h)
    buf, res = sim.getVisionSensorImg(h)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    c = RemoteAPIClient("localhost", port=23000)
    sim = c.require("sim")
    urdf_plugin = c.require("simURDF")
    sim.stopSimulation()
    # start from the insect scene (inherit its floor + lighting), then strip the insect
    sim.loadScene(os.path.abspath(INSECT_SCENE))
    tops = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)   # top-level objects
    removed = []
    for h in tops:
        alias = sim.getObjectAlias(h)
        if any(alias.startswith(k) or alias == k for k in KEEP):
            continue
        try:
            subtree = sim.getObjectsInTree(h, sim.handle_all, 1)       # descendants (exclude base)
            sim.removeObjects(subtree + [h])
            removed.append(alias)
        except Exception as e:
            print("  (could not remove", alias, ")", e)
    kept = [sim.getObjectAlias(h) for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2)]
    print(f"stripped insect: removed {removed}")
    print(f"kept: {kept}")

    # --- import B1 URDF ---
    imp = getattr(urdf_plugin, "importFile", None) or getattr(urdf_plugin, "import")
    try:
        imp(URDF, 0)
    except Exception as e:
        print("importFile(path,0) failed, retrying importFile(path):", e)
        imp(URDF)

    joints = {sim.getObjectAlias(h): h
              for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    shapes = {sim.getObjectAlias(h): h
              for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    print(f"\nimported: {len(joints)} joints, {len(shapes)} shapes")
    print("joint aliases:", sorted(joints))
    print("shape aliases:", sorted(shapes))

    # base = the articulation root: walk up from a leg joint until parent == -1
    base = joints["FR_hip_joint"]
    while sim.getObjectParent(base) != -1:
        base = sim.getObjectParent(base)
    print("base (root) alias:", sim.getObjectAlias(base),
          " static:", sim.getObjectInt32Param(base, sim.shapeintparam_static)
          if sim.getObjectType(base) == sim.object_shape_type else "n/a(non-shape root)")
    # ensure the root is free (non-static) so the robot is a floating base; leave the
    # importer's per-shape respondable/visual flags untouched (it sets them correctly).
    if sim.getObjectType(base) == sim.object_shape_type:
        sim.setObjectInt32Param(base, sim.shapeintparam_static, 0)

    # raise base to spawn height (keep x,y) -- moves the whole tree
    p = sim.getObjectPosition(base, sim.handle_world)
    sim.setObjectPosition(base, sim.handle_world, [p[0], p[1], SPAWN_Z])

    # camera aimed ahead of the base
    target = np.array([p[0] + RUNWAY_AIM, p[1], TARGET_Z])
    cam = add_camera(sim, target)
    print("camera pos:", np.round(target + camera_offset(), 3))

    img = capture(sim, cam)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prev = "/tmp/b1_scene_preview.png"
    plt.imsave(prev, img)
    print("preview:", prev, " mean px:", float(img.mean()))

    if not args.no_save:
        sim.saveScene(OUT)
        print("saved:", OUT)


if __name__ == "__main__":
    main()
