"""Read-only audit of the B1 and insect CoppeliaSim scene/camera configuration."""

import argparse
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCENES = (
    ("insect", "sim/env/medauroidea_stick_insect.ttt", "/head"),
    ("b1_original", "sim/env/b1_flat.ttt", "/base_visual"),
    ("b1_convex", "sim/env/b1_flat_convex.ttt", "/base_visual"),
)
FLOAT_PARAMS = {"near": 1000, "far": 1001, "fov": 1004}
INT_PARAMS = {"width": 1002, "height": 1003}


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()

    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    for label, relative_scene, tracked_alias in SCENES:
        settle(sim)
        scene = os.path.join(ROOT, relative_scene)
        sim.loadScene(scene)
        settle(sim)
        cam = sim.getObject("/vjepa_cam")
        tracked = sim.getObject(tracked_alias)
        cam_pos = sim.getObjectPosition(cam, sim.handle_world)
        tracked_pos = sim.getObjectPosition(tracked, sim.handle_world)
        offset = [cam_pos[0] - tracked_pos[0], cam_pos[1] - tracked_pos[1], cam_pos[2]]
        floats = {name: sim.getObjectFloatParam(cam, key) for name, key in FLOAT_PARAMS.items()}
        ints = {name: sim.getObjectInt32Param(cam, key) for name, key in INT_PARAMS.items()}
        respondable = []
        nonconvex = []
        for handle in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type):
            if sim.getObjectInt32Param(handle, sim.shapeintparam_respondable):
                alias = sim.getObjectAlias(handle, 1)
                respondable.append(alias)
                if not sim.getObjectInt32Param(handle, sim.shapeintparam_convex):
                    nonconvex.append(alias)
        print(f"[{label}] {scene}")
        print(f"  engine={sim.getInt32Param(sim.intparam_dynamic_engine)} camera_offset={offset}")
        print(f"  camera_orientation={sim.getObjectOrientation(cam, sim.handle_world)}")
        print(f"  camera_float={floats} camera_int={ints}")
        print(f"  respondable_shapes={len(respondable)} nonconvex_respondable={len(nonconvex)}")
        if nonconvex:
            print(f"  nonconvex_aliases={nonconvex}")


if __name__ == "__main__":
    main()
