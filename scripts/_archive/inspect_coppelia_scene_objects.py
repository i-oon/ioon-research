"""Print a compact object summary for a CoppeliaSim scene.

Used to identify terrain/floor/bump objects without dumping thousands of robot
links and visual children.
"""
import argparse
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--filter", default="terrain,floor,uneven,bump,plane,Resizable,height,path")
    args = ap.parse_args()

    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)
    sim.loadScene(os.path.abspath(args.scene))
    settle(sim)

    needles = [x.strip().lower() for x in args.filter.split(",") if x.strip()]
    top = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 1)
    all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    shapes = sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0)

    print(f"scene: {args.scene}")
    print(f"top-level objects: {len(top)}")
    for h in top:
        alias = sim.getObjectAlias(h, 1)
        typ = sim.getObjectType(h)
        pos = sim.getObjectPosition(h, sim.handle_world)
        print(f"  TOP {alias:45s} type={typ:3d} pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")

    print(f"\nfiltered objects: {len(all_objs)} total, {len(shapes)} shapes")
    hits = []
    for h in all_objs:
        alias = sim.getObjectAlias(h, 1)
        if any(n in alias.lower() for n in needles):
            hits.append(h)
    for h in hits[:120]:
        alias = sim.getObjectAlias(h, 1)
        typ = sim.getObjectType(h)
        pos = sim.getObjectPosition(h, sim.handle_world)
        parent = sim.getObjectParent(h)
        palias = "<scene>" if parent == -1 else sim.getObjectAlias(parent, 1)
        print(f"  HIT {alias:45s} type={typ:3d} parent={palias:30s} pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")
    if len(hits) > 120:
        print(f"  ... {len(hits)-120} more filtered hits omitted")


if __name__ == "__main__":
    main()
