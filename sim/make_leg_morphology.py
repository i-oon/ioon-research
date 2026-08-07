"""Create a leg-length morphology variant of the base Medauroidea stick insect scene.

Scales the coxa/femur/tibia segments of all 6 legs by a uniform factor along
each segment's local long axis (confirmed to be local Z for all 3 segment
types), and repositions each segment's child joint to match — sim.scaleObject
only resizes geometry, it does not reposition children automatically.

Usage:
  python sim/make_leg_morphology.py --factor 0.7 --out sim/env/medauroidea_stick_insect_short.ttt
  python sim/make_leg_morphology.py --factor 0.85 --out sim/env/medauroidea_stick_insect_medium.ttt
"""
import argparse
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_SCENE = os.path.join(ROOT, "sim", "env", "medauroidea_stick_insect.ttt")
LEG_SUFFIXES = ["_FL", "_ML", "_HL", "_FR", "_MR", "_HR"]
# full kinematic chain per leg, root first. Each segment's own origin sits at
# its geometric center, so its length is split into two equal offsets: one
# from its parent joint to the segment's center, and one from the segment's
# center to the next joint. Both offsets must be scaled, not just the second.
FULL_CHAIN = ["m1", "coxa", "m2", "femur", "m3", "tibia", "forceSensor", "foot"]
SEGMENTS = {"coxa", "femur", "tibia"}  # these have geometry to scale


def get_object_robust(sim, name, suffix):
    """Handles a known naming typo in the base scene: '/tibia_HR' is
    actually named '/tibial_HR' (extra 'l')."""
    try:
        return sim.getObject(f"/{name}{suffix}")
    except Exception:
        if name == "tibia":
            return sim.getObject(f"/tibial{suffix}")
        raise


def scale_leg(sim, suffix, factor):
    for name in FULL_CHAIN[1:]:  # skip m1, it's the root, nothing to reposition it against
        h = get_object_robust(sim, name, suffix)

        pos = sim.getObjectPosition(h, sim.handle_parent)
        new_pos = [p * factor for p in pos]
        sim.setObjectPosition(h, sim.handle_parent, new_pos)

        if name in SEGMENTS:
            # scale only the local long axis (Z), keep cross-section thickness unchanged
            sim.scaleObject(h, 1.0, 1.0, factor, 0)


def total_reach(sim, suffix):
    """Distance from the ThC joint (m1) to the foot tip, for verification."""
    m1 = sim.getObject(f"/m1{suffix}")
    foot = sim.getObject(f"/foot{suffix}")
    p1 = sim.getObjectPosition(m1, sim.handle_world)
    p2 = sim.getObjectPosition(foot, sim.handle_world)
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")

    sim.loadScene(BASE_SCENE)

    reach_before = {s: total_reach(sim, s) for s in LEG_SUFFIXES}

    for suffix in LEG_SUFFIXES:
        scale_leg(sim, suffix, args.factor)

    reach_after = {s: total_reach(sim, s) for s in LEG_SUFFIXES}

    print(f"{'leg':6s} {'before':>10s} {'after':>10s} {'ratio':>8s} (target={args.factor})")
    for s in LEG_SUFFIXES:
        ratio = reach_after[s] / reach_before[s]
        print(f"{s:6s} {reach_before[s]:10.5f} {reach_after[s]:10.5f} {ratio:8.4f}")

    sim.saveScene(args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
