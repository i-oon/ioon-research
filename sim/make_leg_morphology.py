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


# Which segment's factor governs each link in the chain. Every segment's origin sits at its
# geometric centre, so its length is split into two offsets -- parent joint to centre, and
# centre to the next joint -- and both belong to that segment. The foot and force sensor hang
# off the end of the tibia and follow it.
GOVERNED_BY = {
    "coxa": "coxa", "m2": "coxa",
    "femur": "femur", "m3": "femur",
    "tibia": "tibia", "forceSensor": "tibia", "foot": "tibia",
}


def scale_leg(sim, suffix, factors):
    """factors: {"coxa": f, "femur": f, "tibia": f}, each scaling one segment independently.

    Uniform scaling makes morphology a line, so a held-out body is always a point between two
    training bodies and the task reduces to interpolating one number. Scaling the three
    segments separately makes it a volume, and a held-out body can be a combination of segment
    lengths that no training body has, which is a test of composition rather than of
    interpolation along a line.
    """
    for name in FULL_CHAIN[1:]:  # skip m1, it's the root, nothing to reposition it against
        h = get_object_robust(sim, name, suffix)
        factor = factors[GOVERNED_BY[name]]

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


def segment_lengths(sim, suffix):
    """Per-segment span, joint to joint, so each factor can be checked on its own."""
    spans = {}
    for segment, (start, stop) in (("coxa", ("m1", "m2")),
                                   ("femur", ("m2", "m3")),
                                   ("tibia", ("m3", "foot"))):
        a = sim.getObjectPosition(get_object_robust(sim, start, suffix), sim.handle_world)
        b = sim.getObjectPosition(get_object_robust(sim, stop, suffix), sim.handle_world)
        spans[segment] = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float,
                    help="scale all three segments by one value (the original behaviour)")
    ap.add_argument("--coxa", type=float, help="scale the coxa only")
    ap.add_argument("--femur", type=float, help="scale the femur only")
    ap.add_argument("--tibia", type=float, help="scale the tibia only")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    if args.factor is None and args.coxa is args.femur is args.tibia is None:
        raise SystemExit("give --factor, or any of --coxa/--femur/--tibia")
    base = args.factor if args.factor is not None else 1.0
    factors = {"coxa": args.coxa if args.coxa is not None else base,
               "femur": args.femur if args.femur is not None else base,
               "tibia": args.tibia if args.tibia is not None else base}

    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim.loadScene(BASE_SCENE)

    reach_before = {s: total_reach(sim, s) for s in LEG_SUFFIXES}
    spans_before = {s: segment_lengths(sim, s) for s in LEG_SUFFIXES}

    for suffix in LEG_SUFFIXES:
        scale_leg(sim, suffix, factors)

    reach_after = {s: total_reach(sim, s) for s in LEG_SUFFIXES}
    spans_after = {s: segment_lengths(sim, s) for s in LEG_SUFFIXES}

    print("target factors: " + "  ".join(f"{k} {v:.3f}" for k, v in factors.items()))
    print(f"{'leg':6s} {'reach':>18s} {'coxa':>8s} {'femur':>8s} {'tibia':>8s}  (achieved ratios)")
    for s in LEG_SUFFIXES:
        ratios = {k: spans_after[s][k] / spans_before[s][k] for k in factors}
        print(f"{s:6s} {reach_before[s]:8.5f}->{reach_after[s]:8.5f} "
              f"{ratios['coxa']:8.4f} {ratios['femur']:8.4f} {ratios['tibia']:8.4f}")

    worst = max(abs(spans_after[s][k] / spans_before[s][k] - factors[k])
                for s in LEG_SUFFIXES for k in factors)
    print(f"\nlargest deviation from target: {worst:.4f}")
    if worst > 0.01:
        print("WARNING: a segment did not scale as asked; the chain mapping may be wrong")

    # CoppeliaSim resolves a relative path against its own working directory, not ours, so a
    # scene asked for as sim/env/x.ttt silently lands under the simulator's install tree
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sim.saveScene(out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
