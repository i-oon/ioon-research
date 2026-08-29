"""Create a leg-length morphology variant of the base Medauroidea stick insect scene.

Scales the coxa/femur/tibia segments of all 6 legs by a uniform factor along
each segment's local long axis (confirmed to be local Z for all 3 segment
types), and repositions each segment's child joint to match — sim.scaleObject
only resizes geometry, it does not reposition children automatically.

Usage:
  python sim/scene/make_leg_morphology.py --factor 0.7 --out sim/env/medauroidea_stick_insect_short.ttt
  python sim/scene/make_leg_morphology.py --factor 0.85 --out sim/env/medauroidea_stick_insect_medium.ttt


CHOOSING PROPORTIONS
====================

Read this before picking scales. Two bodies already in `data/fwd_hex8body` collapse and rotate
on the spot rather than walking, and two more veer 0.40 m off course, all because of the one
constraint below. See FINDINGS.md F42.

1. The hard constraint: the leg must reach its own trajectory
------------------------------------------------------------
A two-link chain (femur + tibia) reaches only distances between `|femur - tibia|` and
`femur + tibia` — the triangle inequality. Closer than `|femur - tibia|` the knee would have to
fold past straight, so the IK gives up and settles wherever it can.

    |femur - tibia|  <  92.5 mm          the closest commanded target, at collector --scale 0.5
    femur + tibia    >  farthest target   not yet measured; worth recording

`|femur - tibia|` is the **dead zone**. What was measured, per body, on recorded episodes:

    12 - 71 mm    walks normally
    94.6 mm       walks but yaws steadily, ending 0.40 m off course
    208 mm        tips over within ~12 frames and rotates on the spot

The boundary is a step, not a safety margin: 94.6 mm misses 0.3% of its targets, 132.5 mm misses
24%. `check_reachable` below enforces it.

2. What that leaves you
-----------------------
Base segments are femur 342.9 mm, tibia 413.9 mm — ratio 0.83, **tibia longer**, which is the
real stick insect proportion. Holding the femur at 1.0 and moving the tibia:

    tibia scale 0.61 - 1.05   ratio 0.79 - 1.37    reaches its targets
    tibia scale 0.83 - 1.05   ratio 0.79 - 1.00    also plausible as a stick insect

**The usable band is about 0.2 wide.** That is why every body in the dataset has the femur and
tibia effectively tied together — a geometric constraint, not an oversight in dataset design, and
the direct cause of the extrapolation limit in FINDINGS.md F33.

Ratio above 1.0 is not itself a defect: bodies at 1.04, 1.07 and 1.10 walk normally. It only
becomes one by pushing the dead zone past 92.5 mm. But femur longer than tibia inverts the
animal's own proportion, so a body above 1.0 is a robot morphology, not a stick insect.

3. The coxa is the free parameter — use it
------------------------------------------
The coxa does not appear in `|femur - tibia|`; it positions the shoulder. Lengthening it moves
the shoulder away from the foot targets, so a larger dead zone becomes reachable. And it is
behaviourally almost free: coxa explains the gait grouping at ARI +0.038, and c10f10t10 against
c06f10t10 — a 40% coxa change — have contact patterns agreeing at 0.984. **40% of coxa buys 1.6%
of gait.**

Still to measure: how many mm of dead-zone headroom one mm of coxa buys. Readable from the scene
geometry, no simulation needed.

4. Never rescale the foot trajectory per body
---------------------------------------------
It would relieve the constraint and it must not be done. `lambda_cross` is well defined only
because every body walks identical expert episodes: pairing body A's latent with body B's frame
at the same instant means something because the *intent* is shared. Per-body targets turn that
pairing into a wrong label, not a noisy one.

5. Verify after generating — the check above is necessary, not sufficient
------------------------------------------------------------------------
    signed forward displacement    > 0.3 m per episode
    lateral drift, measured apart  < 0.2 m per episode
    watch the video

Use the signed forward component, never `norm(head[-1,:2] - head[0,:2])`. That unsigned form is
how a body that tumbles in place passed inspection: it reads a healthy 0.46 m. And no number
distinguishes "walks oddly" from "fell over and is now spinning" — look at the frames.
`scripts/dataset/compare_ratio_gaits.py` and `scripts/dataset/plot_gait_quality.py` do both.
"""
import argparse
import os

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# Measured from the base scene: the femur and tibia lengths, and the closest the collector's
# foot targets ever come to the shoulder joint. The closest approach was minimised over all 30
# expert episodes the dataset uses, at the collector's default --scale 0.5; the spread across
# episodes is 92.5 to 93.7 mm, so it is a property of the gait rather than of one episode.
BASE_FEMUR_M = 0.3429
BASE_TIBIA_M = 0.4139
CLOSEST_TARGET_M = 0.0925
REFERENCE_SCALE = 0.5


def check_reachable(factors, target_scale, force):
    """Refuse to generate a body whose foot cannot reach the trajectory it will be asked to walk.

    A two-link chain reaches only distances between |femur - tibia| and femur + tibia -- the
    triangle inequality, since those two links and the shoulder-to-foot distance are the three
    sides. Below |femur - tibia| the knee would have to fold past straight, so the IK gives up and
    settles wherever it can: bodies 40 mm past the limit lost a quarter of their targets and
    returned residuals of 350 to 810 mm, against under 20 mm for every body inside it.

    The limit is not a safety margin, it is a step: the closest target sits at 92.5 mm, so a body
    at 94.6 mm misses 0.3 percent of them and still walks, while one at 132.5 mm misses 24 percent
    and does not.
    """
    femur = BASE_FEMUR_M * factors["femur"]
    tibia = BASE_TIBIA_M * factors["tibia"]
    dead_zone = abs(femur - tibia)
    # the targets scale with the collector's --scale, so the limit does too
    limit = CLOSEST_TARGET_M * (target_scale / REFERENCE_SCALE)
    margin = limit - dead_zone
    print(f"reach check: femur {femur*1000:.1f} mm, tibia {tibia*1000:.1f} mm, "
          f"dead zone {dead_zone*1000:.1f} mm against a closest target of {limit*1000:.1f} mm "
          f"-> {margin*1000:+.1f} mm")
    if margin < 0:
        message = (f"this body cannot reach {abs(margin)*1000:.1f} mm inside its own dead zone; "
                   f"the IK will not solve and the body will not walk. Bring the femur and tibia "
                   f"scales closer together, or pass --force.")
        if not force:
            raise SystemExit("refusing to generate: " + message)
        print("WARNING, --force given: " + message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=float,
                    help="scale all three segments by one value (the original behaviour)")
    ap.add_argument("--coxa", type=float, help="scale the coxa only")
    ap.add_argument("--femur", type=float, help="scale the femur only")
    ap.add_argument("--tibia", type=float, help="scale the tibia only")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--target_scale", type=float, default=0.5,
                    help="the collector's --scale; the reach check depends on it")
    ap.add_argument("--force", action="store_true",
                    help="generate even if the body cannot reach the collector's foot targets")
    args = ap.parse_args()

    if args.factor is None and args.coxa is args.femur is args.tibia is None:
        raise SystemExit("give --factor, or any of --coxa/--femur/--tibia")
    base = args.factor if args.factor is not None else 1.0
    factors = {"coxa": args.coxa if args.coxa is not None else base,
               "femur": args.femur if args.femur is not None else base,
               "tibia": args.tibia if args.tibia is not None else base}
    check_reachable(factors, args.target_scale, args.force)

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
