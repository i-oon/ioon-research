"""Collect the twelve matched behaviour conditions for one body, from a recipe kept in code.

**The existing `data/beh12_c10f10t10_flat` was collected by hand, one command per condition, and the
commands were never written down.** Ten of the twelve are recoverable from the condition names --
`speed_c7.1` is `--cycles 7.1`, `turn_s0.29` is `--spin 0.29` -- and the two sideways levels per
direction are recoverable from nothing at all. The base sideways recipe survives in FINDINGS F71,
but which two magnitudes became `lvl0` and `lvl1` does not, and the achieved lateral speeds are not
symmetric between left and right (0.071 / 0.185 against -0.118 / -0.186), so they were tuned
per direction rather than scaled from one number.

That makes the principal Stage 2 dataset unreproducible, which is a defect independent of any new
body. This file is the fix: the recipe is data, and `--verify` re-collects it on the body the
original came from and compares the achieved Froude, yaw and lateral speed against the stored clips.
**`--verify` asks whether the recipe reproduces the recorded *values*, which is the stricter of two
possible standards and not always the right one.** Reproducing the values matters when clips from
two bodies will be compared against each other. It does not matter when a body's clips are only
ever compared with its own -- a planner choosing among one robot's behaviours to reproduce that
robot's own demonstration never crosses bodies, and the achieved Froude of a shorter-legged insect
being 0.11 where the original was 0.126 changes nothing about it.

The standard that matters there is **separability**: are the twelve conditions further apart than
their own spread across clips. `--separability` measures that instead, and the recipe in this file
passes it -- 62 of 66 pairs above 2x, the four sideways conditions 9.3x to 15.6x apart -- while
failing `--verify` on `speed_c8.15` and both left strafes.

**The commands are not portable across bodies and are not meant to be.** `--cycles` is a temporal
frequency on a foot path scaled about each body's hip, so the same number gives a different Froude
on different segment lengths -- task-space quantities scale with leg length and joint-space ones do
not. Matching *achieved* behaviour across bodies is a separate re-derivation; what this reproduces
is the twelve **distinguishable** conditions, which is what a planner needs to choose between.

  .venv/bin/python3 scripts/dataset/collect_beh12.py --verify
  .venv/bin/python3 scripts/dataset/collect_beh12.py --morph c08f09t09=medauroidea_c08f09t09.ttt \\
      --out data/beh12_c08f09t09
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from wm.data.embodiment import body_velocity, yaw_rate  # noqa: E402
from wm.policy.planner import condition_of  # noqa: E402

# Shared by every condition. `--scale 0.65` and `--legtune` are the settings the recorded clips
# carry; `--cam_dx -0.6` is the framing fix without which 56-70% of frames clip the right edge.
COMMON = ["--gait", "cpg", "--scale", "0.65", "--cam_dx", "-0.6", "--behavior", "walk"]

# **`--episodes` is an index into the expert recording, 0-999, and it is not the episode number
# the clips carry.** `merge_behaviour_dirs` overwrites `expert_episode` with its own
# `axis*1000 + level*100 + clip`, so the value actually used at collection time is not recoverable
# from `data/beh12_c10f10t10_flat` -- reading 1000 and 2300 back as expert indices is out of range and
# is what made the first verification run die on the fifth condition.
#
# One expert episode for all twelve. In `--gait cpg` the behaviour comes from the oscillator flags;
# the expert path only supplies the foot geometry the IK targets, so holding it fixed removes a
# variable rather than losing one.
SPEED = [("speed_c5.8", ["--cycles", "5.8"]),
         ("speed_c7.1", ["--cycles", "7.1"]),
         ("speed_c8.15", ["--cycles", "8.15"]),
         ("speed_c8.8", ["--cycles", "8.8"])]

TURN_LEVELS = (0.05, 0.15, 0.29, 0.56)


def turn_conditions(sign=1.0):
    """`--spin` per level, signed. The name keeps the magnitude; the sign is a collection choice."""
    return [(f"turn_s{v:.2f}".rstrip("0").rstrip("."), ["--spin", f"{sign * v:g}"])
            for v in TURN_LEVELS]


TURN = turn_conditions()

# F71's sideways gait: fore-aft amplitude zero, feet half a cycle out of phase, and a `--spin`
# that cancels the yaw the strafe induces -- different per direction, which is why left and right
# do not mirror. **The two levels are a reconstruction**: the base is F71's `--strafe 0.8` and the
# lower level is set to reproduce the recorded lateral speeds. `--verify` is what checks it.
SIDE_BASE = ["--amps", "0.00", "0.20", "0.30", "--ft_phase", "0.5", "--symmetric",
             "--spin_amp", "0.25", "--ik_iters", "8"]
# **`lvl0`'s magnitude is per body, and 0.4 is the base body's value.** On `c08f09t09` the same
# 0.4 produces a robot that barely moves -- +0.017 lateral against `lvl1`'s -0.131, with the sign
# of the residue rather than of a strafe (F106). The recipe's own rule is that commands do not port
# across geometries; this makes the one number that failed adjustable instead of baked in.
LVL0_STRAFE = 0.4


def side_conditions(lvl0=LVL0_STRAFE):
    return [("side_L_lvl0", SIDE_BASE + ["--strafe", f"{-lvl0}", "--spin", "0.19"]),
            ("side_L_lvl1", SIDE_BASE + ["--strafe", "-0.8", "--spin", "0.19"]),
            ("side_R_lvl0", SIDE_BASE + ["--strafe", f"{lvl0}", "--spin", "-0.24"]),
            ("side_R_lvl1", SIDE_BASE + ["--strafe", "0.8", "--spin", "-0.24"])]


SIDE = side_conditions()

CONDITIONS = SPEED + TURN + SIDE

# What `data/beh12_c10f10t10_flat` achieved, measured from the clips. `--verify` reproduces these.
REFERENCE = {"speed_c5.8": (0.126, 0.007, 0.002), "speed_c7.1": (0.151, -0.025, -0.003),
             "speed_c8.15": (0.174, 0.010, 0.003), "speed_c8.8": (0.205, -0.047, 0.001),
             "turn_s0.05": (0.137, -0.003, 0.003), "turn_s0.15": (0.135, 0.001, 0.014),
             "turn_s0.29": (0.141, 0.019, 0.036), "turn_s0.56": (0.128, 0.046, 0.077),
             "side_L_lvl0": (0.015, 0.071, -0.000), "side_L_lvl1": (0.028, 0.185, -0.002),
             "side_R_lvl0": (0.012, -0.118, 0.000), "side_R_lvl1": (0.020, -0.186, -0.000)}


def achieved(directory, condition=None):
    """Median forward, lateral and yaw over every clip of a condition, dimensionless."""
    import glob
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        if condition is not None and condition_of(path) != condition:
            continue
        with np.load(path, allow_pickle=True) as d:
            head, q = d["head"].astype("float64"), d["body_quat"].astype("float64")
        h = float(np.median(head[:, 2]))
        v = body_velocity(head, q, 0.05, "hexapod")
        w = np.asarray(yaw_rate(q, 0.05, "hexapod", h)).ravel()
        rows.append((np.median(v[:, 0]), np.median(v[:, 1]), np.median(w)))
    return tuple(np.mean(rows, axis=0)) if rows else None


def run_condition(name, expert, flags, morph, scene, out_root, port, dry, repeats=4, extra=()):
    out = os.path.join(out_root, name)
    cmd = ([sys.executable, os.path.join(ROOT, "sim", "collect", "collect_ik.py"),
            "--port", str(port), "--morphs", f"{morph}={scene}", "--repeats", str(repeats),
            "--episodes", str(expert), "--out", out] + COMMON + flags + list(extra))
    # print what will actually run, `--extra` included: a dry run that hides the override is
    # worse than no dry run, because it reads as confirmation of the wrong command
    print(f"  {name:<14} {' '.join(flags + list(extra))}", flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True, cwd=ROOT)


def separability(root):
    """Are the conditions further apart than their own spread? Prints the closest pairs.

    A planner cannot resolve two conditions the *robot* does not resolve, so this bounds what any
    representation could do. Measured on `data/beh12_c10f10t10_flat` it also explains nothing about
    F91's failure -- there the turn levels are 2.7x to 6.8x apart and the speed levels 1.7x, and
    the planner resolves speed 9/9 and turn 2/9. It succeeds on the closest axis.
    """
    import itertools
    conds = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    flat = not conds
    mean, spread = {}, {}
    groups = {}
    if flat:
        for path in sorted(glob_npz(root)):
            groups.setdefault(condition_of(path), []).append(path)
    else:
        for c in conds:
            groups[c] = sorted(glob_npz(os.path.join(root, c)))
    for c, paths in groups.items():
        rows = np.array([_channels(p) for p in paths])
        mean[c], spread[c] = rows.mean(0), rows.std(0)

    print(f"{'condition':<14}{'forward':>9}{'lateral':>9}{'yaw':>9}{'clips':>7}")
    for c in sorted(mean):
        f, l, w = mean[c]
        print(f"{c:<14}{f:>9.3f}{l:>9.3f}{w:>9.3f}{len(groups[c]):>7}")

    # **Separable is not the same as correct, and separability alone passed a broken body.** On
    # `c08f09t09` both `side_*_lvl0` conditions came out with the wrong sign -- `side_R_lvl0` at
    # +0.017 lateral, motionless in all three channels -- because the strafe recipe under-drives
    # shorter legs. Every pair was still 2x apart, so this function reported the set as fine, and
    # the body went on to carry the project's headline closed-loop result (F106). A condition that
    # barely moves is trivially separable from one that moves a lot; what has to be asked instead
    # is whether each condition does **what its name says**.
    print()
    bad = []
    for c, (f, l, w) in mean.items():
        if c.startswith("side_L") and l <= 0:
            bad.append(f"{c}: lateral {l:+.3f}, should travel left (positive)")
        if c.startswith("side_R") and l >= 0:
            bad.append(f"{c}: lateral {l:+.3f}, should travel right (negative)")
    # **Turning is checked for sign, not only for size, and that gap cost a week.** The four turn
    # levels of one body must all rotate the same way, and they must rotate the way the reference
    # body does -- `--turn_sign`. Checking `|yaw|` alone is what let `c10f10t10` turn one way and
    # `c08f09t09` the other through four bodies and two robots (F75, F115, F117): the calibration
    # tables all reported magnitudes and agreed to within 3%.
    turns = {c: w for c, (f, l, w) in mean.items() if c.startswith("turn")}
    if turns:
        signs = {np.sign(w) for w in turns.values() if abs(w) > 1e-3}
        if len(signs) > 1:
            bad.append("turn levels disagree on direction: "
                       + ", ".join(f"{c} {w:+.4f}" for c, w in sorted(turns.items())))
        elif args.turn_sign and signs and args.turn_sign not in signs:
            bad.append(f"turns rotate {'positive' if 1 in signs else 'negative'}, "
                       f"--turn_sign asked for {'positive' if args.turn_sign > 0 else 'negative'}; "
                       "a body whose turns oppose the reference is not collecting the same behaviour")
    for fam, ch in (("side_L", 1), ("side_R", 1), ("turn", 2), ("speed", 0)):
        levels = sorted((c for c in mean if c.startswith(fam)), key=str)
        for a, b in zip(levels, levels[1:]):
            if abs(mean[b][ch]) <= abs(mean[a][ch]):
                bad.append(f"{b} is weaker than {a} on its own channel "
                           f"({abs(mean[b][ch]):.3f} <= {abs(mean[a][ch]):.3f})")
    if bad:
        print("**FAILS the semantic check** -- separable, and not what the names claim:")
        for line in bad:
            print(f"  {line}")
        print("\nRe-derive these for this body. The commands are not portable across geometries;")
        print("what strafes gently on the base body may not move a shorter-legged one at all.")
    else:
        print("semantic check passed: every condition moves the way its name says, and each level")
        print("exceeds the one below it on its own channel.")

    pairs = []
    for a, b in itertools.combinations(sorted(mean), 2):
        sep = np.linalg.norm(mean[a] - mean[b])
        noise = np.linalg.norm(spread[a]) + np.linalg.norm(spread[b]) + 1e-9
        pairs.append((sep / noise, a, b))
    pairs.sort()
    print(f"\nclosest pairs, separation in units of their combined spread:")
    for r, a, b in pairs[:6]:
        print(f"  {a:<14}{b:<14}{r:>7.1f}x")
    close = sum(1 for r, _, _ in pairs if r < 2)
    print(f"\n{close} of {len(pairs)} pairs closer than 2x their own spread")


def glob_npz(directory):
    import glob as _glob
    return _glob.glob(os.path.join(directory, "*.npz"))


def _channels(path):
    with np.load(path, allow_pickle=True) as z:
        head, q = z["head"].astype("float64"), z["body_quat"].astype("float64")
    h = float(np.median(head[:, 2]))
    v = body_velocity(head, q, 0.05, "hexapod")
    w = np.asarray(yaw_rate(q, 0.05, "hexapod", h)).ravel()
    return np.array([np.median(v[:, 0]), np.median(v[:, 1]), np.median(w)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", default="c08f09t09=medauroidea_c08f09t09.ttt", metavar="NAME=SCENE")
    ap.add_argument("--out", default="data/beh12_c08f09t09_raw",
                    help="one subdirectory per condition; flatten afterwards with "
                         "scripts/dataset/merge_behaviour_dirs.py")
    ap.add_argument("--only", nargs="*", default=[], help="condition names, for a partial re-run")
    ap.add_argument("--repeats", type=int, default=4,
                    help="clips per condition. Drop to 1 while sweeping a recipe -- but F71 "
                         "measured the sideways amplitude as a *narrow* optimum whose neighbours "
                         "scatter by a factor of ten across identical runs, so a value chosen on "
                         "one clip has to be confirmed on four.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="passed through to collect_ik after everything else, so a sweep is a "
                         "shell loop rather than an edit to this file. Must come last.")
    ap.add_argument("--expert", type=int, default=0,
                    help="index into the 1000 expert episodes, 0-999. Same value for every "
                         "condition: the behaviour comes from the oscillator, not from this.")
    ap.add_argument("--lvl0_strafe", type=float, default=LVL0_STRAFE,
                    help="strafe magnitude for the two `lvl0` lateral conditions. 0.4 is the base "
                         "body's value; a shorter-legged body needs more to move at all")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--dry_run", action="store_true", help="print the commands and collect nothing")
    ap.add_argument("--verify", action="store_true",
                    help="re-collect on c10f10t10 and compare against data/beh12_c10f10t10_flat. **Run "
                         "this first**: two of the twelve recipes are reconstructed, and a wrong "
                         "one produces a dataset that differs from the original without saying so.")
    ap.add_argument("--verify_out", default="data/beh12_verify_raw")
    ap.add_argument("--spin_sign", type=float, default=1.0,
                    help="multiply every turn level's --spin by this. **The same positive spin "
                         "rotates c10f10t10 one way and c08f09t09 the other** (F117), so a body "
                         "whose turns must match a reference collects them with -1 here. Verify "
                         "with --separability --turn_sign rather than assuming the flag flips "
                         "the motion: the two bodies already disagree under an identical command.")
    ap.add_argument("--turn_sign", type=float, default=0.0,
                    help="the yaw sign this body's turns must have, to match the body the "
                         "goals come from. 0 disables the check. Two hexapod bodies running "
                         "the same --spin turned opposite ways and nothing noticed for a "
                         "week, because every table reported |yaw| (F117).")
    ap.add_argument("--separability", default="",
                    help="measure a collected directory instead of collecting: how far apart the "
                         "conditions sit in body-motion space, in units of their own spread. This "
                         "is the standard a single-body planning test needs; --verify is the "
                         "stricter one, for when two bodies will be compared.")
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="relative agreement required on the condition's dominant channel")
    args = ap.parse_args()
    if args.lvl0_strafe != LVL0_STRAFE or args.spin_sign != 1.0:
        global CONDITIONS, SIDE, TURN
        SIDE = side_conditions(args.lvl0_strafe)
        TURN = turn_conditions(args.spin_sign)
        CONDITIONS = SPEED + TURN + SIDE
        if args.lvl0_strafe != LVL0_STRAFE:
            print(f"lvl0 strafe {args.lvl0_strafe} (default {LVL0_STRAFE})")
        if args.spin_sign != 1.0:
            print(f"spin sign {args.spin_sign:+g}: " +
                  " ".join(f"{n}={c[1]}" for n, c in TURN))

    if args.separability:
        separability(os.path.join(ROOT, args.separability))
        return

    if args.verify:
        morph, scene = "c10f10t10", "medauroidea_c10f10t10.ttt"
        out_root = os.path.join(ROOT, args.verify_out)
    else:
        morph, scene = args.morph.split("=", 1)
        out_root = os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    todo = [c for c in CONDITIONS if not args.only or c[0] in args.only]
    print(f"{morph} <- {scene}   {len(todo)} conditions, expert episode {args.expert} "
          f"-> {os.path.relpath(out_root, ROOT)}\n")
    for name, flags in todo:
        run_condition(name, args.expert, flags, morph, scene, out_root, args.port, args.dry_run,
                      args.repeats, args.extra)

    if args.dry_run:
        return
    if not args.verify:
        print(f"\nnow flatten:\n  .venv/bin/python3 scripts/dataset/merge_behaviour_dirs.py "
              f"--src {os.path.relpath(out_root, ROOT)} --out data/beh12_{morph}_flat "
              f"--embodiment hexapod")
        return

    print(f"\n{'condition':<14}{'channel':>9}{'recorded':>10}{'re-run':>10}{'agree':>8}")
    bad = []
    for name, _ in todo:
        got = achieved(os.path.join(out_root, name))
        if got is None:
            bad.append((name, "no clips")); continue
        want = REFERENCE[name]
        k = int(np.argmax(np.abs(want)))
        label = ("forward", "lateral", "yaw")[k]
        rel = abs(got[k] - want[k]) / max(abs(want[k]), 1e-6)
        ok = rel < args.tolerance
        print(f"{name:<14}{label:>9}{want[k]:>10.3f}{got[k]:>10.3f}{rel:>7.1%}{'' if ok else '  X'}")
        if not ok:
            bad.append((name, f"{rel:.1%} off"))
    if bad:
        print("\nrecipe does NOT reproduce the recorded set:")
        for name, why in bad:
            print(f"  {name:<14}{why}")
        raise SystemExit(1)
    print("\nevery condition reproduces within tolerance; the recipe in this file is the one used.")


if __name__ == "__main__":
    main()
