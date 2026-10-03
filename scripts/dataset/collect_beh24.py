"""Collect the 24 beh24 behaviour conditions for one hexapod body, from the recipe kept in this file.

The 24 conditions: forward at 4 speeds (`speed_cX`), backward at 4 (`speed_cX_bwd`), left turns at 4 levels
(`turn_sX`), right turns at 4 (`turn_sX_neg`), sideways left and right at 4 levels each (`side_L/R_lvlN`).

**The recipe below reproduces the canonical c10f10t10 clips** (`data/egocentric/beh24_c10f10t10_ego_flat`):
every value was fitted back from the clips' stored joint commands (residual 0.0000 rad, F296 / F298), because
the clips were assembled from several hand-run collections (`build_beh24_hex_ego_flat.py`, F223) whose commands
were never logged, and an earlier version of this table did not match them. Conventions that bit before:

- **Negative `--spin` yaws the hexapod LEFT (CCW).** `turn_sX` (left) runs `--spin -X`, `turn_sX_neg` (right)
  runs `--spin +X` (F66, F298; verified by heading and by ego image motion).
- Backward uses its own cycles (3.49 / 3.90 / 4.97 / 5.05) so its displacement mirrors the forward ladder.
- Sideways strafes were retuned per side to lateral targets |0.03 / 0.08 / 0.13 / 0.14|, so L and R differ.

**Commands are not portable across bodies.** `--cycles` and `--strafe` act on foot paths scaled about each
body's hip, so the same numbers give different Froude on another leg geometry (F96, F108). For another body,
re-tune (`--lvl0_strafe`, `--spin_sign`) and check with `--separability` / `--turn_sign` rather than assuming.

`--verify` no longer re-collects: the reference set it compared against (`data/allocentric/beh12_c10f10t10_flat`,
12 conditions, older label code) is archived (`data/_archive_old_datasets/`), and the current hexapod data
(`data/counterfactual_walks/c10_*`) are made by `collect_c10_walks_and_branches.py` from the plan table
(`collect_switch_hex.COND`), not by this file. It prints the checks of the current data instead:
`scripts/dataset/check_branches.py` (+ `--c08`), `scripts/diagnostics/dataset/check_splits.py`,
`tests/test_froude_labels.py`, `collect_c10_walks_and_branches.py targets` (doc/DATA.md);
`--separability` checks that the conditions are further apart than their own spread across clips.

  .venv/bin/python3 scripts/dataset/collect_beh24.py --dry_run
  .venv/bin/python3 scripts/dataset/collect_beh24.py --verify
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
# from `data/allocentric/beh12_c10f10t10_flat` -- reading 1000 and 2300 back as expert indices is out of range and
# is what made the first verification run die on the fifth condition.
#
# One expert episode for all conditions. In `--gait cpg` the behaviour comes from the oscillator flags;
# the expert path only supplies the foot geometry the IK targets, so holding it fixed removes a
# variable rather than losing one.
SPEED = [("speed_c5.8", ["--cycles", "5.8"]),
         ("speed_c7.1", ["--cycles", "7.1"]),
         ("speed_c8.15", ["--cycles", "8.15"]),
         ("speed_c8.8", ["--cycles", "8.8"])]

# **New.** `--lead` sets stride direction (`collect_ik.py`: "0.25 and 0.75 walk opposite ways");
# `COMMON` never sets it, so every condition above defaults to 0.25 (forward). Same 4 `--cycles`
# magnitudes, `--lead 0.75` added, nothing else changed -- the mirror of SPEED, not a new tuning.
# **Recalibrated, fitted from the stored c10f10t10 clips (F298).** The same cycles as SPEED walked backward
# a flat -0.78..-0.82 m per clip at every level (no ladder); the canonical clips ran these cycles, whose
# displacement mirrors the forward ladder (-0.586 / -0.672 / -0.826 / -0.827 m vs +0.585 ... +0.819).
BWD_CYCLES = {"speed_c5.8": 3.49, "speed_c7.1": 3.90, "speed_c8.15": 4.97, "speed_c8.8": 5.05}
SPEED_BWD = [(f"{name}_bwd", ["--cycles", f"{BWD_CYCLES[name]:g}", "--lead", "0.75"]) for name, _ in SPEED]

TURN_LEVELS = (0.05, 0.15, 0.29, 0.56)


def turn_conditions(sign=1.0, suffix=""):
    """`--spin` per level. **Negative `--spin` yaws the hexapod LEFT (CCW, + heading)** -- F66, F298.
    The name keeps the magnitude; `turn_sX` (sign=+1) turns left and therefore runs `--spin -X`;
    `turn_sX_neg` (sign=-1) turns right and runs `--spin +X`. Verified on the stored c10f10t10 clips
    by heading and by ego image motion (F298)."""
    return [(f"turn_s{v:.2f}".rstrip("0").rstrip(".") + suffix, ["--spin", f"{-sign * v:g}"])
            for v in TURN_LEVELS]


TURN = turn_conditions()
# **New.** The opposite-direction mirror -- `turn_conditions(sign=-1.0)` already existed as a
# mechanism (used for per-body sign matching, `--spin_sign`) but was never added to the base
# `CONDITIONS` set itself. `_neg` suffix keeps the original 4 names and their data untouched.
TURN_NEG = turn_conditions(sign=-1.0, suffix="_neg")

# F62's sideways gait: fore-aft amplitude zero, feet half a cycle out of phase, and a `--spin`
# that cancels the yaw the strafe induces -- different per direction, which is why left and right
# do not mirror. **The two levels are a reconstruction**: the base is F62's `--strafe 0.8` and the
# lower level is set to reproduce the recorded lateral speeds. `--verify` is what checks it.
SIDE_BASE = ["--amps", "0.00", "0.20", "0.30", "--ft_phase", "0.5", "--symmetric",
             "--spin_amp", "0.25", "--ik_iters", "8"]
# **`lvl0`'s magnitude is per body, and 0.4 is the base body's value.** On `c08f09t09` the same
# 0.4 produces a robot that barely moves -- +0.017 lateral against `lvl1`'s -0.131, with the sign
# of the residue rather than of a strafe (F96). The recipe's own rule is that commands do not port
# across geometries; this makes the one number that failed adjustable instead of baked in.
LVL0_STRAFE = 0.48
# **Recalibrated, fitted from the stored c10f10t10 clips (F298):** all four levels per side were retuned
# together to lateral targets |0.03 / 0.08 / 0.13 / 0.14| (`recollect_b1_more.py`), so left and right
# differ. The original 0.4 / 0.8 / 1.2 / 1.6 ladder is not what the canonical clips ran.
SIDE_STRAFE = {"L": (-0.48, -0.83, -1.30, -1.40), "R": (0.488, 0.73, 0.83, 0.87)}


def side_conditions(lvl0=LVL0_STRAFE):
    """`lvl0` overrides only level 0 (per-body tuning, F96); default = the canonical values."""
    L, R = list(SIDE_STRAFE["L"]), list(SIDE_STRAFE["R"])
    if lvl0 != LVL0_STRAFE:
        L[0], R[0] = -lvl0, lvl0
    return [(f"side_L_lvl{i}", SIDE_BASE + ["--strafe", f"{L[i]:g}", "--spin", "0.19"]) for i in (0, 1)] + \
           [(f"side_R_lvl{i}", SIDE_BASE + ["--strafe", f"{R[i]:g}", "--spin", "-0.24"]) for i in (0, 1)]


SIDE = side_conditions()

# **New, unverified magnitudes.** lvl0->lvl1 steps by +0.4 strafe on both sides in the original
# recipe; lvl2/lvl3 extrapolate that same step (1.2, 1.6). Spin-cancellation is NOT scaled with
# strafe magnitude in the original recipe either (lvl0 and lvl1 both use the same spin value per
# side), so lvl2/lvl3 reuse the same per-side spin rather than guessing a new one. **This is a
# starting point for `--separability`/`--verify`, not a trusted number** -- flag raised in this
# file's own module docstring.
SIDE_EXTRA = [(f"side_{d}_lvl{i}", SIDE_BASE + ["--strafe", f"{SIDE_STRAFE[d][i]:g}",
                                                 "--spin", "0.19" if d == "L" else "-0.24"])
              for d in ("L", "R") for i in (2, 3)]

CONDITIONS = SPEED + SPEED_BWD + TURN + TURN_NEG + SIDE + SIDE_EXTRA

# What `data/allocentric/beh12_c10f10t10_flat` (now archived) achieved, measured from the clips (the old `--verify` target).
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


def separability(root, turn_sign=0.0):
    """Are the conditions further apart than their own spread? Prints the closest pairs.

    A planner cannot resolve two conditions the *robot* does not resolve, so this bounds what any
    representation could do. Measured on `data/allocentric/beh12_c10f10t10_flat` it also explains nothing about
    F81's failure -- there the turn levels are 2.7x to 6.8x apart and the speed levels 1.7x, and
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
    # the body went on to carry the project's headline closed-loop result (F96). A condition that
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
    # `c08f09t09` the other through four bodies and two robots (F66, F106, F108): the calibration
    # tables all reported magnitudes and agreed to within 3%.
    turns = {c: w for c, (f, l, w) in mean.items() if c.startswith("turn")}
    if turns:
        signs = {np.sign(w) for w in turns.values() if abs(w) > 1e-3}
        if len(signs) > 1:
            bad.append("turn levels disagree on direction: "
                       + ", ".join(f"{c} {w:+.4f}" for c, w in sorted(turns.items())))
        elif turn_sign and signs and turn_sign not in signs:
            bad.append(f"turns rotate {'positive' if 1 in signs else 'negative'}, "
                       f"--turn_sign asked for {'positive' if turn_sign > 0 else 'negative'}; "
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
    ap.add_argument("--out", default="data/allocentric/beh12_c08f09t09_raw",
                    help="one subdirectory per condition; flatten afterwards with "
                         "scripts/dataset/merge_behaviour_dirs.py")
    ap.add_argument("--only", nargs="*", default=[], help="condition names, for a partial re-run")
    ap.add_argument("--repeats", type=int, default=4,
                    help="clips per condition. Drop to 1 while sweeping a recipe -- but F62 "
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
                    help="strafe magnitude for the two `lvl0` lateral conditions. 0.48 is c10f10t10's "
                         "canonical value (F298); a shorter-legged body needs more to move at all")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--dry_run", action="store_true", help="print the commands and collect nothing")
    ap.add_argument("--verify", action="store_true",
                    help="print the checks of the current hexapod data (data/counterfactual_walks c10_*, made by "
                         "collect_c10_walks_and_branches.py) and exit; collects nothing")
    ap.add_argument("--spin_sign", type=float, default=1.0,
                    help="multiply every turn level's --spin by this. Default 1 = the canonical c10f10t10 "
                         "commands (turn_sX runs --spin -X, F298). **The same positive spin "
                         "rotates c10f10t10 one way and c08f09t09 the other** (F108), so a body "
                         "whose turns must match a reference collects them with -1 here. Verify "
                         "with --separability --turn_sign rather than assuming the flag flips "
                         "the motion: the two bodies already disagree under an identical command.")
    ap.add_argument("--turn_sign", type=float, default=0.0,
                    help="the yaw sign this body's turns must have, to match the body the "
                         "goals come from. 0 disables the check. Two hexapod bodies running "
                         "the same --spin turned opposite ways and nothing noticed for a "
                         "week, because every table reported |yaw| (F108).")
    ap.add_argument("--separability", default="",
                    help="measure a collected directory instead of collecting: how far apart the "
                         "conditions sit in body-motion space, in units of their own spread. This "
                         "is the standard a single-body planning test needs; --verify is the "
                         "stricter one, for when two bodies will be compared.")
    ap.add_argument("--tolerance", type=float, default=0.15,
                    help="relative agreement required on the condition's dominant channel")
    args = ap.parse_args()
    if args.lvl0_strafe != LVL0_STRAFE or args.spin_sign != 1.0:
        # **All 24 conditions stay.** This used to rebuild CONDITIONS as SPEED + TURN + SIDE, so
        # either flag silently dropped SPEED_BWD, TURN_NEG and SIDE_EXTRA (12 of 24) and left the
        # mirrored turns un-re-signed. TURN_NEG is the mirror of TURN, so it takes -spin_sign.
        global CONDITIONS, SIDE, TURN, TURN_NEG
        SIDE = side_conditions(args.lvl0_strafe)
        TURN = turn_conditions(args.spin_sign)
        TURN_NEG = turn_conditions(-args.spin_sign, suffix="_neg")
        CONDITIONS = SPEED + SPEED_BWD + TURN + TURN_NEG + SIDE + SIDE_EXTRA
        assert len(CONDITIONS) == 24, len(CONDITIONS)
        if args.lvl0_strafe != LVL0_STRAFE:
            print(f"lvl0 strafe {args.lvl0_strafe} (default {LVL0_STRAFE})")
        if args.spin_sign != 1.0:
            print(f"spin sign {args.spin_sign:+g}: " +
                  " ".join(f"{n}={c[1]}" for n, c in TURN + TURN_NEG))

    if args.separability:
        separability(os.path.join(ROOT, args.separability), args.turn_sign)
        return

    if args.verify:
        print("The current hexapod data (data/counterfactual_walks/c10_*) are made by "
              "scripts/dataset/collect_c10_walks_and_branches.py, not by this file; check them with:\n"
              "  .venv/bin/python3 scripts/dataset/check_branches.py [--c08]\n"
              "  .venv/bin/python3 scripts/diagnostics/dataset/check_splits.py\n"
              "  .venv/bin/python3 tests/test_froude_labels.py\n"
              "  .venv/bin/python3 scripts/dataset/collect_c10_walks_and_branches.py targets\n"
              "(doc/DATA.md). The old re-collection check against data/allocentric/beh12_c10f10t10_flat "
              "(REFERENCE) was removed with that dataset's archiving.")
        return
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
    print(f"\nnow flatten:\n  .venv/bin/python3 scripts/dataset/merge_behaviour_dirs.py "
          f"--src {os.path.relpath(out_root, ROOT)} --out data/allocentric/beh12_{morph}_flat "
          f"--embodiment hexapod")

if __name__ == "__main__":
    main()
