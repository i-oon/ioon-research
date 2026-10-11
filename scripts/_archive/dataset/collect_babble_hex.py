"""Motor babble for a hexapod body, with behaviour switches INSIDE each clip.

**Why.** Every beh12/beh24 clip runs one behaviour for its whole length, so the current frame always
tells what the next ones will do; the world model and its read-out learned that shortcut and cannot
read an action applied from a state it did not come from (F252, F266). Babble whose drives change
every 10-20 steps breaks it: from the same kind of state, different continuations follow.

**Coverage matches beh24** (`collect_beh24.py`): forward and backward walking over the speed range
(`--cycles` 5.3-8.8 equivalent, via the pace drive on a base of 8.8), turning both ways (spin up to
+-0.56), and sideways walking both ways (strafe 0.4-1.6 on F62's sideways gait, with that recipe's
per-direction yaw-cancelling spin). Every drive is ramped over 4 frames between segments.

One JSON plan per clip is written next to the output and passed to `collect_ik.py --plan`; the clip
stores the per-frame drives as `plan_*` arrays. `--pilot` collects a few clips and prints the achieved
Froude per segment so coverage and switching can be checked before a full run.

    .venv/bin/python3 scripts/dataset/collect_babble_hex.py --pilot 4 \\
        --morph c08f09t09=medauroidea_c08f09t09.ttt --out data/egocentric/babble_c08f09t09_pilot
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EP = 66
BASE_CYCLES = 8.8
WALK = dict(a0=0.25, a1=0.20, a2=0.20, ft_phase=0.0, strafe=0.0)          # collect_beh24 SPEED
SIDE = dict(a0=0.00, a1=0.20, a2=0.30, ft_phase=0.5, lead=0.25)            # collect_beh24 SIDE_BASE
# shared flags: beh24's COMMON plus the sideways recipe's pose/IK settings, which a clip that switches
# between walking and crabbing must hold constant throughout
COMMON = ["--gait", "cpg", "--scale", "0.65", "--cam_dx", "-0.6", "--behavior", "babble",
          "--cycles", str(BASE_CYCLES), "--symmetric", "--spin_amp", "0.25", "--ik_iters", "8",
          "--view", "egocentric"]


def segment(rng):
    """One segment's drives, drawn to cover beh24's behaviour range."""
    mode = rng.choice(["fwd", "bwd", "side"], p=[0.4, 0.25, 0.35])
    if mode == "side":
        left = rng.random() < 0.5
        d = dict(SIDE, pace=6.0 / BASE_CYCLES,                               # beh24 sideways: 6 cycles
                 strafe=(-1 if left else 1) * rng.uniform(0.8, 1.6),     # pilot: 0.4-1.6 reached only +-0.09
                 spin=0.19 if left else -0.24)
    else:
        turn = rng.random() < 0.7
        # +-0.8, not beh24's +-0.56: on c08 the pilot measured yaw ~ -0.1 x spin - 0.02 (sign flipped
        # for this body, drift -0.02), so +-0.56 cannot reach beh24's +-0.08; 0.8 is the gait's safe edge
        d = dict(WALK, lead=0.25 if mode == "fwd" else 0.75,
                 pace=rng.uniform(5.3, 8.8) / BASE_CYCLES,
                 spin=rng.uniform(-0.8, 0.8) if turn else rng.uniform(-0.05, 0.05))
    d["mode"] = str(mode)
    return d


def make_plan(rng, ramp=4, steady=False, seg=(15, 25)):
    """Per-frame drives for one clip: segments of seg[0]-seg[1] frames, linearly ramped between.
    `steady`: one segment for the whole clip, drawn from the same distribution -- the control that
    matches coverage and diversity without any switch. `seg=(5, 5)`: a new drive every model step
    at stride 5, so the frame never tells which drive comes next (random-action arm)."""
    keys = ("pace", "spin", "strafe", "lead", "a0", "a1", "a2", "ft_phase")
    segs, t = [], 0
    while t < EP:
        n = EP if steady else int(min(rng.integers(seg[0], seg[1] + 1), EP - t))  # pilot: 10-step segments barely took hold
        if 0 < EP - (t + n) < min(8, seg[0]):   # a tail shorter than 8 frames would be mostly ramp: absorb it
            n = EP - t
        segs.append((t, n, segment(rng)))
        t += n
    plan = {k: np.zeros(EP) for k in keys}
    for t0, n, d in segs:
        for k in keys:
            plan[k][t0:t0 + n] = d[k]
    for (t0, _, _) in segs[1:]:                                              # smooth each switch
        lo, hi = max(0, t0 - ramp // 2), min(EP, t0 + ramp // 2)
        for k in keys:
            a, b = plan[k][lo - 1] if lo > 0 else plan[k][lo], plan[k][hi] if hi < EP else plan[k][hi - 1]
            plan[k][lo:hi] = np.linspace(a, b, hi - lo + 2)[1:-1]
    return {k: v.tolist() for k, v in plan.items()}, [(t0, n, d["mode"]) for t0, n, d in segs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", required=True, help="NAME=SCENE, e.g. c08f09t09=medauroidea_c08f09t09.ttt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--pilot", type=int, default=0, help="collect this many clips and report per-segment Froude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steady", action="store_true", help="no switches: one drive per clip (control)")
    ap.add_argument("--seg", type=int, nargs=2, default=(15, 25), metavar=("MIN", "MAX"),
                    help="segment length range in frames (5 5: a new drive every stride-5 step)")
    ap.add_argument("--ramp", type=int, default=4, help="frames of linear ramp across each switch")
    ap.add_argument("--expert", type=int, default=0, help="expert index for the foot geometry (beh24: 0)")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--centre_from", required=True,
                    help="an existing clip of this body recorded with --symmetric --ik_iters 8 (a "
                         "sideways clip): its mean command is the CPG's centre pose (collect_ik.py)")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    n = args.pilot or args.clips
    for i in range(n):
        rng = np.random.default_rng((args.seed, i))
        plan, segs = make_plan(rng, ramp=args.ramp, steady=args.steady, seg=tuple(args.seg))
        pf = os.path.join(out, f"plan_{i:03d}.json")
        with open(pf, "w") as fh:
            json.dump(plan, fh)
        clip_out = os.path.join(out, f"clip{i:03d}")
        cmd = [sys.executable, os.path.join(ROOT, "sim", "collect", "collect_ik.py"),
               "--port", str(args.port), "--morphs", args.morph, "--episodes", str(args.expert),
               "--ego_seed", str(args.seed * 1000 + i), "--plan", pf, "--out", clip_out, "--centre_from", os.path.join(ROOT, args.centre_from)] + COMMON
        print(f"clip {i}: " + " ".join(f"{m}[{t0}:{t0 + k}]" for t0, k, m in segs), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        if args.pilot:
            report(clip_out, segs)


def report(clip_dir, segs):
    """Achieved Froude (fwd, lat, yaw) per segment, from the clip's own body pose."""
    import glob
    from wm.data.embodiment import body_velocity, yaw_rate
    path = sorted(glob.glob(os.path.join(clip_dir, "*.npz")))[0]
    with np.load(path, allow_pickle=True) as d:
        head, q = d["head"].astype(float), d["body_quat"].astype(float)
    h = float(np.median(head[:, 2]))
    v = body_velocity(head, q, 0.05, "hexapod")        # already dimensionless (collect_beh24.achieved)
    w = np.asarray(yaw_rate(q, 0.05, "hexapod", h)).ravel()
    for t0, k, m in segs:
        a, b = t0 + 2, t0 + k - 2
        if b > a:
            print(f"    {m:<5} [{t0:2d}:{t0 + k:2d}]  fwd {np.median(v[a:b, 0]):+.3f}  "
                  f"lat {np.median(v[a:b, 1]):+.3f}  yaw {np.median(w[a:b]):+.3f}", flush=True)


if __name__ == "__main__":
    main()
