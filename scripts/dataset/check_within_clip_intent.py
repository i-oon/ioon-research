"""Did the robot actually change what it was doing **inside** the clip?

    .venv/bin/python3 scripts/dataset/check_within_clip_intent.py \\
        --data data/allocentric/beh12_c10f10t10_intent_flat --clean data/allocentric/beh12_c10f10t10_sweepn00_flat

**A required gate, and it exists because everything else missed the failure.** F165's first
collection passed `--schedule` to a `--gait cpg` run, which discards it: eight of twelve conditions
had no within-clip speed change at all. The command lines showed the flag, the log echoed it, the
`walk_check` verdicts printed, the separability gate passed and the R2 tables came out clean.
**Nothing in that chain looks at whether the intervention reached the robot.** This does.

**The statistic is the sd of the *smoothed* channel, not the raw one.** A gait oscillates the body
at stride frequency whether or not anything is scheduled, and that oscillation is most of the raw
variance -- the void run's `speed_*` conditions read 0.036-0.038 against clean clips at 0.033-0.047
and looked unremarkable in both directions. Averaging over one stride removes the gait and leaves
the **envelope**: what the robot was asked to do differently, and when. A clean steady clip has
almost none.

**Each family is checked on the channel it means**: speed on forward, turn on yaw, sideways on
lateral. **The bar is the clean arm's own maximum on that channel**, so the comparison is against
the most variable steady clip rather than an invented threshold.
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import REGISTRY, load  # noqa: E402

CHANNEL = {"speed": (0, "forward"), "turn": (2, "yaw"), "side": (1, "lateral")}
FAMILY = lambda c: "side" if c.startswith("side") else c.split("_")[0]


def envelope_sd(path, ch, window, trim):
    """**The ends are trimmed, and that is not a loosened threshold -- it is a removed confound.**

    Every clip in this project accelerates from rest at frame 0 and slows at the end, which
    `schedule_path`'s own docstring names as the reason a probe can read *when* the robot is
    stationary off the body's position. That transient is present in the steady arm too, it was
    never commanded, and it is most of the steady arm's envelope: `speed_c7.1` runs 0.081 -> 0.184
    -> 0.131 while doing nothing at all. Leaving it in sets the bar for "an intentional change" at
    the size of an unintentional one, and it is removed from **both** arms identically.

    Decided from the traces, not from the result: a stop mid-clip reads 0.198 -> 0.069 -> 0.194
    against a flat 0.16-0.18, so the event is unmistakable once the ends are gone.
    """
    clip = load(path, REGISTRY["hexapod"])
    x = np.asarray(clip["body_motion"], dtype=float)[:, ch]
    cut = int(round(trim * len(x)))
    if cut and len(x) - 2 * cut > window:
        x = x[cut:len(x) - cut]
    if len(x) <= window:
        return float(np.std(x))
    k = np.ones(window) / window
    return float(np.std(np.convolve(x, k, mode="valid")))


def collect(root, window, trim):
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*.npz"))):
        with np.load(path, allow_pickle=True) as raw:
            cond = str(raw["condition"] if "condition" in raw.files else raw["behavior"])
        fam = FAMILY(cond)
        ch, _ = CHANNEL[fam]
        out[cond].append(envelope_sd(path, ch, window, trim))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the intentful set under test")
    ap.add_argument("--clean", required=True, help="a steady set collected the same way")
    ap.add_argument("--window", type=int, default=9,
                    help="smoothing window in frames, about one stride at cycles 7.1 over 66")
    ap.add_argument("--trim", type=float, default=0.15,
                    help="fraction cut from each end before measuring. **Removes the start-and-stop "
                         "transient every clip has and nobody commanded**, which otherwise sets the "
                         "bar for an intended change at the size of an unintended one")
    ap.add_argument("--factor", type=float, default=1.5,
                    help="how far above the clean arm's maximum a condition must sit")
    args = ap.parse_args()

    test = collect(os.path.join(ROOT, args.data), args.window, args.trim)
    clean = collect(os.path.join(ROOT, args.clean), args.window, args.trim)

    bar = {}
    for cond, vals in clean.items():
        fam = FAMILY(cond)
        bar[fam] = max(bar.get(fam, 0.0), float(np.mean(vals)))

    print(f"within-clip envelope, sd of the channel smoothed over {args.window} frames, "
          f"ends trimmed {args.trim:.0%}")
    print(f"  test  {args.data}\n  clean {args.clean}\n")
    print(f"  {'condition':<20}{'channel':>9}{'envelope sd':>13}{'clean max':>11}"
          f"{'ratio':>8}   verdict")
    failed = []
    for cond in sorted(test):
        fam = FAMILY(cond)
        ch, name = CHANNEL[fam]
        v = float(np.mean(test[cond]))
        b = bar.get(fam, 0.0)
        ratio = v / max(b, 1e-9)
        ok = ratio >= args.factor
        if not ok:
            failed.append(cond)
        print(f"  {cond:<20}{name:>9}{v:>13.4f}{b:>11.4f}{ratio:>8.2f}   "
              f"{'ok' if ok else '**NO WITHIN-CLIP INTENT**'}")

    print()
    for fam, b in sorted(bar.items()):
        print(f"  clean {fam:<6} maximum envelope sd {b:.4f}")

    if failed:
        print(f"\n**GATE FAILED** on {len(failed)} of {len(test)} conditions: "
              + ", ".join(failed))
        print("The intervention did not reach the robot in these. **Do not run the measurement** "
              "-- a set that\nwas never perturbed cannot answer whether perturbation helps, and "
              "every other gate will pass it.")
        raise SystemExit(1)
    print(f"\ngate passed: all {len(test)} conditions change within the clip, "
          f"by at least {args.factor}x the steadiest arm's own maximum")


if __name__ == "__main__":
    main()
