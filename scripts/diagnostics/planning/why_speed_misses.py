"""Is the candidate library too coarse to hit the commanded speed, or is the planner missing?

**Every closed loop in this project picks the right behaviour and runs it at the wrong rate.** On a
held-out hexapod body: behaviour 100%, speed within 15% on **33%** of runs. On the quadruped:
behaviour 2/3, speed **0/3**. Three explanations have not been separated, and this file settles the
first without running anything:

  1. **the library is too coarse** -- no recorded behaviour actually travels at the demonstrated
     speed, so no choice could have been right;
  2. the score, a distance between predicted and goal *frames*, does not constrain speed;
  3. the projector does not know the new body's scale.

Only (1) is answerable from files already on disk. If a candidate existed that would have matched
and the planner passed it over, (1) is refuted and the fault is in selection -- which is a
different piece of work from collecting more behaviours.

**What each candidate would have achieved is read from the candidate's own recording**, not
predicted: these are clips of the robot performing that behaviour, so their achieved speed is
measured fact.

    .venv/bin/python3 scripts/diagnostics/planning/why_speed_misses.py \\
        results/wm/closed_loop/b1_physics3/*.npz --demo_dir data/allocentric/beh12_b1_flat
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diagnostics.planning.score_closed_loop import channels  # noqa: E402


def track(path):
    with np.load(path, allow_pickle=True) as d:
        if "head" in d.files:
            head, quat = d["head"].astype("float64"), d["body_quat"].astype("float64")
        else:
            head, quat = d["base_pos"].astype("float64"), d["base_quat"].astype("float64")
        dt = float(d["dt"]) if "dt" in d.files else 0.05
        emb = str(d["embodiment"]) if "embodiment" in d.files else "hexapod"
        cond = str(d["condition"]) if "condition" in d.files else ""
        chosen = [str(c) for c in np.asarray(d["chosen"], str)] if "chosen" in d.files else []
    return channels(head, quat, dt, emb), cond, chosen


def dominant(c):
    """Which channel this clip is actually about: the one furthest from zero."""
    med = np.median(c, axis=0)
    return int(np.argmax(np.abs(med))), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--demo_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.15)
    args = ap.parse_args()

    # every recorded behaviour of this robot, and what it actually achieved
    lib = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.demo_dir, "*.npz"))):
        c, cond, _ = track(p)
        lib.setdefault(cond, []).append(np.median(c, axis=0))
    lib = {k: np.mean(v, axis=0) for k, v in lib.items()}

    names = ("forward", "lateral", "yaw")
    print(f"  {'demonstration':<16}{'channel':>9}{'demo':>9}{'loop':>9}{'err':>8}"
          f"{'best in library':>18}{'its err':>9}")
    verdicts = []
    for run in args.runs:
        c, cond, chosen = track(run)
        demo_c, _dcond, _ = track(os.path.join(ROOT, args.demo_dir, cond_file(run, args.demo_dir)))
        k, demo_med = dominant(demo_c)
        want = demo_med[k]
        got = float(np.median(c[:, k]))
        err = abs(got - want) / max(abs(want), 1e-6)

        # the best the library could have done on this channel, had the planner been perfect
        best, best_err = None, np.inf
        for name, med in lib.items():
            e = abs(med[k] - want) / max(abs(want), 1e-6)
            if e < best_err:
                best, best_err = name, e
        reachable = best_err <= args.tol
        verdicts.append((reachable, err <= args.tol))
        print(f"  {cond:<16}{names[k]:>9}{want:>9.3f}{got:>9.3f}{err:>8.0%}"
              f"{best:>18}{best_err:>9.0%}")

    n = len(verdicts)
    hit = sum(1 for _r, h in verdicts if h)
    reach = sum(1 for r, _h in verdicts if r)
    print(f"\n  the loop hit the speed on {hit}/{n}.")
    print(f"  a candidate that would have hit it existed on {reach}/{n}.")
    print()
    if reach > hit:
        print(f"**The library was not the limit on {reach - hit} of {n}.** A recorded behaviour")
        print("travelling at the demonstrated rate was in the candidate list and was not chosen,")
        print("so collecting more behaviours would not have helped those runs -- the selection did")
        print("not resolve speed. That points at the score or the projector, not at coverage.")
    elif reach == hit:
        print("**The library is the limit.** Wherever the loop missed, nothing in the candidate")
        print("list would have hit either, so no amount of better selection could have helped.")
        print("Denser behaviour coverage is the fix, and it is a collection problem.")


def cond_file(run, demo_dir):
    with np.load(run, allow_pickle=True) as d:
        return str(d["demo"])


if __name__ == "__main__":
    main()
