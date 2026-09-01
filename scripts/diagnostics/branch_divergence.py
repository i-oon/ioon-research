"""After a shared prefix, do two actions produce separable futures -- in position *and* heading?

    .venv/bin/python3 scripts/diagnostics/branch_divergence.py --embodiment hexapod \\
        --branch 33 --noise a.npz a_repeat.npz --pair forward=a.npz turn=b.npz

**This is the number that decides the counterfactual design, and the easy version does not.**
Comparing two behaviours from the same *spawn* gives 10-22x signal over noise on the insect, but
they diverge from frame 0 and share no momentum. The real design branches **mid-clip from a shared
pose**, so the two futures start with identical velocity and contact state and the action has to
overcome that first. **Expect a lower ratio here, and this is the one to believe.**

**Heading is reported beside position because position understates turning.** A robot that rotates
in place moves its head barely at all; the B1's turn counterfactual read 1.07 px of displacement at
h=30 while its quaternion had moved 0.138. Turning is also the behaviour F136 found weakest, so it
is the one most likely to fail and is broken out separately.

`--noise` takes two runs of the **same** commands, which is the floor everything is read against;
on CoppeliaSim that is not zero and on MuJoCo it is.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import heading  # noqa: E402

POS = {"hexapod": "head", "b1": "base_pos"}
QUAT = {"hexapod": "body_quat", "b1": "base_quat"}


def load(path, emb):
    with np.load(os.path.join(ROOT, path), allow_pickle=True) as z:
        return np.asarray(z[POS[emb]], float), np.asarray(z[QUAT[emb]], float)


def diverge(a, b, emb, branch, h):
    """`branch` is the index of the **first divergent** frame, so the last shared one is `branch-1`.

    **Referencing `branch` instead is an off-by-one that biases every heading number**, and it is
    visible rather than subtle: rendered during a bit-identical shared prefix it made two panels
    read +1.4 and +1.2 degrees where they must read the same. `h=1` is the first frame after the
    split.
    """
    (pa, qa), (pb, qb) = a, b
    ref = max(branch - 1, 0)
    t = ref + h
    if t >= min(len(pa), len(pb)):
        return None, None
    # **Displacement *since the branch*, not absolute separation** -- the same reference the heading
    # channel already used. Measuring position absolutely charges the counterfactual for an offset
    # that was already there at the split and is common to signal and noise alike: on the insect the
    # two identical-command runs sit 26 mm apart at the branch simply from gait-phase drift, and
    # that offset was being counted as though the actions had caused it. On a bit-identical prefix
    # (the B1) the two references coincide and this changes nothing.
    dp = float(np.linalg.norm((pa[t] - pa[ref]) - (pb[t] - pb[ref]))) * 1000.0
    ha, hb = heading(qa, emb), heading(qb, emb)
    # **relative to the last shared frame**, since an absolute heading off an aft-pointing axis is
    # not a quantity either robot agrees on -- only one robot's own heading differences are meaningful
    d = (ha[t] - ha[ref]) - (hb[t] - hb[ref])
    return dp, abs(float(np.degrees(np.arctan2(np.sin(d), np.cos(d)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", choices=("hexapod", "b1"), required=True)
    ap.add_argument("--branch", type=int, required=True)
    ap.add_argument("--noise", nargs=2, required=True, metavar=("RUN", "REPEAT"),
                    help="two runs of the SAME commands -- the floor")
    ap.add_argument("--pair", nargs="+", required=True, metavar="NAME=CLIP",
                    help="the branch arms, e.g. forward=a.npz turn=b.npz side=c.npz; the first is "
                         "the reference every other is compared against")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10, 15])
    ap.add_argument("--floor", type=float, default=3.0,
                    help="pre-registered pass mark: signal must exceed noise by this factor at "
                         "every horizon we would train on, **including turning on heading**")
    args = ap.parse_args()

    emb = args.embodiment
    nz = [load(p, emb) for p in args.noise]
    arms = [(s.split("=", 1)[0], load(s.split("=", 1)[1], emb)) for s in args.pair]
    ref_name, ref = arms[0]

    print(f"{emb}, branch at frame {args.branch}, reference arm '{ref_name}'\n")
    print(f"  {'arm':>10}{'h':>5}{'position mm':>14}{'noise mm':>10}{'x':>7}"
          f"{'heading deg':>14}{'noise deg':>11}{'x':>7}   verdict")
    failed = []
    for name, arm in arms[1:]:
        for h in args.horizons:
            sp, sh = diverge(ref, arm, emb, args.branch, h)
            np_, nh = diverge(nz[0], nz[1], emb, args.branch, h)
            if sp is None or np_ is None:
                continue
            # **An exactly zero floor is a result, not a divide-by-zero.** MuJoCo restores its
            # state bit-identically, so the ratio is unbounded and printing a huge number instead
            # of saying so reads as a formatting bug and hides the strongest fact in the table.
            exact = np_ == 0.0 and nh == 0.0
            rp = float("inf") if np_ == 0.0 else sp / np_
            rh = float("inf") if nh == 0.0 else sh / nh
            ok = rp >= args.floor and rh >= args.floor
            if not ok:
                failed.append((name, h, round(rp, 1), round(rh, 1)))
            fmt = lambda r: "  exact" if r == float("inf") else f"{r:>6.1f}x"
            print(f"  {name:>10}{h:>5}{sp:>14.2f}{np_:>10.4f}{fmt(rp)}"
                  f"{sh:>14.3f}{nh:>11.4f}{fmt(rh)}   "
                  f"{'ok' if ok else '**BELOW ' + str(args.floor) + 'x**'}"
                  + ("" if not exact else ""))

    print()
    if failed:
        print(f"**{len(failed)} cells below {args.floor}x**: "
              + ", ".join(f"{n} h={h} (pos {a}x, head {b}x)" for n, h, a, b in failed))
        print("**A cell that fails on heading and passes on position is still a failure** -- it "
              "means the two\nfutures end up in the same place facing different ways, which is "
              "exactly what turning is.")
    else:
        print(f"every arm clears {args.floor}x on position and heading at every horizon")
        floors = [diverge(nz[0], nz[1], emb, args.branch, h)[0] for h in args.horizons]
        if all(v == 0.0 for v in floors if v is not None):
            print("noise floor is exactly zero at every horizon: the reset is bit-identical, so\n"
                  "every divergence above is the action and nothing else")


if __name__ == "__main__":
    main()
