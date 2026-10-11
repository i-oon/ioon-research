"""Direct action selection over a candidate library, graded per timestep against goal_t -- and, for the
same library, the best that ANY pick could have achieved (the oracle) and what a random pick gets.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/babble_library_eval.py \\
        --ckpt <ckpt with a fitted `projector` key> --candidates_dir data/egocentric/<library> \\
        --goal_dir data/egocentric/beh12_c10f10t10_ego_flat

Per goal condition (one hexapod goal clip each), at every horizon-step t: the planner picks the candidate
whose `body_head(proj(a))` is closest to the goal's Froude at t (`froude_t`, never a clip mean); the pick is
graded by the TRUE local Froude of that candidate's own recorded motion at the same offset t, in real
Froude units. Three numbers per library:

    selected  mean |true Froude of the pick - goal_t|
    oracle    the same, choosing the best candidate at every step with ground truth -- the LIBRARY's own
              ceiling, i.e. how much of goal space it covers at all, independent of any model
    random    a uniformly random candidate -- the floor

`selected - oracle` is what the scorer loses; `oracle` is what the library loses. Separating the two is the
point: a library can be bad (high oracle) or a scorer can be bad (selected far above oracle).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load       # noqa: E402
from final_2x2x2_test import build_planner          # noqa: E402
from froude_match_timevarying import run_direct     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--goal_dir", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--per_condition", type=int, default=999, help="999 = every clip is a candidate")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    planner = build_planner(args.ckpt, os.path.join(ROOT, args.candidates_dir), args.embodiment,
                            args.horizon, free_offset=False, device=args.device,
                            per_condition=args.per_condition)
    spec = REGISTRY[args.embodiment]
    mean, std = planner.mean_s, planner.std_s
    cand_bm = [np.asarray(load(c["path"], spec)["body_motion"])[:, :3] for c in planner.candidates]
    print(f"library: {args.candidates_dir}  ({len(cand_bm)} candidates)")

    goal_by_cond = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            goal_by_cond.setdefault(str(z["condition"]), p)

    rng = np.random.default_rng(0)
    rows = []
    for cond, gp in goal_by_cond.items():
        goal = np.asarray(load(gp, REGISTRY[args.goal_embodiment])["body_motion"])[:, :3]
        steps, ach = run_direct(planner, (goal - mean) / std, spec, args.horizon)
        sel, orc, rnd = [], [], []
        for k, t in enumerate(steps):
            local = np.array([b[t:t + args.horizon].mean(0) if len(b) > t else b[-1] for b in cand_bm])
            d = np.linalg.norm(local - goal[t], axis=1)
            sel.append(np.linalg.norm(ach[k] - goal[t]))
            orc.append(d.min())
            rnd.append(d.mean())
        rows.append((cond, np.mean(sel), np.mean(orc), np.mean(rnd)))
        print(f"  {cond:>14}  selected {rows[-1][1]:.4f}  oracle {rows[-1][2]:.4f}  random {rows[-1][3]:.4f}")

    a = np.array([r[1:] for r in rows])
    print(f"\nMEAN over {len(rows)} goals:  selected {a[:,0].mean():.4f}   oracle {a[:,1].mean():.4f}   "
          f"random {a[:,2].mean():.4f}")
    print(f"scorer loss (selected - oracle) {a[:,0].mean() - a[:,1].mean():.4f};  "
          f"share of the random-to-oracle gap recovered {100 * (a[:,2].mean() - a[:,0].mean()) / max(a[:,2].mean() - a[:,1].mean(), 1e-9):.0f}%")


if __name__ == "__main__":
    main()
