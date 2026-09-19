"""Run froude_match_timevarying.py's exact per-timestep methodology across all 12 hexapod goal
conditions in one process (single model/encoder load), instead of one clip -- extends Slide 21's
single-clip (turn_s0.05) result and is the intended replacement for the withdrawn "92% across all
12 goal conditions" aggregate (that number used final_2x2x2_test.py's whole-clip-mean goal, the bug
already found and fixed; this is the corrected methodology run over the full condition set instead
of one clip).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/froude_match_all12.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/stage3_b1_nce_s0.pt \\
        --candidates_dir data/egocentric/beh12_b1_ego_flat \\
        --goal_dir data/egocentric/beh12_c10f10t10_ego_flat

**Checkpoint note.** Slide 21's own single-clip numbers (0.038/0.082/0.034/0.038) used
`beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt`, which is not on this
machine (checkpoints move via Google Drive only). This runs the identical methodology on the
nearest available sibling checkpoint instead -- numbers will not exactly match Slide 21's single
point, by construction, but the shape of the result (does direct beat rollout, does vision cost
nothing, across every condition not just one) is a fresh, real measurement.

No new architecture, no retraining -- reuses `froude_match_timevarying.py`'s own functions.
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

from wm.data.embodiment import REGISTRY, load          # noqa: E402
from wm.evaluate import offset_for                      # noqa: E402
from wm.models.itm import InverseTransitionModel        # noqa: E402
from wm.policy.planner import RolloutFroudePlanner       # noqa: E402
from final_2x2x2_test import build_planner               # noqa: E402
from froude_match_timevarying import (                   # noqa: E402
    read_vision_goal_per_timestep, run_direct, run_rollout,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--goal_dir", required=True)
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--goal_horizon", type=int, default=1)
    ap.add_argument("--per_condition", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--skip_rollout", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

    device = args.device
    planner = build_planner(args.ckpt, args.candidates_dir, args.embodiment, args.horizon,
                            free_offset=False, device=device, per_condition=args.per_condition)
    spec = REGISTRY[args.embodiment]

    # one goal clip per condition, same selection rule as score_by_body_motion.py: first match wins
    goal_by_cond = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            cond = str(z["condition"])
        goal_by_cond.setdefault(cond, p)
    print(f"{len(goal_by_cond)} goal conditions from {args.goal_dir}")

    itm = InverseTransitionModel(planner.cfg).to(device).eval()
    itm.load_state_dict(planner.checkpoint["itm"])
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    offset = offset_for(planner.checkpoint, args.goal_embodiment)
    mean, std = planner.mean_s, planner.std_s

    rollout_planner = None
    e_t_fixed = None
    if not args.skip_rollout:
        rollout_planner = RolloutFroudePlanner.from_checkpoint(
            args.ckpt, args.candidates_dir, embodiment=args.embodiment, horizon=args.horizon,
            device=device, per_condition=args.per_condition)
        # fixed e_t: first frame of the first candidate (same simplification as
        # froude_match_timevarying.py's own single-clip run -- see its module docstring)
        first_cand_path = planner.candidates[0]["path"]
        with np.load(first_cand_path, allow_pickle=True) as fc:
            from wm.evaluate import encode_clip
            e_all = encode_clip(encoder, fc["frames"][:1], 1).float()
        e_t_fixed = e_all[0].to(device)

    goal_spec = REGISTRY[args.goal_embodiment]
    rows = []
    for cond, gp in goal_by_cond.items():
        goal_vec = read_vision_goal_per_timestep(itm, planner.md, encoder, offset, gp,
                                                  args.goal_horizon, args.chunk)
        clip = load(gp, goal_spec)
        goal_p_real = np.asarray(clip["body_motion"])[:len(goal_vec), :3]
        goal_v_real = goal_vec * std + mean
        goal_p_std = (goal_p_real - mean) / std

        steps_d, ach_d = run_direct(planner, goal_p_std, spec, args.horizon)
        goal_p_at_steps = goal_p_real[steps_d]
        err_direct_physics = float(np.linalg.norm(ach_d - goal_p_at_steps, axis=1).mean())

        steps_dv, ach_dv = run_direct(planner, (goal_v_real - mean) / std, spec, args.horizon)
        goal_v_at_steps = goal_v_real[steps_dv]
        err_direct_vision = float(np.linalg.norm(ach_dv - goal_v_at_steps, axis=1).mean())

        err_rollout = None
        if rollout_planner is not None:
            steps_r, ach_r = run_rollout(rollout_planner, e_t_fixed, goal_p_std, spec, args.horizon)
            err_rollout = float(np.linalg.norm(ach_r - goal_p_real[steps_r], axis=1).mean())

        row = dict(condition=cond, direct_physics=err_direct_physics,
                   direct_vision=err_direct_vision, rollout=err_rollout, n=len(steps_d))
        rows.append(row)
        print(f"  {cond:>14}  direct(physics)={err_direct_physics:.4f}  "
              f"direct(vision)={err_direct_vision:.4f}  "
              f"rollout={'--' if err_rollout is None else f'{err_rollout:.4f}'}  n={row['n']}")

    dp = np.array([r["direct_physics"] for r in rows])
    dv = np.array([r["direct_vision"] for r in rows])
    print(f"\nmean across {len(rows)} conditions:")
    print(f"  direct, physics goal: {dp.mean():.4f}  (range {dp.min():.4f}-{dp.max():.4f})")
    print(f"  direct, vision goal:  {dv.mean():.4f}  (range {dv.min():.4f}-{dv.max():.4f})")
    if rollout_planner is not None:
        rr = np.array([r["rollout"] for r in rows])
        print(f"  rollout, physics goal: {rr.mean():.4f}  (range {rr.min():.4f}-{rr.max():.4f})")
        beat = int((dp < rr).sum())
        print(f"  direct beats rollout on {beat}/{len(rows)} conditions")


if __name__ == "__main__":
    main()
