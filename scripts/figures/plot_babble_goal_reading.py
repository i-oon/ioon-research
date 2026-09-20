"""Goal reading, physics vs. vision, no planner involved -- the same figure as
results/deck/groundtruth_b1/froude_1_physics_vs_vision_goal.png, for a checkpoint adapted on a babble library.

    .venv/bin/python3 scripts/figures/plot_babble_goal_reading.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/ckpt_full_b1_babble_v2_ego_flat.pt \\
        --candidates_dir data/egocentric/b1_babble_v2_ego_flat --tag babble_v2 \\
        --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout

The vision goal is read by THAT checkpoint's own ITM + body_head from the hexapod goal clip's video, so it
depends on what the checkpoint was adapted on. Same reading function as froude_match_timevarying.py.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load                # noqa: E402
from wm.evaluate import offset_for                            # noqa: E402
from wm.models.itm import InverseTransitionModel              # noqa: E402
from final_2x2x2_test import build_planner                    # noqa: E402
from froude_match_timevarying import (read_vision_goal_per_timestep, shared_ylims,   # noqa: E402
                                      STYLE, LABELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--goal_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--ylims", default="", help="fwd_lo,fwd_hi,lat_lo,lat_hi,yaw_lo,yaw_hi: one fixed scale for "
                                                "every plot, so figures are comparable; empty = per-plot")
    ap.add_argument("--out_dir", default="results/deck/babble_selection")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vjepa2_encoder import VJEPA2FrameEncoder

    planner = build_planner(args.ckpt, os.path.join(ROOT, args.candidates_dir), "b1", args.horizon,
                            free_offset=False, device=args.device, per_condition=1)
    itm = InverseTransitionModel(planner.cfg).to(args.device).eval()
    itm.load_state_dict(planner.checkpoint["itm"])
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    offset = offset_for(planner.checkpoint, "hexapod")
    mean, std = planner.mean_s, planner.std_s

    goal_by_cond = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            goal_by_cond.setdefault(str(z["condition"]), p)
    want = [c for c in args.conditions.split(",") if c] or list(goal_by_cond)
    out = os.path.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)

    fixed = [float(x) for x in args.ylims.split(",")] if args.ylims else None
    errs, span = [], []
    for cond in want:
        gp = goal_by_cond[cond]
        gv_std = read_vision_goal_per_timestep(itm, planner.md, encoder, offset, gp, 1, 4)
        phys = np.asarray(load(gp, REGISTRY["hexapod"])["body_motion"])[:, :3]
        steps = np.arange(0, len(gv_std) - args.horizon, args.horizon)
        goal_p, goal_v = phys[steps], (gv_std * std + mean)[steps]
        err = np.linalg.norm(goal_v - goal_p, axis=1).mean()
        errs.append(err)
        ylims = ([(fixed[2 * i], fixed[2 * i + 1]) for i in range(3)] if fixed
                 else shared_ylims([goal_p, goal_v]))
        span.append(np.stack([np.minimum(goal_p, goal_v).min(0), np.maximum(goal_p, goal_v).max(0)]))
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(steps, goal_p[:, i], **STYLE["goal_physics"])
            ax.plot(steps, goal_v[:, i], **STYLE["goal_vision"])
            ax.set_ylabel(LABELS[i]); ax.set_ylim(*ylims[i])
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"1. Goal reading: physics vs vision, no planner involved\n[{args.tag}]  "
                          f"goal={cond} (hex)  |  mean |vision-read - physics| = {err:.4f}")
        axes[-1].set_xlabel(f"timestep (horizon={args.horizon}, goal read at every step)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{args.tag}_goal_reading_{cond}.png"), dpi=120)
        plt.close(fig)
        print(f"{cond:>14}  goal read err {err:.4f}")
    sp = np.array(span)
    print(f"MEAN goal read err over {len(errs)}: {np.mean(errs):.4f}")
    print("DATA RANGE per channel (min,max):", [(round(float(sp[:, 0, i].min()), 3), round(float(sp[:, 1, i].max()), 3)) for i in range(3)])


if __name__ == "__main__":
    main()
