"""Direct vs. FTM-rollout candidate scoring on the held-out leg-length body (c08f09t09), same
methodology as plot_direct_vs_rollout.py (B1 version), adapted to embodiment="hexapod" since
c08f09t09 is a morphology variant of the SAME embodiment/action-space, not a cross-embodiment case.

    .venv/bin/python3 scripts/figures/plot_direct_vs_rollout_c08f09t09.py \\
        --ckpt wm/runs/beh24_hinge_cleansplit/full_c08f09t09.pt \\
        --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat \\
        --goal_dir data/egocentric/beh12_c08f09t09_ego_flat --tag beh24_c08f09t09 --condition turn_s0.29
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

from wm.data.embodiment import REGISTRY, load                  # noqa: E402
from wm.evaluate import encode_clip, offset_for                # noqa: E402
from wm.policy.planner import RolloutFroudePlanner             # noqa: E402
from final_2x2x2_test import build_planner                     # noqa: E402
from froude_match_timevarying import run_direct, run_rollout, STYLE, LABELS   # noqa: E402

EMBODIMENT = "hexapod"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--goal_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--display_name", default="")
    ap.add_argument("--cond_label", default="")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--per_condition", type=int, default=999)
    ap.add_argument("--ylims", default="-0.12,0.33,-0.20,0.15,-0.07,0.14")
    ap.add_argument("--out_dir", default="results/deck/c08f09t09_selection")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vjepa2_encoder import VJEPA2FrameEncoder

    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    cdir = os.path.join(ROOT, args.candidates_dir)
    planner = build_planner(os.path.join(ROOT, args.ckpt), cdir, EMBODIMENT, args.horizon,
                            free_offset=False, device=args.device, per_condition=args.per_condition)
    spec = REGISTRY[EMBODIMENT]
    gp = None
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            if str(z["condition"]) == args.condition:
                gp = p
                break
    if gp is None:
        raise SystemExit(f"condition {args.condition!r} not found in {args.goal_dir}")
    goal = np.asarray(load(gp, REGISTRY[EMBODIMENT])["body_motion"])[:, :3]
    goal_std = np.stack([planner.standardize(g) for g in goal])

    steps, ach_d = run_direct(planner, goal_std, spec, args.horizon)

    rp = RolloutFroudePlanner.from_checkpoint(os.path.join(ROOT, args.ckpt), cdir, embodiment=EMBODIMENT,
                                              horizon=args.horizon, per_condition=args.per_condition, device=args.device)
    raw = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    r_offset = offset_for(raw, EMBODIMENT)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    with np.load(rp.candidates[0]["path"], allow_pickle=True) as d0:
        frame0 = d0["frames"][:1]
    e0 = encode_clip(encoder, frame0, 1).float()
    if r_offset is not None:
        e0 = e0 - r_offset.to(e0.device)
    del encoder
    torch.cuda.empty_cache()
    _, ach_r = run_rollout(rp, e0.to(args.device)[0], goal_std, spec, args.horizon)

    gs = goal[steps]
    err_d = np.linalg.norm(ach_d - gs, axis=1).mean()
    err_r = np.linalg.norm(ach_r - gs, axis=1).mean()
    f = [float(x) for x in args.ylims.split(",")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(steps, gs[:, i], **STYLE["goal_physics"])
        ax.plot(steps, ach_d[:, i], **STYLE["achieved_direct_p"])
        ax.plot(steps, ach_r[:, i], **STYLE["achieved_rollout"])
        ax.set_ylabel(LABELS[i]); ax.set_ylim(f[2 * i], f[2 * i + 1])
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"c08f09t09 held-out body: direct vs rollout scoring  [{args.display_name or args.tag}]\n"
                      f"goal: {args.cond_label or args.condition}  |  mean err: direct={err_d:.4f}  rollout={err_r:.4f}")
    axes[-1].set_xlabel(f"timestep (horizon={args.horizon}, goal read at every step)")
    plt.tight_layout()
    out = os.path.join(ROOT, args.out_dir, f"{args.tag}_direct_vs_rollout_{args.condition}.png")
    plt.savefig(out, dpi=120)
    print(f"direct {err_d:.4f}  rollout {err_r:.4f}  -> {out}")


if __name__ == "__main__":
    main()
