"""Goal-vs-picked Froude, tracked per timestep -- the corrected replacement for final_2x2x2_test.py's
whole-clip-mean goal, on real B1 direct and rollout planners.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/froude_match_timevarying.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt \\
        --candidates_dir data/egocentric/beh12_b1_ego_flat \\
        --goal_path data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep1001.npz \\
        --out_dir results/deck

**What this fixes.** `final_2x2x2_test.py`'s mode A/D (`load_goals`) reads the goal as
`body_motion.mean(0)` -- one constant 3-vector for the whole clip -- then measures whether the
picked candidate's family matches it at each decision step. The user's own correction: a goal that
changes every frame (a turn accelerating into its steady state, say) is not well served by a single
mean, and the real question is a continuous distance to whatever the goal *actually is at that
instant*, not a family-match rate against an average. This script re-reads the goal at every
timestep instead of once, both ways it can be read (physics/privileged, vision-read), and reports
the L2 distance from the picked candidate's own true local Froude to the goal's value AT THAT SAME
TIMESTEP, plotted as three time series (forward/lateral/yaw), not collapsed into one number.

**Rollout's `e_t` is a real simplification, stated plainly.** `RolloutFroudePlanner` needs the
controlled body's own live observation embedding at every step; there is no running closed loop
here, so `e_t` is a single fixed B1 frame, held constant across the whole trial. A real closed loop
would update it as the body actually moved. This measures rollout's scoring mechanism in isolation,
not a live control episode -- read the rollout numbers with that caveat attached.

**A bug this script fixes vs. the first cut of this diagnostic (session note, not yet an F-number):**
saving `physics_goal_traj[:len(achieved)]` (the first N *consecutive* frames) instead of the
frames actually sampled at the strided decision timesteps silently misaligned the printed rollout
error by about 10%. Every saved trace here is indexed by the same `steps` array its goal was read
at -- verify this by construction, not by eye, if this script is ever edited.

Diagnosis only. No tuning, no retraining, no new checkpoints.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load                     # noqa: E402
from wm.evaluate import encode_clip, offset_for                   # noqa: E402
from wm.models.itm import InverseTransitionModel                  # noqa: E402
from wm.policy.planner import RolloutFroudePlanner                # noqa: E402
from final_2x2x2_test import build_planner, true_local_froude     # noqa: E402

LABELS = ("forward", "lateral", "yaw")

# one fixed style per quantity, shared across every plot so the same series always looks the same
STYLE = {
    "goal_physics":      dict(color="black",   ls="-",  lw=2,   label="goal, physics (privileged)"),
    "goal_vision":       dict(color="gray",    ls=":",  lw=1.8, label="goal, vision-read"),
    "achieved_direct_p": dict(color="#2a7a3b", ls="-",  lw=1.8, label="achieved -- direct (physics goal)"),
    "achieved_direct_v": dict(color="#1f6fb2", ls="--", lw=1.8, label="achieved -- direct (vision goal)"),
    "achieved_rollout":  dict(color="crimson", ls="-.", lw=1.8, label="achieved -- rollout (physics goal)"),
}


def read_vision_goal_per_timestep(itm, md, encoder, offset, goal_path, goal_horizon, chunk):
    """Every (t, t+goal_horizon) pair, read via ITM+body_head, kept separate -- never `.mean(0)`'d
    into one number the way `close_loop_direct_froude.vision_goal` does for the original test.
    Batched in `chunk`-sized groups: a full-batch ITM call OOM'd once on a GPU shared with another
    job (11 GB card, ~1 GB free) -- small chunks make this robust to sharing, not just faster."""
    device = next(itm.parameters()).device
    with np.load(goal_path, allow_pickle=True) as gd:
        frames = gd["frames"]
    n_pairs = max(1, len(frames) - goal_horizon)
    idx0 = np.arange(n_pairs)
    idx1 = idx0 + goal_horizon
    ge = encode_clip(encoder, frames[np.concatenate([idx0, idx1])], 2).float()
    if offset is not None:
        ge = ge - offset.to(ge.device)
    g0, g1 = ge[:n_pairs], ge[n_pairs:]
    outs = []
    with torch.no_grad():
        for s in range(0, n_pairs, chunk):
            z = itm(g0[s:s + chunk].to(device), g1[s:s + chunk].to(device))
            outs.append(md.body(None, z).cpu().numpy())
            torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)  # (n_pairs, 3), standardised units, per-timestep


def run_direct(planner, goal_std_traj, spec, horizon):
    """Real candidates, real projector+body_head, time-varying goal read fresh at every t."""
    n = len(goal_std_traj) - horizon
    steps, achieved = [], []
    t = 0
    while t < n:
        _, i, _, _ = planner.act(goal_std_traj[t], t)
        cand = planner.candidates[i]
        achieved.append(true_local_froude(cand, spec, t, horizon))
        steps.append(t)
        t += horizon
    return np.array(steps), np.array(achieved)


def run_rollout(planner, e_t_fixed, goal_std_traj, spec, horizon):
    """Real FTM rollout scoring, same time-varying goal, fixed e_t (see module docstring)."""
    n = len(goal_std_traj) - horizon
    steps, achieved = [], []
    t = 0
    while t < n:
        goal_t = torch.as_tensor(goal_std_traj[t], dtype=torch.float32)
        _, i, _ = planner.act(e_t_fixed, goal_t, t)
        cand = planner.candidates[i]
        achieved.append(true_local_froude(cand, spec, t, horizon))
        steps.append(t)
        t += horizon
    return np.array(steps), np.array(achieved)


def shared_ylims(series_list, pad_frac=0.08):
    ylims = []
    for i in range(3):
        lo = min(s[:, i].min() for s in series_list)
        hi = max(s[:, i].max() for s in series_list)
        pad = pad_frac * (hi - lo)
        ylims.append((lo - pad, hi + pad))
    return ylims


def _base_fig(ylims, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.set_ylabel(LABELS[i])
        ax.set_ylim(*ylims[i])
    axes[-1].set_xlabel(xlabel)
    return plt, fig, axes


def make_plots(out_dir, steps, goal_p, goal_v, achieved_p, achieved_v, achieved_r, goal_name, horizon):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    all_series = [goal_p, goal_v, achieved_p, achieved_v, achieved_r]
    ylims = shared_ylims(all_series)
    xlabel = f"timestep (horizon={horizon}, goal read at every step)"

    def line(ax, data, key, i):
        ax.plot(steps, data[:, i], **STYLE[key])

    # 1. goal reading only
    err_gv = np.linalg.norm(goal_v - goal_p, axis=1).mean()
    plt_, fig, axes = _base_fig(ylims, xlabel)
    for i, ax in enumerate(axes):
        line(ax, goal_p, "goal_physics", i); line(ax, goal_v, "goal_vision", i)
        if i == 0: ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"1. Goal reading: physics vs vision, no planner involved\n"
                      f"goal={goal_name}  |  mean |vision-read - physics| = {err_gv:.4f}")
    plt_.tight_layout()
    plt_.savefig(os.path.join(out_dir, "froude_1_physics_vs_vision_goal.png"), dpi=120)
    plt_.close(fig)

    # 2. physics goal, direct vs rollout
    err_d = np.linalg.norm(achieved_p - goal_p, axis=1).mean()
    err_r = np.linalg.norm(achieved_r - goal_p, axis=1).mean()
    plt_, fig, axes = _base_fig(ylims, xlabel)
    for i, ax in enumerate(axes):
        line(ax, goal_p, "goal_physics", i)
        line(ax, achieved_p, "achieved_direct_p", i)
        line(ax, achieved_r, "achieved_rollout", i)
        if i == 0: ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"2. Physics goal: direct vs rollout scoring\n"
                      f"goal={goal_name}  |  mean err: direct={err_d:.4f}  rollout={err_r:.4f}")
    plt_.tight_layout()
    plt_.savefig(os.path.join(out_dir, "froude_2_physics_goal_direct_vs_rollout.png"), dpi=120)
    plt_.close(fig)

    # 3. vision goal, direct
    err_v = np.linalg.norm(achieved_v - goal_v, axis=1).mean()
    plt_, fig, axes = _base_fig(ylims, xlabel)
    for i, ax in enumerate(axes):
        line(ax, goal_v, "goal_vision", i)
        line(ax, achieved_v, "achieved_direct_v", i)
        if i == 0: ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"3. Vision goal, direct scoring\ngoal={goal_name}  |  mean err = {err_v:.4f}")
    plt_.tight_layout()
    plt_.savefig(os.path.join(out_dir, "froude_3_vision_goal_direct.png"), dpi=120)
    plt_.close(fig)

    return dict(vision_vs_physics_goal=err_gv, direct=err_d, rollout=err_r, vision_goal_direct=err_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--goal_path", required=True)
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--horizon", type=int, default=2,
                    help="candidate-scoring window. 5 (final_2x2x2_test.py's default) was found "
                         "too coarse for a per-timestep check; 1 alone was too noisy (compares a "
                         "single unsmoothed frame's achieved value against the goal's own smoothed "
                         "one). 2 is this script's default, not a tuned optimum -- vary it and look.")
    ap.add_argument("--goal_horizon", type=int, default=1,
                    help="ITM pairing spacing for the vision-read goal. Matches "
                         "final_2x2x2_test.py's own --goal_horizon default.")
    ap.add_argument("--per_condition", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4, help="ITM batch size for vision-goal reading")
    ap.add_argument("--skip_rollout", action="store_true",
                    help="rollout needs its own model load + a fixed e_t frame encoded; skip if "
                         "only the goal-reading/direct comparison is needed")
    ap.add_argument("--out_dir", default="results/deck")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402  (heavy import, deferred)

    device = args.device
    planner = build_planner(args.ckpt, args.candidates_dir, args.embodiment, args.horizon,
                            free_offset=False, device=device, per_condition=args.per_condition)
    spec = REGISTRY[args.embodiment]

    itm = InverseTransitionModel(planner.cfg).to(device).eval()
    itm.load_state_dict(planner.checkpoint["itm"])

    print("encoding the goal clip (VJEPA2, real cost, one clip)...", flush=True)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    offset = offset_for(planner.checkpoint, args.goal_embodiment)
    goal_vision_std = read_vision_goal_per_timestep(itm, planner.md, encoder, offset,
                                                    args.goal_path, args.goal_horizon, args.chunk)
    del encoder
    torch.cuda.empty_cache()

    goal_clip = load(args.goal_path, REGISTRY[args.goal_embodiment])
    goal_physics_traj = np.asarray(goal_clip["body_motion"])[:, :3]
    mean, std = planner.mean_s, planner.std_s
    goal_physics_std = np.stack([planner.standardize(goal_physics_traj[t])
                                 for t in range(len(goal_physics_traj))])

    steps, achieved_p = run_direct(planner, goal_physics_std, spec, args.horizon)
    _, achieved_v = run_direct(planner, goal_vision_std, spec, args.horizon)
    goal_p_raw = goal_physics_traj[steps]
    goal_v_raw = goal_vision_std[steps] * std + mean

    if args.skip_rollout:
        achieved_r = np.full_like(achieved_p, np.nan)
    else:
        print("building rollout planner + fixed e_t stand-in...", flush=True)
        rp = RolloutFroudePlanner.from_checkpoint(args.ckpt, args.candidates_dir,
                                                  embodiment=args.embodiment, horizon=args.horizon,
                                                  per_condition=args.per_condition, device=device)
        raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        r_offset = offset_for(raw_ckpt, args.embodiment)
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        cand0_path = os.path.join(args.candidates_dir, os.path.basename(rp.candidates[0]["path"]))
        with np.load(cand0_path, allow_pickle=True) as d0:
            frame0 = d0["frames"][:1]
        e0 = encode_clip(encoder, frame0, 1).float()
        if r_offset is not None:
            e0 = e0 - r_offset.to(e0.device)
        e_t_fixed = e0.to(device)
        del encoder
        torch.cuda.empty_cache()
        _, achieved_r = run_rollout(rp, e_t_fixed, goal_physics_std, spec, args.horizon)

    goal_name = os.path.splitext(os.path.basename(args.goal_path))[0]
    errs = make_plots(args.out_dir, steps, goal_p_raw, goal_v_raw, achieved_p, achieved_v,
                      achieved_r, goal_name, args.horizon)
    print("mean errors:", {k: round(v, 4) for k, v in errs.items()})
    print(f"plots -> {args.out_dir}/froude_{{1,2,3}}_*.png")


if __name__ == "__main__":
    main()
