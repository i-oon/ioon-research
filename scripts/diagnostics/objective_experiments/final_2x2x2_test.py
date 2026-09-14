"""The corrected, single-mechanism 2x2x2 for Q21: fit/candidate pipeline x goal source x free_offset.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/final_2x2x2_test.py \\
        --pool expert --ckpt wm/runs/b1_adapt/body_head_b1.pt \\
        --candidates_dir data/egocentric/beh12_b1_ego_flat --per_condition 1
    .venv/bin/python3 scripts/diagnostics/objective_experiments/final_2x2x2_test.py \\
        --pool babble --ckpt wm/runs/b1_babble_clean/body_head_b1_hex.pt \\
        --candidates_dir data/egocentric/b1_babble_ego_flat

**Why this script exists and supersedes the two it merges.** Two earlier scripts each measured half
of this and, worse, measured "free_offset=False" two DIFFERENT ways: `score_babble_vs_expert_
candidates.py` used a whole-clip-mean convention (expert 42%/8% for mode A/D); `free_offset_
candidate_test.py` used a windowed-replan convention for its own "locked" baseline (expert 96%),
which is a finer, different measurement, not a second run of the same thing. Conflating the two
under one "free_offset=False" cell was caught only after both had already been reported (F193).
This script does ONE thing throughout: windowed scoring, matching exactly what
`wm.policy.planner.DirectFroudePlanner` does at decision time in the real closed loop -- so every
cell in the resulting table is the same mechanism, differing only in the three axes that are
supposed to differ.

**Fit/candidate pairing is never crossed, by design (the user's own correction to an earlier,
invalid comparison this project ran).** A checkpoint whose projector was fit on one action
distribution scores its OWN matching candidates -- an expert-fit checkpoint is never asked to score
babble candidates or vice versa. Run this script once per pipeline (`--pool expert` with the
expert-fit checkpoint and expert candidates, `--pool babble` with the babble-fit checkpoint and
babble candidates) and compare the two pipelines' own numbers, not a checkpoint against a foreign
candidate pool.

**Per-family (fwd/lat/yaw) breakdown always reported, never just the aggregate** -- F192 found the
aggregate can sit at 75-78% while the lateral-specific accuracy is at or below chance, entirely
hidden by 8 of 12 goals being forward-family.

**Goal source is orthogonal to everything else.** `--goal_source physics` reads the goal as a
recorded number (`body_motion.mean(0)`, mode A). `--goal_source vision` reads it from the goal
clip's own frames via `close_loop_direct_froude.vision_goal` (mode D) -- reused, not reimplemented,
so this script's mode-D goal is identical to what the real closed loop would compute.
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

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from close_loop_direct_froude import vision_goal  # noqa: E402 -- reuse, do not reimplement

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from wm.policy.planner import DirectFroudePlanner, load_candidates  # noqa: E402

FAMS = ["fwd", "lat", "yaw"]


def family(v):
    return int(np.argmax(np.abs(v)))


def build_planner(ckpt_path, candidates_dir, embodiment, horizon, free_offset, device, per_condition):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    cands = load_candidates(candidates_dir, embodiment, per_condition=per_condition)
    action_dim = cands[0]["actions"].shape[1]
    md = MotionDecoder(cfg, {embodiment: action_dim}).to(device).eval()
    md.load_state_dict(checkpoint["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(checkpoint)).to(device).eval()
    proj.load_state_dict(checkpoint["projector"])
    planner = DirectFroudePlanner(proj, md, cands, embodiment, horizon, device, free_offset=free_offset)
    planner.action_lag = max(1, cfg.action_lag)
    channels = [int(c) for c in cfg.body_channels]
    mean_s, std_s = checkpoint["body_stats"]
    planner.mean_s = np.asarray(mean_s).ravel()[:len(channels)]
    planner.std_s = np.asarray(std_s).ravel()[:len(channels)]
    planner.channels = channels
    planner.checkpoint = checkpoint
    planner.cfg = cfg
    return planner


def true_local_froude(cand, spec, offset, horizon):
    clip = load(cand["path"], spec)
    bm = np.asarray(clip["body_motion"])[:, :3]
    h = min(horizon, len(bm) - offset)
    return bm[offset:offset + max(h, 1)].mean(0)


def load_goals(goal_dir, goal_embodiment, planner, goal_source, goal_horizon, device):
    spec = REGISTRY[goal_embodiment]
    by_cond = {}
    for p in sorted(glob.glob(os.path.join(ROOT, goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            cond = str(d["condition"])
        by_cond.setdefault(cond, p)

    goals = []
    if goal_source == "vision":
        itm = InverseTransitionModel(planner.cfg).to(device).eval()
        itm.load_state_dict(planner.checkpoint["itm"])
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        offset = offset_for(planner.checkpoint, goal_embodiment)
        mean, std = planner.mean_s, planner.std_s
        # `md.body(None, z)` needs a MotionDecoder built for this embodiment's action_dim, but
        # `vision_goal` never reads actions -- planner.md already has whatever action head it was
        # built with and `.body` does not touch it, so reusing planner.md here is correct.
        for cond, p in by_cond.items():
            gvec, _ = vision_goal(itm, planner.md, encoder, offset, p, goal_horizon)
            gvec = gvec.cpu().numpy() * std + mean
            goals.append({"condition": cond, "froude": gvec})
        del encoder
        torch.cuda.empty_cache()
    else:
        for cond, p in by_cond.items():
            clip = load(p, spec)
            fr = np.asarray(clip["body_motion"])[:, :3].mean(0)
            goals.append({"condition": cond, "froude": fr})
    return goals


def pool_chance(planner, spec, goals):
    """Random-pick-from-the-SAME-pool expectation (F184's convention), NOT a shuffled-goal
    control -- shuffling was tried elsewhere and found invalid when a pool is dominated by one
    family (F188). Each candidate's family is its WHOLE-CLIP mean (a static property, unlike the
    time-windowed accuracy this script otherwise measures) -- chance is "if you closed your eyes
    and picked ANY candidate," which has no time axis to window over."""
    cand_fams = np.array([family(true_local_froude(c, spec, 0, 10**9)) for c in planner.candidates])
    return float(np.mean([(cand_fams == family(g["froude"])).mean() for g in goals]))


def pool_chance_dist(planner, spec, goals):
    """Mean |froude| L1 distance to the goal from a RANDOM pick of the same pool -- the continuous
    counterpart to `pool_chance`'s family-match rate. Needed for the same reason F102's own README
    already states: a family-match rate hides whether a miss was by a point or by a factor of
    three. Whole-clip mean froude per candidate (same static convention as `pool_chance`)."""
    cand_fr = np.stack([true_local_froude(c, spec, 0, 10**9) for c in planner.candidates])
    return float(np.mean([np.abs(cand_fr - g["froude"]).sum(-1).mean() for g in goals]))


def run(planner, goals, spec, horizon, n_steps, free):
    hits, n_trials = 0, 0
    by_family = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    boundary_jumps, interior_jumps = [], []
    froude_dists = []
    for g in goals:
        goal_std = planner.standardize(g["froude"])
        gfam = family(g["froude"])
        prev_action = None
        t = 0
        free_i = free_tau0 = free_t0 = None   # only used when free=True, see below
        while t < n_steps:
            if free:
                # **Correction (2026-09-11): `score_offsets` is deterministic under an unchanging
                # goal, so re-searching every loop iteration (the original form of this branch)
                # returns the IDENTICAL (candidate, tau0) every single time -- confirmed directly,
                # and it is what caused a real closed-loop episode to compound one short window's
                # own non-zero net drift into a catastrophic tip-over (see
                # sim/control/close_loop_direct_froude.py's matching fix and its docstring for the
                # full incident). Re-search ONCE, then advance `tau` continuously; only re-search
                # again if the current window would run past this candidate's own recorded length.
                if free_i is None or free_tau0 + (t - free_t0) + horizon > len(planner.candidates[free_i]["actions"]):
                    rows = planner.score_offsets(goal_std)
                    free_i = int(np.argmin([np.min(r) for r in rows]))
                    free_tau0 = int(np.argmin(rows[free_i]))
                    free_t0 = t
                i, tau = free_i, free_tau0 + (t - free_t0)
                cand = planner.candidates[i]
                window = cand["actions"][tau:tau + horizon]
            else:
                _, i, _, _ = planner.act(goal_std, t)
                cand = planner.candidates[i]
                tau, window = t, cand["actions"][t:t + horizon]
            local_true = true_local_froude(cand, spec, min(tau, len(cand["actions"]) - 1), horizon)
            ok = int(family(local_true) == gfam)
            hits += ok
            by_family[gfam][0] += ok
            by_family[gfam][1] += 1
            n_trials += 1
            # **The continuous metric the family-match rate hides**: not "same family, yes/no"
            # but how far the TRUE achieved motion sits from the goal actually given (right or
            # wrong -- this is the same goal vector `gfam` was computed from, so it carries no new
            # confound between goal-reading error and selection error). Two "correct" picks can
            # differ enormously in how close they actually are; this is what F102's own
            # closed-loop README already warned a family-only rate would hide.
            froude_dists.append(float(np.abs(local_true - g["froude"]).sum()))
            for k, action in enumerate(window):
                if prev_action is not None:
                    d = float(np.linalg.norm(action - prev_action))
                    (boundary_jumps if k == 0 else interior_jumps).append(d)
                prev_action = action
            t += max(1, len(window))
    return {
        "accuracy": hits / max(n_trials, 1),
        "n_trials": n_trials,
        "hits": hits,
        "by_family": by_family,
        "froude_dist_mean": float(np.mean(froude_dists)) if froude_dists else 0.0,
        "froude_dist_median": float(np.median(froude_dists)) if froude_dists else 0.0,
        "boundary_jump_median": float(np.median(boundary_jumps)) if boundary_jumps else 0.0,
        "boundary_jump_p95": float(np.percentile(boundary_jumps, 95)) if boundary_jumps else 0.0,
        "interior_jump_median": float(np.median(interior_jumps)) if interior_jumps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=("expert", "babble"), help="label only, "
                    "for the printed report -- does not change any computation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--per_condition", type=int, default=None)
    ap.add_argument("--goal_dir", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizon", type=int, default=5,
                    help="the PLANNER's window: how many actions a chosen candidate contributes "
                         "before the next decision. Nothing to do with reading the goal.")
    ap.add_argument("--goal_horizon", type=int, default=1,
                    help="frame spacing (t, t+goal_horizon) used to READ the goal from the source "
                         "clip's video in mode C/D. Until 2026-09-14 this file passed --horizon "
                         "here, so a planner window of 5 silently read the goal at a 5-frame "
                         "spacing the ITM was never trained on -- F208's one-flag-two-jobs bug, "
                         "fixed in close_loop_direct_froude.py and plan_without_library.py but "
                         "missed here. Mode C/D numbers produced by this script before that date "
                         "were read at the planner's horizon, not at 1.")
    ap.add_argument("--n_steps", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(ROOT, args.ckpt)
    candidates_dir = os.path.join(ROOT, args.candidates_dir)
    spec = REGISTRY[args.embodiment]

    print(f"##### pool={args.pool}  ckpt={args.ckpt}  candidates={args.candidates_dir} #####\n")

    for goal_source in ("physics", "vision"):
        for free in (False, True):
            planner = build_planner(ckpt_path, candidates_dir, args.embodiment, args.horizon,
                                    free, device, args.per_condition)
            goals = load_goals(args.goal_dir, args.goal_embodiment, planner, goal_source,
                               args.goal_horizon, device)
            r = run(planner, goals, spec, args.horizon, args.n_steps, free)
            chance = pool_chance(planner, spec, goals)
            chance_dist = pool_chance_dist(planner, spec, goals)
            mode = "A (physics)" if goal_source == "physics" else "D (vision)"
            fo = "free_offset=True " if free else "free_offset=False"
            print(f"=== goal {mode}  {fo} ===")
            print(f"  accuracy={r['accuracy']:.0%} ({r['hits']}/{r['n_trials']})  chance={chance:.0%}  "
                 f"boundary_jump median={r['boundary_jump_median']:.4f} "
                 f"p95={r['boundary_jump_p95']:.4f}  "
                 f"interior_jump median={r['interior_jump_median']:.4f}")
            print(f"  froude_dist (continuous, NOT hidden by family match) "
                 f"mean={r['froude_dist_mean']:.4f}  median={r['froude_dist_median']:.4f}  "
                 f"chance(random pick)={chance_dist:.4f}")
            for fi in (0, 1, 2):
                h, n = r["by_family"][fi]
                if n:
                    print(f"    {FAMS[fi]:4s} goals: {h}/{n} ({h/n:.0%})")
                else:
                    print(f"    {FAMS[fi]:4s} goals: none in this goal set")
            print()


if __name__ == "__main__":
    main()
