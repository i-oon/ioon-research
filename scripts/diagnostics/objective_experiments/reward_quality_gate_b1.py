"""Reward-quality gate for B1, rebuilt on B1's own MuJoCo harness -- the pre-registered Q21 check
that must clear before any RL controller is built (F136's question, F179's reason it matters).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/reward_quality_gate_b1.py

**The question, precisely**: can the current checkpoint's scoring function
(`body_head(proj(action))`, the exact thing that would serve as an RL reward) tell a genuinely good
action from a slightly-worse LOCAL perturbation of it? PPO (or any local-search RL) explores by
nudging its current action a little and needs the reward to say which nudge was better -- F136
measured the insect-side teacher failing exactly this (33% against a 50% coin) while succeeding at
the coarser "which recorded behaviour" question (84-95%). This rebuilds the same test on B1,
because F136's own script is tied to the insect/teacher-student pipeline (a different checkpoint, a
student network that does not exist for B1, and the insect CoppeliaSim scene).

**2026-09-18 correction: the action is continuous, so "did it pick the exact argmin" (a discrete
top-1 hit) is the wrong grade for it -- it can only ever say pass/fail on one candidate out of
`n_cands`, and throws away how close the model's ranking was on every trial it "failed."** The
metric now reported is the model's own dist-to-goal (`model_dist`) rank-correlated (Spearman) against
the true dist-to-goal (`true_dists`) across all `n_cands` candidates per trial -- a continuous
measure of ranking quality, not a coin-flip grade -- plus **regret**: the true distance-to-goal of
the action the model *actually* picked, minus the true distance-to-goal of the best real candidate,
in the same physical units the rest of this deck reports Froude error in. The old discrete hit-rate
is kept alongside for continuity with earlier numbers, not as the headline.

**Judged in REAL PHYSICS, not by the model** -- anything else lets the model grade its own homework.
Every perturbed action is executed for real in MuJoCo (`collect_b1_cpg_babble.py`'s own model/
control convention, reused verbatim: `il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)`), from a
snapshotted state saved/restored between trials so every variant starts from the IDENTICAL branch
point -- not a hypothetical, not a compounding rollout.

**No babble involved, deliberately.** An earlier version of this plan used a babble candidate's
action as the "known good" baseline to perturb -- that quietly reintroduces "is babble good" into a
test that is supposed to be about the reward function's own local discriminative power, independent
of any candidate source. The baseline action here is a recorded EXPERT clip's own action (B1's own
`data/egocentric/beh12_b1_ego_flat`), executed to reach a real branch state, precisely to keep this
test uncontaminated by babble's separately-open status (`doc/OPEN_QUESTION.md` Q21).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import mujoco
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

# **Inlined rather than imported, deliberately -- same reasoning as `wm/policy/b1_coppelia_env.py`.**
# These were pulled from `collect_b1_cpg_babble.py`, which broke this script's import once already
# when a parallel session moved it to `sim/collect/_archive/`. Copied here so this script has no
# cross-file dependency left to break. **Trade-off, stated plainly**: this is now a second copy of
# the B1 action-space convention -- if `DEFAULT_IL`/`ACTION_SCALE`/`IL_TO_SDK` ever change at the
# source, this copy can silently drift. Worth reconciling to one source of truth once the pipeline
# this serves stabilizes; not done now, to stop this exact breakage from recurring while it's in flux.
MODEL = os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat_real.xml")
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
IL_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])


def il_to_sdk(v):
    out = np.empty(12)
    out[IL_TO_SDK] = np.asarray(v)
    return out


from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load, body_velocity, yaw_rate  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def true_froude(positions, quats, dt, embodiment="b1"):
    height = float(np.median(positions[:, 2]))
    v = body_velocity(positions, quats, dt, embodiment)
    w = yaw_rate(quats, dt, embodiment, height)
    return np.concatenate([v, w], axis=1).mean(0)   # (fwd, lat, yaw), averaged over the window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/b1_babble_clean/body_head_b1_hex.pt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--expert_clip", default="data/egocentric/beh12_b1_ego_flat/b1_ep0.npz")
    ap.add_argument("--goal_clip", default="data/egocentric/beh12_c10f10t10_ego_flat/hexapod_ep100.npz",
                    help="the FORWARD goal reused throughout this session -- kept constant across "
                         "branch points so any variance measured is about the reward, not the goal")
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--branch_points", type=int, default=8, help="how many distinct states to test")
    ap.add_argument("--samples", type=int, default=16, help="Gaussian perturbations per branch point")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                    help="perturbation sizes, in raw action units (F136 used 0.5 as its default)")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--decimation", type=int, default=4, help="mujoco substeps per control step, "
                    "matching collect_b1_cpg_babble.py's --dt_decimation default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]
    adim = 12
    md = MotionDecoder(cfg, {"b1": adim}).to(device).eval(); md.load_state_dict(ck["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval(); proj.load_state_dict(ck["projector"])

    # 2026-09-18 fix (user-caught): this used to be `body_motion.mean(0)` -- one constant goal for
    # the WHOLE goal clip, reused identically at every branch point regardless of where in the
    # expert clip's own trajectory that branch sits. The exact same whole-clip-mean anti-pattern
    # already found and fixed in `final_2x2x2_test.py`/`froude_match_timevarying.py`, just not
    # caught here yet. Fix: keep the goal clip's per-frame trajectory, and read a short local
    # window from it, matched to how far through ITS OWN episode a fraction equal to the current
    # branch point's fraction through the EXPERT clip -- so an early branch point compares against
    # an early-goal-clip window, a late one against a late-goal-clip window, not one smeared average.
    goal_clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY[args.goal_embodiment])
    goal_bm = np.asarray(goal_clip["body_motion"])[:, :3]

    def goal_at(frac):
        idx = min(int(round(frac * (len(goal_bm) - 1))), len(goal_bm) - args.horizon - 1)
        idx = max(idx, 0)
        return goal_bm[idx:idx + args.horizon].mean(0)

    def model_dist(action_batch, goal_std):
        a = torch.as_tensor(action_batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = proj(a, "b1")
            pred = md.body(None, z).cpu().numpy()
        return np.abs(pred - goal_std).sum(-1)

    clip = np.load(os.path.join(ROOT, args.expert_clip), allow_pickle=True)
    clip_actions = clip["action"]
    n_clip = len(clip_actions)

    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, args.model))
    d = mujoco.MjData(m)
    real_dt = args.decimation * m.opt.timestep

    def apply(action):
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(args.decimation):
            mujoco.mj_step(m, d)

    # replay the clip's own actions from the start to reach realistic branch states (real qvel,
    # real contact dynamics -- not a teleported qpos with zero velocity)
    d.qpos[0:3] = clip["base_pos"][0]
    d.qpos[3:7] = clip["base_quat"][0]
    d.qpos[7:19] = clip["joint_pos"][0]
    mujoco.mj_forward(m, d)

    branch_step = max(1, n_clip // (args.branch_points + 1))
    hits_by_sigma = {s: [] for s in args.sigmas}
    hits_overall = []
    rho_by_sigma = {s: [] for s in args.sigmas}
    rho_overall = []
    regret_by_sigma = {s: [] for s in args.sigmas}
    regret_overall = []

    for bp in range(args.branch_points):
        t0 = (bp + 1) * branch_step
        # advance to t0 using the clip's own recorded actions
        for t in range(min(t0, n_clip - args.horizon - 1)):
            apply(clip_actions[t])
        qpos_snap, qvel_snap = d.qpos.copy(), d.qvel.copy()
        good_action = clip_actions[min(t0, n_clip - args.horizon - 1)]

        # goal matched to this branch point's own progress through the expert episode, not one
        # value reused for every branch point
        goal_fr = goal_at(t0 / max(n_clip - 1, 1))
        goal_std = (goal_fr - mean_s) / std_s

        for sigma in args.sigmas:
            actions = [good_action] + [good_action + rng.normal(0, sigma, size=12)
                                       for _ in range(args.samples)]
            true_dists = []
            for a in actions:
                d.qpos[:], d.qvel[:] = qpos_snap, qvel_snap
                mujoco.mj_forward(m, d)
                positions, quats = [], []
                for _ in range(args.horizon):
                    apply(a)
                    positions.append(d.qpos[0:3].copy())
                    quats.append(d.qpos[3:7].copy())
                fr = true_froude(np.asarray(positions), np.asarray(quats), real_dt)
                true_dists.append(float(np.abs(fr - goal_fr).sum()))
            m_dists = model_dist(np.stack(actions), goal_std)
            true_dists = np.asarray(true_dists)
            true_best = int(np.argmin(true_dists))
            model_best = int(np.argmin(m_dists))
            hit = int(true_best == model_best)
            hits_by_sigma[sigma].append(hit)
            hits_overall.append(hit)

            # continuous grades, since the action is continuous and a discrete top-1 hit throws
            # away how close the ranking was on every trial that "fails" it.
            rho = spearmanr(m_dists, true_dists).statistic
            if not np.isnan(rho):
                rho_by_sigma[sigma].append(rho)
                rho_overall.append(rho)
            regret = float(true_dists[model_best] - true_dists[true_best])
            regret_by_sigma[sigma].append(regret)
            regret_overall.append(regret)
        d.qpos[:], d.qvel[:] = qpos_snap, qvel_snap
        mujoco.mj_forward(m, d)

    n_cands = args.samples + 1
    chance = 1.0 / n_cands
    print(f"n_candidates per trial={n_cands}  chance={chance:.1%}")
    print("--- continuous grades (the real read, since the action is continuous) ---")
    print(f"overall Spearman rho (model rank vs true rank): {np.mean(rho_overall):.3f}"
          f"  (0 = model's ranking uncorrelated with truth, i.e. chance)")
    print(f"overall regret (true dist of model's pick − true dist of the real best, "
          f"same units as goal_fr): {np.mean(regret_overall):.4f}")
    for sigma in args.sigmas:
        r, g = rho_by_sigma[sigma], regret_by_sigma[sigma]
        print(f"  sigma={sigma:.2f}  rho={np.mean(r):.3f}  regret={np.mean(g):.4f}")
    print("--- discrete top-1 hit rate (kept for continuity with earlier reported numbers; "
          "NOT the headline -- see script docstring, 2026-09-18 correction) ---")
    print(f"overall hit rate: {np.mean(hits_overall):.1%}  ({sum(hits_overall)}/{len(hits_overall)})")
    for sigma in args.sigmas:
        h = hits_by_sigma[sigma]
        print(f"  sigma={sigma:.2f}  hit rate={np.mean(h):.1%}  ({sum(h)}/{len(h)})")


if __name__ == "__main__":
    main()
