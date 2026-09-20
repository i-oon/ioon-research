"""F237's generalization check: does the same recipe (short horizon + symlog/return-norm +
periodic MC-anchor) still converge across MANY (start, goal) pairs, or only the one fixed pair
the isolation test used?

**Why this exists.** F237 and its two follow-ups all used a SINGLE fixed start state and a SINGLE
fixed goal, deliberately -- that is the isolation test's own design, meant to cleanly separate
"does this converge at all" from "does it generalize." The periodic-MC-anchor version was the first
in this whole arc where the actor's real, independently-verified return converged and held
(F237's second follow-up). That result says nothing yet about whether the same actor/critic can
learn a policy that works for more than the one state/goal pair it saw -- this script is that
direct test.

**What changes from `rl_imagination_isolation_v2.py`.** The actor and critic are now
**goal-conditioned**: input is `concat(pooled(e_t), goal)`, not just `pooled(e_t)`, since a single
set of weights now has to produce different actions for different goals. Each training iteration
samples a batch of (start, goal) TASKS (with replacement) from a pool built from real B1 clips --
one task per clip, start = that clip's own first frame, goal = that clip's own steady-state Froude
(same steady-state-window convention as v2 and Slide 30's bug-3 fix). Training tasks come from
`_cleantrain` (12 clips, one per condition); evaluation tasks come from `_cleanheldout` (12 clips,
the other instance of each condition) -- genuinely held out, never used to anchor or train.

**Everything else is unchanged from F237's best recipe**: H=12 (F236's window), symlog critic +
return normalization (F179's rung 2), periodic hard-reset to a real Monte Carlo anchor (F237's own
addition) -- now anchored per-task (a random subset of the training pool each anchor point) instead
of the single fixed state the isolation test used.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/rl_imagination_isolation_v3.py \\
        --iters 2000 --horizon 12 --tasks_per_iter 4 --anchor_every 200
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.config import from_checkpoint                # noqa: E402
from wm.data.embodiment import REGISTRY, load         # noqa: E402
from wm.evaluate import encode_clip                   # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel      # noqa: E402
from wm.models.itm import InverseTransitionModel      # noqa: E402
from wm.models.motion_decoder import MotionDecoder    # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder         # noqa: E402

EMBODIMENT = "b1"


def pooled(e):
    return e.mean(-2)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.expm1(x.abs()))


class ReturnScale:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.scale = None

    def update(self, returns):
        with torch.no_grad():
            lo = torch.quantile(returns, 0.05)
            hi = torch.quantile(returns, 0.95)
            batch_scale = (hi - lo).clamp(min=1.0)
            self.scale = batch_scale if self.scale is None else \
                self.decay * self.scale + (1 - self.decay) * batch_scale
        return self.scale


class GaussianActor(nn.Module):
    """`concat(pooled(e_t), goal)` -> a Gaussian over raw actions -- goal-conditioned so one set of
    weights can serve every task in the pool, not just the single pair v2's isolation test used."""

    def __init__(self, state_dim, goal_dim, action_dim, action_mean, action_std,
                 range_sigma=2.0, hidden=128):
        super().__init__()
        in_dim = state_dim + goal_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.range_sigma = range_sigma

    def _bound(self, raw_mean):
        return self.action_mean + self.range_sigma * self.action_std * torch.tanh(raw_mean)

    def forward(self, state, goal, deterministic=False):
        h = self.net(torch.cat([state, goal], dim=-1))
        mean = self._bound(self.mean_head(h))
        if deterministic:
            return mean, None
        std = self.log_std.exp().expand_as(mean)
        eps = torch.randn_like(mean)
        action = mean + std * eps
        log_prob = (-0.5 * ((action - mean) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden=128):
        super().__init__()
        in_dim = state_dim + goal_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, state, goal):
        return self.net(torch.cat([state, goal], dim=-1)).squeeze(-1)


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


def load_frozen(ckpt_path, device):
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, None).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    for m in (itm, ftm, md):
        for p in m.parameters():
            p.requires_grad_(False)
    mean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32, device=device)
    std = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32, device=device)
    return ck, cfg, itm, ftm, md, mean, std


def load_projector(proj_path, cfg, device):
    saved = torch.load(os.path.join(ROOT, proj_path), map_location="cpu", weights_only=False)
    dims = action_dims_from(saved)
    proj = ActionProjector(cfg, dims).to(device).eval()
    state = saved.get("projector", saved)
    proj.load_state_dict(state)
    for p in proj.parameters():
        p.requires_grad_(False)
    mean = getattr(proj, f"mean_{EMBODIMENT}").clone()
    std = getattr(proj, f"std_{EMBODIMENT}").clone()
    return proj, mean, std


@torch.no_grad()
def build_task_pool(encoder, data_dir, body_mean, body_std, device, chunk=4, one_per_condition=True):
    """One (start, goal) task per clip: start = the clip's own first frame; goal = the clip's own
    standardised steady-state Froude. `one_per_condition` keeps only the first clip seen per
    condition, so the train pool covers all 12 conditions without doubling encode cost on the
    train dir's 2-clips-per-condition duplicates."""
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz")))
    spec = REGISTRY[EMBODIMENT]
    seen_conditions = set()
    tasks = []
    for p in paths:
        clip = load(p, spec)
        with np.load(p, allow_pickle=True) as d:
            cond = str(d["condition"])
        if one_per_condition and cond in seen_conditions:
            continue
        seen_conditions.add(cond)
        e = encode_clip(encoder, clip["frames"], chunk).float().to(device)
        e0 = e[0]
        motion = np.asarray(clip["body_motion"])[:, [0, 1, 2]]
        steady = motion[len(motion) // 3: 2 * len(motion) // 3].mean(0)
        goal_raw = torch.tensor(steady, dtype=torch.float32, device=device)
        goal_std = (goal_raw - body_mean) / body_std
        tasks.append((e0, goal_std, cond))
    return tasks


def sample_batch(tasks, n, device):
    idx = np.random.randint(0, len(tasks), size=n)
    e0_batch = torch.stack([tasks[i][0] for i in idx])          # (n, tokens, dim)
    goal_batch = torch.stack([tasks[i][1] for i in idx])        # (n, body_dim)
    return e0_batch, goal_batch, idx


def rollout(actor, ftm, proj, md, e0_batch, goal_batch, horizon, device, deterministic=False):
    """`e0_batch`/`goal_batch` already carry the batch dimension -- one (start, goal) task per
    row, sampled independently, unlike v2 where the whole batch shared one fixed task."""
    e = e0_batch.clone()
    states, rewards = [], []
    for _ in range(horizon):
        state = pooled(e)
        states.append(state)
        action, _ = actor(state, goal_batch, deterministic=deterministic)
        z = proj(action, EMBODIMENT)
        pred_std = md.body(None, z)
        r = -(pred_std - goal_batch).abs().sum(-1)
        rewards.append(r)
        e = ftm(e, z, EMBODIMENT)
    return states, rewards, pooled(e)


@torch.no_grad()
def mc_return(actor, ftm, proj, md, e0, goal, horizon, gamma, device):
    """Real, non-bootstrapped discounted return for ONE task (e0, goal), current deterministic
    actor, no critic anywhere -- same role as v2's `mc_return`, now per-task."""
    _, rewards, _ = rollout(actor, ftm, proj, md, e0.unsqueeze(0), goal.unsqueeze(0),
                            horizon, device, deterministic=True)
    return sum((gamma ** k) * rewards[k].item() for k in range(horizon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt")
    ap.add_argument("--proj_ckpt", default="wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_clean.pt")
    ap.add_argument("--train_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--heldout_dir", default="data/egocentric/beh12_b1_ego_flat_cleanheldout")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--tasks_per_iter", type=int, default=4, help="how many (start,goal) tasks per training step")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target_tau", type=float, default=0.01)
    ap.add_argument("--actor_lr", type=float, default=1e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--anchor_every", type=int, default=200)
    ap.add_argument("--anchor_tasks", type=int, default=3, help="how many training tasks to anchor per anchor point")
    ap.add_argument("--anchor_steps", type=int, default=10)
    ap.add_argument("--out", default="results/wm/closed_loop/rl_imagination_isolation_v3.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, body_mean, body_std = load_frozen(args.ckpt, device)
    proj, act_mean, act_std = load_projector(args.proj_ckpt, cfg, device)
    action_dim = act_mean.shape[0]
    goal_dim = body_mean.shape[0]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    train_tasks = build_task_pool(encoder, args.train_dir, body_mean, body_std, device)
    heldout_tasks = build_task_pool(encoder, args.heldout_dir, body_mean, body_std, device,
                                    one_per_condition=False)
    del encoder
    torch.cuda.empty_cache()
    print(f"train tasks: {len(train_tasks)} ({[t[2] for t in train_tasks]})")
    print(f"heldout tasks: {len(heldout_tasks)} ({[t[2] for t in heldout_tasks]})")

    state_dim = cfg.token_dim
    actor = GaussianActor(state_dim, goal_dim, action_dim, act_mean, act_std).to(device)
    critic = Critic(state_dim, goal_dim).to(device)
    critic_target = Critic(state_dim, goal_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    return_scale = ReturnScale()

    log = {"iter": [], "train_return": [], "critic_loss": []}
    eval_log = {"iter": [], "train_mean": [], "heldout_mean": []}

    def eval_pool(tasks):
        vals = []
        for e0, goal, _cond in tasks:
            _, rewards, _ = rollout(actor, ftm, proj, md, e0.unsqueeze(0), goal.unsqueeze(0),
                                    args.horizon, device, deterministic=True)
            vals.append(sum((args.gamma ** k) * rewards[k].item() for k in range(args.horizon)))
        return float(np.mean(vals)), vals

    for it in range(args.iters):
        e0_batch, goal_batch, _ = sample_batch(train_tasks, args.tasks_per_iter, device)
        states, rewards, final_state = rollout(actor, ftm, proj, md, e0_batch, goal_batch,
                                               args.horizon, device, deterministic=False)

        with torch.no_grad():
            bootstrap = symexp(critic_target(final_state, goal_batch))
        returns = [None] * args.horizon
        running = bootstrap
        for k in reversed(range(args.horizon)):
            running = rewards[k] + args.gamma * running
            returns[k] = running

        critic_pred = torch.stack([critic(s.detach(), goal_batch) for s in states])
        real_targets = torch.stack(returns).detach()
        critic_loss = F.mse_loss(critic_pred, symlog(real_targets))
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        soft_update(critic_target, critic, args.target_tau)

        if args.anchor_every and it % args.anchor_every == 0:
            anchor_idx = np.random.choice(len(train_tasks),
                                          size=min(args.anchor_tasks, len(train_tasks)), replace=False)
            for i in anchor_idx:
                e0_i, goal_i, cond_i = train_tasks[i]
                mc_val = mc_return(actor, ftm, proj, md, e0_i, goal_i, args.horizon, args.gamma, device)
                target_t = symlog(torch.tensor(mc_val, device=device)).expand(1)
                state_i = pooled(e0_i).unsqueeze(0)
                goal_i_b = goal_i.unsqueeze(0)
                for _ in range(args.anchor_steps):
                    loss = F.mse_loss(critic(state_i, goal_i_b), target_t)
                    critic_opt.zero_grad()
                    loss.backward()
                    critic_opt.step()
                print(f"  [anchor @ {it}, {cond_i}] MC-return {mc_val:+.3f}  "
                      f"critic-after {symexp(critic(state_i, goal_i_b)).item():+.3f}")
            critic_target.load_state_dict(critic.state_dict())

        scale = return_scale.update(returns[0].detach())
        actor_loss = -(returns[0] / scale).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_opt.step()

        log["iter"].append(it)
        log["train_return"].append(float(returns[0].mean().item()))
        log["critic_loss"].append(float(critic_loss.item()))

        if it % args.eval_every == 0 or it == args.iters - 1:
            train_mean, _ = eval_pool(train_tasks)
            heldout_mean, heldout_vals = eval_pool(heldout_tasks)
            eval_log["iter"].append(it)
            eval_log["train_mean"].append(train_mean)
            eval_log["heldout_mean"].append(heldout_mean)
            print(f"iter {it:5d}  train_return {log['train_return'][-1]:+8.3f}  "
                  f"critic_loss {log['critic_loss'][-1]:8.3f}  "
                  f"eval_train_pool {train_mean:+8.3f}  eval_heldout_pool {heldout_mean:+8.3f}")

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    np.savez(os.path.join(ROOT, args.out),
             **{f"log_{k}": np.array(v) for k, v in log.items()},
             **{f"eval_{k}": np.array(v) for k, v in eval_log.items()})
    print(f"\nsaved: {args.out}")
    n = len(eval_log["iter"])
    print(f"train pool: first-quarter {np.mean(eval_log['train_mean'][:n//4]):+.3f}  "
          f"last-quarter {np.mean(eval_log['train_mean'][-n//4:]):+.3f}")
    print(f"heldout pool: first-quarter {np.mean(eval_log['heldout_mean'][:n//4]):+.3f}  "
          f"last-quarter {np.mean(eval_log['heldout_mean'][-n//4:]):+.3f}")


if __name__ == "__main__":
    main()
