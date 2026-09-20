"""Redo of F179's actor-critic-in-imagination isolation test, on the horizon F236 actually measured.

**Why this exists.** F179's whole arc (Slide 16) trained an actor-critic purely against a frozen
FTM's imagined rollouts, bootstrapping the critic's value over GAMMA=0.99's ~100-step effective
horizon. Every stabilization tried (symlog, return normalization, hard target updates, a two-hot
distributional critic) narrowed the gap but never closed it, and switching to PPO hit the identical
wall -- evidence the failure traces to the FTM's own rollout, not the critic algorithm (F179's
addenda). F236 then measured directly how far this specific FTM's auto-regressive rollout stays
worth trusting: its edge over a trivial "predict no change" baseline is already thin (ratio 0.80-0.84)
by k=20-30, on a checkpoint whose clips cap at 66 frames. GAMMA=0.99 was chosen by convention, never
checked against that curve. This script is the actual redo: same isolation-test shape (frozen FTM,
no re-grounding, a goal that cannot move), but with the imagination horizon set FROM that curve
(H=20 by default) instead of from an unchecked discount factor.

**What this does not carry over on purpose.** No symlog critic, no return normalization, no two-hot
distributional critic, no PPO. F179's own reading was that those are fixes for a problem this
horizon change might not have -- adding them before checking whether the plain, vanilla setup
converges at H=20 would re-confound exactly the question this script exists to isolate. If this
also fails to converge, the stabilization ladder is the next, separate step, run on top of this
file's own foundation.

**The original F179 script was never committed to git** (confirmed: `git log` on the `.npz` results
it produced shows no accompanying `.py` in the same commit) -- this is a fresh implementation, not a
patched copy, built from FINDINGS.md's own description of the isolation test's shape and this
project's already-fitted, already-frozen components (FTM, ITM's projector, the Cross-Body Head).

**The isolation test's own logic, restated so it's not lost in the code:**
  - environment = the frozen FTM only, imagined forward from one real embedding, never re-grounded
  - reward = -|body_head(proj(action)) - goal| in standardized Froude units, goal is a FIXED vector
    (a real B1 clip's own steady-state Froude, held constant -- "a target that provably cannot
    move", matching F179's own phrase) so nothing about the reward signal itself can drift
  - actor is a small Gaussian policy over pooled(e_t); critic is a small value head over the same
    pooled state; both operate purely on imagined trajectories, no real physics anywhere
  - a fixed evaluation state/goal pair, checked every EVAL_EVERY iterations with the actor's
    deterministic (mean) action, reports `realized_return` (actually summing imagined rewards along
    the eval rollout) against `value0` (the critic's belief before rolling) -- F179's own
    pass/fail read: these two should converge together if the critic and actor are doing real work.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/rl_imagination_isolation_v2.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt \\
        --proj_ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_clean.pt \\
        --horizon 20 --iters 5000
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
    """Mean over the token axis (second-to-last dim), the same pooling every diagnostic in this
    project uses to turn a full embedding into a fixed-size state vector. Works for a single
    embedding `(tokens, token_dim) -> (token_dim,)` and a batch `(batch, tokens, token_dim) ->
    (batch, token_dim)` alike, since the token axis is always second-to-last in both."""
    return e.mean(-2)


class GaussianActor(nn.Module):
    """pooled(e_t) -> a Gaussian over raw actions, tanh-squashed into a fixed multiple of the
    projector's own per-dim action std -- keeps proposals near the range the projector (and
    everything downstream of it) was actually fitted on, the caution F166 raised about actions
    leaving the fitted range applying just as much to an imagined action as a real one."""

    def __init__(self, state_dim, action_dim, action_mean, action_std, range_sigma=2.0, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        self.range_sigma = range_sigma

    def _bound(self, raw_mean):
        return self.action_mean + self.range_sigma * self.action_std * torch.tanh(raw_mean)

    def forward(self, state, deterministic=False):
        h = self.net(state)
        mean = self._bound(self.mean_head(h))
        if deterministic:
            return mean, None
        std = self.log_std.exp().expand_as(mean)
        eps = torch.randn_like(mean)
        action = mean + std * eps  # reparameterised: differentiable w.r.t. the actor's parameters
        log_prob = (-0.5 * ((action - mean) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def soft_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.expm1(x.abs()))


class ReturnScale:
    """DreamerV3-style running 5th/95th percentile scale for the actor's return signal --
    F179's rung (2) stabilization, applied here (untested combination) on top of F236's shorter
    horizon rather than F179's own long one."""

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


def load_frozen(ckpt_path, device):
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    ftm.load_state_dict(ck["ftm"])
    # This checkpoint never fit a per-joint action head for b1 (only `body_head`, which is all
    # this script reads via `md.body(None, z)`) -- the head dict below is a placeholder,
    # `strict=False` lets body_head load correctly regardless.
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
def pick_start_and_goal(encoder, data_dir, device, chunk=4):
    """One real branch embedding (a clip's first frame) as the fixed imagination start, and one
    real clip's own steady-state body-motion, standardised, as the fixed goal -- "a target that
    provably cannot move" is guaranteed by construction: the goal is read once, then frozen for
    the whole run."""
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz")))
    spec = REGISTRY[EMBODIMENT]
    start_clip = load(paths[0], spec)
    goal_clip = load(paths[1 % len(paths)], spec)

    e_start = encode_clip(encoder, start_clip["frames"], chunk).float().to(device)
    e0 = e_start[0]

    motion = np.asarray(goal_clip["body_motion"])[:, [0, 1, 2]]
    steady = motion[len(motion) // 3: 2 * len(motion) // 3].mean(0)  # steady-state window, avoids
    # the accel/decel edges the same way Slide 30's bug-3 fix does
    return e0, torch.tensor(steady, dtype=torch.float32, device=device)


MODE = "direct_const"     # direct_const | direct_tv | rollout_froude | embed
GOAL_SEQ = None           # (L, 3) standardised per-timestep Froude of the goal clip (time-varying modes)
E_GOAL = None             # (L+1, tokens, D) the goal clip's own embeddings, frame t0..t0+L


def load_goal_traj(encoder, data_dir, device, t0, length, body_mean, body_std, chunk=4):
    """Start AND goal from the SAME clip: e0 = frame t0, the goal is that clip's next `length` steps
    (its embeddings and its per-timestep body_motion), so the actor is asked to reproduce a whole
    real episode from a real start -- not to hit one constant number."""
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz")))
    clip = load(paths[1 % len(paths)], REGISTRY[EMBODIMENT])
    frames = clip["frames"][t0:t0 + length + 1]
    e_seq = encode_clip(encoder, frames, chunk).float().to(device)
    motion = np.asarray(clip["body_motion"])[t0:t0 + length, [0, 1, 2]]
    goal = (torch.tensor(motion, dtype=torch.float32, device=device) - body_mean) / body_std
    return e_seq[0], e_seq, goal


def rollout(actor, itm, ftm, proj, md, e0, goal_std, horizon, batch, device, deterministic=False,
            return_actions=False):
    """Roll `batch` imagined trajectories forward `horizon` steps from the SAME start `e0`
    (a single real embedding, `(tokens, token_dim)`), returns per-step rewards and pooled states
    for the lambda-return/critic-loss computation below. `e` stays the FULL token sequence
    throughout -- FTM and the projector's downstream consumers need it -- and is only pooled when
    handed to the actor/critic, which read a compact state, not the raw tokens."""
    e = e0.unsqueeze(0).expand(batch, -1, -1).clone()   # (batch, tokens, token_dim)
    states, rewards, actions = [], [], []
    for k in range(horizon):
        state = pooled(e)                                # (batch, token_dim)
        states.append(state)
        action, _ = actor(state, deterministic=deterministic)
        actions.append(action)
        z = proj(action, EMBODIMENT)                      # (batch, z_dim)
        e_next = ftm(e, z, EMBODIMENT)                    # (batch, tokens, token_dim)
        if MODE == "direct_const":                        # the original test: reward never sees the FTM
            r = -(md.body(None, z) - goal_std).abs().sum(-1)
        elif MODE == "direct_tv":                         # same readout, goal changes every step
            r = -(md.body(None, z) - GOAL_SEQ[k]).abs().sum(-1)
        elif MODE == "rollout_froude":                    # FTM's imagined next frame -> ITM -> head
            r = -(md.body(None, itm(e, e_next)) - GOAL_SEQ[k]).abs().sum(-1)
        elif MODE == "embed":                             # FTM's imagined next frame vs the goal clip's
            r = -(e_next - E_GOAL[k + 1]).abs().mean((-1, -2))
        else:
            raise ValueError(MODE)
        rewards.append(r)
        e = e_next
    if return_actions:
        return states, rewards, pooled(e), torch.stack(actions)
    return states, rewards, pooled(e)


@torch.no_grad()
def mc_return(actor, itm, ftm, proj, md, e0, goal_std, horizon, gamma, device):
    """A real, non-bootstrapped discounted return at `e0` under the CURRENT deterministic actor --
    no critic anywhere in this computation. `horizon` should be long enough that `gamma**horizon`
    is negligible (F179's own MC-check used 800 steps for GAMMA=0.99; this project's clips cap
    around 66 frames, and `--mc_horizon 60` keeps `gamma**60` ~= 0.55 at 0.99, not fully negligible
    but the only budget this checkpoint's own rollout fidelity (F236) supports before the FTM's
    predictions stop meaning much anyway -- stated as a real limitation, not hidden)."""
    _, rewards, _ = rollout(actor, itm, ftm, proj, md, e0, goal_std, horizon, 1, device,
                            deterministic=True)
    return sum((gamma ** k) * rewards[k].item() for k in range(horizon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt")
    ap.add_argument("--proj_ckpt", default="wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_clean.pt")
    ap.add_argument("--data_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--horizon", type=int, default=20, help="F236's recommended 15-25 range")
    ap.add_argument("--iters", type=int, default=5000, help="matches F179's own budget")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target_tau", type=float, default=0.01)
    ap.add_argument("--actor_lr", type=float, default=1e-4)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--stabilize", action="store_true",
                    help="F179's rung-2 fix (symlog critic + return normalization), combined here "
                         "with F236's shorter horizon for the first time -- neither alone was "
                         "tested at the other's setting before this script")
    ap.add_argument("--anchor_every", type=int, default=0,
                    help="F237's own next step: every N iterations, regress the critic directly "
                         "toward a long, non-bootstrapped Monte Carlo return at e0 (--mc_horizon "
                         "steps, real reward only, no critic in the loop), then hard-copy the "
                         "target network from it -- breaks the self-referential bootstrap loop "
                         "periodically instead of letting the EMA target drift indefinitely. "
                         "0 disables (the untouched F237 behaviour).")
    ap.add_argument("--mc_horizon", type=int, default=60, help="long enough that gamma**H is negligible")
    ap.add_argument("--anchor_steps", type=int, default=20, help="extra critic gradient steps per anchor")
    ap.add_argument("--out", default="results/wm/closed_loop/rl_imagination_isolation_v2.npz")
    ap.add_argument("--mode", default="direct_const",
                    choices=["direct_const", "direct_tv", "rollout_froude", "embed"],
                    help="reward: direct_const = the original (constant goal, reward never touches the "
                         "FTM); direct_tv = same readout, per-timestep goal; rollout_froude = Froude read "
                         "from the FTM-imagined transition; embed = distance to the goal clip's own "
                         "embedding")
    ap.add_argument("--t0", type=int, default=5, help="start frame of the goal clip (time-varying modes)")
    ap.add_argument("--save_actor", default="",
                    help="write the trained actor (+ the fixed start embedding and goal it was trained "
                         "on) here, so it can be run outside imagination (real physics)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, body_mean, body_std = load_frozen(args.ckpt, device)
    proj, act_mean, act_std = load_projector(args.proj_ckpt, cfg, device)
    action_dim = act_mean.shape[0]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    global MODE, GOAL_SEQ, E_GOAL
    MODE = args.mode
    if MODE == "direct_const":
        e0, goal_raw = pick_start_and_goal(encoder, args.data_dir, device)
        goal_std = (goal_raw - body_mean) / body_std
    else:
        length = max(args.horizon, args.mc_horizon)
        e0, E_GOAL, GOAL_SEQ = load_goal_traj(encoder, args.data_dir, device, args.t0, length,
                                              body_mean, body_std)
        goal_std, goal_raw = GOAL_SEQ[0], GOAL_SEQ[0] * body_std + body_mean
    del encoder
    torch.cuda.empty_cache()

    state_dim = cfg.token_dim
    actor = GaussianActor(state_dim, action_dim, act_mean, act_std).to(device)
    critic = Critic(state_dim).to(device)
    critic_target = Critic(state_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    for p in critic_target.parameters():
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    return_scale = ReturnScale() if args.stabilize else None

    log_iter, log_return, log_actor_loss, log_critic_loss = [], [], [], []
    eval_iter, eval_realized, eval_value0 = [], [], []

    for it in range(args.iters):
        states, rewards, final_state = rollout(
            actor, itm, ftm, proj, md, e0, goal_std,
            args.horizon, args.batch, device, deterministic=False)

        with torch.no_grad():
            # critic_target's own output lives in symlog space when --stabilize is on (that is
            # what it is trained to predict, below) -- decode before using it in the REAL-space
            # return recursion, exactly F179's "decoded via symexp for the lambda-return
            # recursion" description.
            bootstrap = symexp(critic_target(final_state)) if args.stabilize else critic_target(final_state)
        returns = [None] * args.horizon
        running = bootstrap
        for k in reversed(range(args.horizon)):
            running = rewards[k] + args.gamma * running
            returns[k] = running

        critic_pred = torch.stack([critic(s.detach()) for s in states])
        real_targets = torch.stack(returns).detach()
        critic_target_vals = symlog(real_targets) if args.stabilize else real_targets
        critic_loss = F.mse_loss(critic_pred, critic_target_vals)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        soft_update(critic_target, critic, args.target_tau)

        if args.anchor_every and it % args.anchor_every == 0:
            anchor_return = mc_return(actor, itm, ftm, proj, md, e0, goal_std,
                                      args.mc_horizon, args.gamma, device)
            anchor_target = torch.tensor(anchor_return, device=device)
            anchor_target_t = symlog(anchor_target) if args.stabilize else anchor_target
            e0_state = pooled(e0).unsqueeze(0)
            for _ in range(args.anchor_steps):
                anchor_loss = F.mse_loss(critic(e0_state), anchor_target_t.expand(1))
                critic_opt.zero_grad()
                anchor_loss.backward()
                critic_opt.step()
            critic_target.load_state_dict(critic.state_dict())  # hard reset, breaks the drift
            print(f"  [anchor @ {it}] MC-return {anchor_return:+.3f}  "
                  f"critic(e0) before->after: -> {critic(e0_state).item():+.3f}")

        if args.stabilize:
            scale = return_scale.update(returns[0].detach())
            actor_loss = -(returns[0] / scale).mean()
        else:
            actor_loss = -returns[0].mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        actor_opt.step()

        log_iter.append(it)
        log_return.append(float(returns[0].mean().item()))
        log_actor_loss.append(float(actor_loss.item()))
        log_critic_loss.append(float(critic_loss.item()))

        if it % args.eval_every == 0 or it == args.iters - 1:
            with torch.no_grad():
                raw_value0 = critic(pooled(e0).unsqueeze(0))
                value0 = symexp(raw_value0).item() if args.stabilize else raw_value0.item()
                _, eval_rewards, _ = rollout(
                    actor, itm, ftm, proj, md, e0, goal_std,
                    args.horizon, 1, device, deterministic=True)
                discounted = sum((args.gamma ** k) * eval_rewards[k].item() for k in range(args.horizon))
            eval_iter.append(it)
            eval_realized.append(discounted)
            eval_value0.append(value0)
            print(f"iter {it:5d}  train_return {log_return[-1]:+8.3f}  "
                  f"critic_loss {log_critic_loss[-1]:8.3f}  "
                  f"eval_realized {discounted:+8.3f}  eval_value0 {value0:+8.3f}  "
                  f"gap {value0 - discounted:+8.3f}")

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    np.savez(os.path.join(ROOT, args.out),
             log_iter=np.array(log_iter), log_return=np.array(log_return),
             log_actor_loss=np.array(log_actor_loss), log_critic_loss=np.array(log_critic_loss),
             eval_iter=np.array(eval_iter), eval_realized=np.array(eval_realized),
             eval_value0=np.array(eval_value0), horizon=args.horizon)
    print(f"\nsaved: {args.out}")
    # The one number comparable ACROSS modes (each mode's own reward has its own scale): the actor's
    # deterministic imagined rollout, every step read through the direct Froude head, against the goal
    # clip's per-timestep Froude, in real (un-standardised) Froude units.
    with torch.no_grad():
        _, _, _, acts = rollout(actor, itm, ftm, proj, md, e0, goal_std, args.horizon, 1, device,
                                deterministic=True, return_actions=True)
        pred = md.body(None, proj(acts[:, 0], EMBODIMENT))            # (H, 3) standardised
        if MODE == "direct_const":
            goal_seq_eval = goal_std.expand_as(pred)
        else:
            goal_seq_eval = GOAL_SEQ[:args.horizon]
        common = ((pred - goal_seq_eval) * body_std).abs().sum(-1).mean().item()
    print(f"COMMON METRIC (direct head vs per-step goal, real Froude L1, mean over {args.horizon} steps): "
          f"{common:.4f}")
    if args.save_actor:
        torch.save({"actor": actor.state_dict(), "e0": e0.cpu(), "goal_std_seq": goal_seq_eval.cpu(),
                    "actions": acts[:, 0].cpu(), "mode": MODE, "t0": args.t0,
                    "body_mean": body_mean.cpu() if hasattr(body_mean, "cpu") else body_mean,
                    "body_std": body_std.cpu() if hasattr(body_std, "cpu") else body_std,
                    "act_mean": act_mean.cpu(), "act_std": act_std.cpu(), "horizon": args.horizon,
                    "common_metric": common}, os.path.join(ROOT, args.save_actor))
        print(f"actor saved: {args.save_actor}")
    print(f"horizon={args.horizon}  first-quarter mean realized "
          f"{np.mean(eval_realized[:len(eval_realized)//4]):+.3f}  "
          f"last-quarter mean realized {np.mean(eval_realized[-len(eval_realized)//4:]):+.3f}")


if __name__ == "__main__":
    main()
