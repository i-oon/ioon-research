"""Standard PPO, real MuJoCo physics, reward from the world model -- Q21 step 3's actual
mechanism. Not the thing F179 killed: no FTM rollout, no imagined trajectory, no bootstrapped
value function through a compounding model.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/b1_real_ppo.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1.pt \\
        --updates 5

**Design choices, and why, checked against this project's own history before writing any of it.**

Hyperparameters (gamma 0.99, GAE lambda 0.95, clip 0.2, lr 3e-4, epochs 10, minibatch 64, hidden
(64,64) Tanh, grad-norm clip 10) match `doc/ref/airl-insect-walking/algorithms/ppo.py`, a working
legged-locomotion PPO already in this repo, not invented from scratch.

**What F179 diagnosed does NOT apply here, and why.** F179's critic bootstrapped a value function
through ~100-step FTM-imagined rollouts, and died because the FTM's own compounding prediction
error polluted the value target -- a genuine value-learning/function-approximation problem,
confirmed across two unrelated algorithms (a Dreamer-style critic ladder and PPO itself). Here, the
critic bootstraps over REAL physics trajectories; the world model is consulted only once per step,
to score the action just taken, never rolled forward. There is no compounding rollout for a value
function to be corrupted by. The DreamerV3-grade remedies that arc needed (two-hot distributional
critic, symlog return normalization) are not carried over here -- they were fixes for a failure
mode specific to imagination, not general PPO hygiene, and adding them without the problem they
solve would just be unjustified complexity.

**What F179 DID find that transfers directly: the myopia trap.** A GAMMA=0.95 (short-horizon) PPO
run there converged cleanly on its return curve while the policy had actually collapsed into a
frozen, degenerate stance (action variation down ~92%, first half of rollout vs second half) --
looking successful on the metric everyone was watching while doing nothing real. This script tracks
the identical diagnostic (`action_std_early` vs `action_std_late` per training checkpoint) and
prints it every log interval, not only at the end, so this can be caught during training rather
than after wasting a full run.

**Reward is `body_head(proj(action))` vs a fixed goal, never rolled through the FTM** -- see
`b1_mujoco_env.py`'s own docstring for why this is the deliberate choice (validates the
reward the deployed system would actually have, not a ground-truth shortcut).

**Validate short before committing long**, per this session's own repeated discipline (F199's
config bug, this arc's stage-1 erosion bug -- neither would have been caught without checking a
short run first): `--updates` defaults small; only raise it after a short run's diagnostics look
sane (rewards finite, `action_std` not collapsing, `died_frac` not near 1.0 or 0.0).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_mujoco_env import VecB1MuJoCoEnv, OBS_DIM, ACTION_DIM  # noqa: E402
from b1_coppelia_env import B1CoppeliaEnv  # noqa: E402

HIDDEN = (64, 64)


def mlp(in_dim, out_dim, hidden=HIDDEN):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy, tanh-squashed to bound actions to [-1, 1] -- the env clips there anyway,
    but an unsquashed Gaussian gives PPO no reason not to propose arbitrarily large raw actions."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mean = mlp(obs_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs):
        mu = self.mean(obs)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def sample(self, obs, eps=None):
        mu, std = self(obs)
        if eps is None:
            eps = torch.randn_like(mu)
        u = mu + std * eps                      # pre-squash sample
        a = torch.tanh(u)
        # log_prob with the standard tanh-squash correction, numerically stable form
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_prob, u

    def log_prob_of(self, obs, u):
        mu, std = self(obs)
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        entropy = (std.log() + 0.5 * np.log(2 * np.pi * np.e)).sum(-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v = mlp(obs_dim, 1)

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class RunningMeanStd:
    """Welford's online algorithm -- a running mean/variance estimate that updates one sample (or
    one batch) at a time, never storing the full history."""

    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        batch_mean, batch_var, batch_count = np.mean(x), np.var(x), len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean, self.var, self.count = new_mean, m2 / tot_count, tot_count


class RewardNormalizer:
    """The standard PPO/Gym reward-scaling trick (OpenAI Gym's `NormalizeReward` wrapper, SB3's
    `VecNormalize`): track a running estimate of the DISCOUNTED RETURN's variance (not the raw
    reward's), divide each incoming reward by that running std. Deliberately does not subtract the
    mean -- that would change the sign/sparsity structure of a reward that's meant to stay
    interpretable (e.g. the fall penalty should stay clearly negative). This only rescales
    magnitude so the critic's regression target has a stable scale across training, independent of
    the raw reward's own units."""

    def __init__(self, gamma, eps=1e-8):
        self.gamma = gamma
        self.eps = eps
        self.ret = 0.0
        self.rms = RunningMeanStd()

    def normalize(self, reward, done):
        self.ret = self.ret * self.gamma + reward
        self.rms.update(np.array([self.ret]))
        normalized = reward / (np.sqrt(self.rms.var) + self.eps)
        if done:
            self.ret = 0.0
        return float(normalized)


# AR(1)-filtered exploration noise: i.i.d. per-step noise on 12 joints essentially never samples a
# multi-step coordinated pattern (measured: tracking stayed flat across three independent fixes to
# the reward/env). This correlates noise across ~0.4s (a fraction of one stride, BODY_WINDOW_S=1.0
# elsewhere in this project), giving continuous movement attempts instead of per-step jitter that
# averages to ~0 net displacement. Marginal variance is unchanged (still N(mu, std) at every step,
# so PPO's log_prob/ratio math is untouched) -- only how successive steps correlate changes.
OU_ALPHA = 0.95


def collect_rollout(env, actor, critic, steps, obs_mean, obs_std, device, reward_norm=None):
    obs_buf, act_buf, u_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], [], []
    ep_returns, ep_lens, died = [], [], []
    action_stds_first_half, action_stds_second_half = [], []
    true_fwd = []           # measured forward speed, from physics -- NOT the reward being trained
                            # on, so a reward the policy has learned to exploit cannot move it
    tracking_rewards = []   # the raw exp(-distance) term alone -- separates "tracking well" from
                            # "just surviving on the alive bonus" in the printed diagnostics

    obs = env.reset()
    prev_eps = None
    ep_ret, ep_len, ep_actions = 0.0, 0, []
    for t in range(steps):
        obs_n = (obs - obs_mean) / obs_std
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            raw = torch.randn(1, actor.log_std.numel(), device=device)
            eps = raw if prev_eps is None else OU_ALPHA * prev_eps + (1 - OU_ALPHA ** 2) ** 0.5 * raw
            prev_eps = eps
            a, logp, u = actor.sample(obs_t, eps=eps)
            v = critic(obs_t)
        action = a.squeeze(0).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        # `ep_ret`/diagnostics stay in raw reward units (interpretable); only what feeds GAE/PPO
        # is normalized, so the printed mean_return means the same thing across every update.
        train_reward = reward_norm.normalize(reward, done) if reward_norm is not None else reward

        obs_buf.append(obs_n); act_buf.append(action); u_buf.append(u.squeeze(0).cpu().numpy())
        logp_buf.append(logp.item()); rew_buf.append(train_reward); val_buf.append(v.item())
        done_buf.append(float(done))

        ep_ret += reward; ep_len += 1; ep_actions.append(action)
        tracking_rewards.append(info["tracking_reward"])
        if info.get("true_froude") is not None:
            true_fwd.append(float(info["true_froude"][0]))
        obs = next_obs
        if done:
            ep_returns.append(ep_ret); ep_lens.append(ep_len); died.append(float(info["fell"]))
            half = max(1, len(ep_actions) // 2)
            action_stds_first_half.append(float(np.std(ep_actions[:half])))
            action_stds_second_half.append(float(np.std(ep_actions[half:]) if len(ep_actions) > half else 0.0))
            obs = env.reset()
            prev_eps = None
            ep_ret, ep_len, ep_actions = 0.0, 0, []

    with torch.no_grad():
        obs_n = (obs - obs_mean) / obs_std
        last_val = critic(torch.as_tensor(obs_n, dtype=torch.float32, device=device).unsqueeze(0)).item()

    data = {
        "obs": np.stack(obs_buf), "act": np.stack(act_buf), "u": np.stack(u_buf),
        "logp": np.array(logp_buf), "rew": np.array(rew_buf), "val": np.array(val_buf),
        "done": np.array(done_buf),
    }
    diag = {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "mean_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "died_frac": float(np.mean(died)) if died else float("nan"),
        "action_std_early": float(np.mean(action_stds_first_half)) if action_stds_first_half else float("nan"),
        "action_std_late": float(np.mean(action_stds_second_half)) if action_stds_second_half else float("nan"),
        "mean_tracking_reward": float(np.mean(tracking_rewards)),  # in (0, 1] -- 1.0 = perfect
        "mean_true_fwd": float(np.mean(true_fwd)) if true_fwd else float("nan"),
        "n_episodes": len(ep_returns), "last_val": last_val,
    }
    return data, diag


def gae(rew, val, done, last_val, gamma, lam, normalize=True):
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    next_val = last_val
    gae_t = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + gamma * next_val * nonterminal - val[t]
        gae_t = delta + gamma * lam * nonterminal * gae_t
        adv[t] = gae_t
        next_val = val[t]
    ret = adv + val
    if normalize:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def collect_rollout_vec(vec_env, actor, critic, steps_per_env, obs_mean, obs_std, device,
                        reward_norms, gamma, lam):
    """Same rollout as `collect_rollout`, `n_envs` MuJoCo processes at once. Each env's trajectory
    stays temporally contiguous (needed for GAE's backward recursion) and gets its own bootstrap
    value/return normalizer; only the resulting (obs, u, logp, adv, ret) tensors are pooled across
    envs for the PPO update."""
    n = vec_env.n
    act_dim = actor.log_std.numel()
    per_env = [{"obs": [], "u": [], "logp": [], "val": [], "rew": [], "done": []} for _ in range(n)]
    ep_returns, ep_lens, died = [], [], []
    action_stds_first_half, action_stds_second_half = [], []
    tracking_rewards = []
    true_fwd = []           # see collect_rollout's note: measured, not the trained reward

    obs = vec_env.reset()
    prev_eps = torch.zeros(n, act_dim, device=device)
    fresh = np.ones(n, dtype=bool)   # True right after a reset -- next eps must not correlate with
                                      # the previous (different) episode's noise
    ep_ret = np.zeros(n); ep_len = np.zeros(n, dtype=int)
    ep_actions = [[] for _ in range(n)]

    for t in range(steps_per_env):
        obs_n = (obs - obs_mean) / obs_std
        obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw = torch.randn(n, act_dim, device=device)
            fresh_t = torch.as_tensor(fresh, device=device).unsqueeze(1)
            eps = torch.where(fresh_t, raw, OU_ALPHA * prev_eps + (1 - OU_ALPHA ** 2) ** 0.5 * raw)
            prev_eps = eps
            a, logp, u = actor.sample(obs_t, eps=eps)
            v = critic(obs_t)
        fresh[:] = False
        actions_np = a.cpu().numpy()
        next_obs, reward, done, infos = vec_env.step(actions_np)

        for i in range(n):
            b = per_env[i]
            b["obs"].append(obs_n[i]); b["u"].append(u[i].cpu().numpy())
            b["logp"].append(logp[i].item()); b["val"].append(v[i].item())
            b["rew"].append(reward_norms[i].normalize(float(reward[i]), bool(done[i])))
            b["done"].append(float(done[i]))
            ep_ret[i] += reward[i]; ep_len[i] += 1; ep_actions[i].append(actions_np[i])
            tracking_rewards.append(infos[i]["tracking_reward"])
            if infos[i].get("true_froude") is not None:
                true_fwd.append(float(infos[i]["true_froude"][0]))
            if done[i]:
                ep_returns.append(ep_ret[i]); ep_lens.append(ep_len[i]); died.append(float(infos[i]["fell"]))
                half = max(1, len(ep_actions[i]) // 2)
                action_stds_first_half.append(float(np.std(ep_actions[i][:half])))
                action_stds_second_half.append(
                    float(np.std(ep_actions[i][half:]) if len(ep_actions[i]) > half else 0.0))
                ep_ret[i] = 0.0; ep_len[i] = 0; ep_actions[i] = []
                fresh[i] = True
        obs = next_obs

    with torch.no_grad():
        obs_n = (obs - obs_mean) / obs_std
        last_vals = critic(torch.as_tensor(obs_n, dtype=torch.float32, device=device)).cpu().numpy()

    obs_all, u_all, logp_all, adv_all, ret_all = [], [], [], [], []
    for i in range(n):
        b = per_env[i]
        adv_i, ret_i = gae(np.array(b["rew"]), np.array(b["val"]), np.array(b["done"]),
                          float(last_vals[i]), gamma, lam, normalize=False)
        obs_all.append(np.stack(b["obs"])); u_all.append(np.stack(b["u"]))
        logp_all.append(np.array(b["logp"])); adv_all.append(adv_i); ret_all.append(ret_i)

    adv = np.concatenate(adv_all)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # one global normalization, not per env
    data = {
        "obs": np.concatenate(obs_all), "u": np.concatenate(u_all),
        "logp": np.concatenate(logp_all), "adv": adv, "ret": np.concatenate(ret_all),
    }
    diag = {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else float("nan"),
        "mean_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "died_frac": float(np.mean(died)) if died else float("nan"),
        "action_std_early": float(np.mean(action_stds_first_half)) if action_stds_first_half else float("nan"),
        "action_std_late": float(np.mean(action_stds_second_half)) if action_stds_second_half else float("nan"),
        "mean_tracking_reward": float(np.mean(tracking_rewards)),
        "mean_true_fwd": float(np.mean(true_fwd)) if true_fwd else float("nan"),
        "n_episodes": len(ep_returns),
    }
    return data, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--updates", type=int, default=5, help="small by default -- validate before raising")
    ap.add_argument("--rollout_steps", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=10.0)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--engine", choices=("mujoco", "coppelia"), default="mujoco",
                    help="mujoco = b1_mujoco_env.py (default, validated). coppelia = "
                         "b1_coppelia_env.py, trains natively in CoppeliaSim-Bullet -- slower per "
                         "step (remote-API round trip), and only mechanically smoke-tested, not "
                         "yet run through PPO. Both share the identical observation/reward/step "
                         "interface, so this is the only thing that needs to change to switch.")
    ap.add_argument("--coppelia_port", type=int, default=23000)
    ap.add_argument("--reward_mode", choices=("wm", "true_froude"), default="wm",
                    help="true_froude is a diagnostic: rewards real measured base velocity instead "
                         "of the WM's action-only estimate, to isolate whether the RL/env stack can "
                         "learn to walk at all. mujoco engine only.")
    ap.add_argument("--n_envs", type=int, default=16,
                    help="parallel MuJoCo environments (subprocesses). mujoco engine only -- a "
                         "single env cannot collect enough real steps to discover a coordinated "
                         "gait (the reference Isaac Lab B1 policy used 4096 parallel envs).")
    ap.add_argument("--curriculum_updates", type=int, default=0,
                    help="ramp the goal from near-zero to full difficulty linearly over this many "
                         "updates (0 = disabled, full difficulty from update 0, the current "
                         "behavior). The reference Isaac Lab task keeps most environments' commanded "
                         "velocity nonzero but starts small and grows via its own curriculum "
                         "(lin_vel_cmd_levels) -- this is the same idea applied to our single fixed "
                         "goal: an easy near-zero target early gives a cold-start policy an "
                         "achievable win instead of one fixed, moderately-hard target from step 0.")
    ap.add_argument("--save_every", type=int, default=25,
                    help="also write a snapshot every N updates, and keep a separate best-so-far "
                         "checkpoint selected on MEASURED forward speed (never on the reward being "
                         "trained, which in wm mode is the model grading itself and is exploitable). "
                         "0 disables both.")
    ap.add_argument("--resume", default="",
                    help="continue training from a prior run's --out checkpoint (actor/critic "
                         "weights and obs_mean/obs_std), instead of a fresh random init. Skips the "
                         "observation-normalization estimation step too.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vectorized = args.engine == "mujoco"
    if vectorized:
        env = VecB1MuJoCoEnv(args.ckpt, args.n_envs, horizon=args.horizon, reward_mode=args.reward_mode)
    else:
        env = B1CoppeliaEnv(args.ckpt, horizon=args.horizon, port=args.coppelia_port)
    actor = Actor(OBS_DIM, ACTION_DIM).to(device)
    critic = Critic(OBS_DIM).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    if args.resume:
        saved = torch.load(os.path.join(ROOT, args.resume), map_location=device, weights_only=False)
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        obs_mean, obs_std = saved["obs_mean"], saved["obs_std"]
        print(f"resumed from {args.resume}\n")
    else:
        # Observation normalization stats, from a short random rollout -- proprioceptive channels
        # (joint pos/vel, base quat/vel) sit on very different scales, and PPO trains far more
        # reliably when its input is roughly unit-scaled.
        print("estimating observation normalization from a short random rollout...")
        tmp_obs = []
        if vectorized:
            env.reset()
            for _ in range(max(1, 1000 // args.n_envs)):
                o2, _, _, _ = env.step(np.random.normal(0, 0.1, size=(args.n_envs, ACTION_DIM)))
                tmp_obs.append(o2)
            tmp_obs = np.concatenate(tmp_obs, axis=0)
        else:
            o = env.reset()
            for _ in range(1000):
                o2, _, d, _ = env.step(np.random.normal(0, 0.1, size=ACTION_DIM))
                tmp_obs.append(o2)
                o = env.reset() if d else o2
            tmp_obs = np.stack(tmp_obs)
        obs_mean = tmp_obs.mean(axis=0)
        obs_std = tmp_obs.std(axis=0) + 1e-6
        print(f"obs_mean range [{obs_mean.min():.2f}, {obs_mean.max():.2f}]  "
             f"obs_std range [{obs_std.min():.3f}, {obs_std.max():.2f}]\n")

    reward_norm = ([RewardNormalizer(gamma=args.gamma) for _ in range(args.n_envs)] if vectorized
                   else RewardNormalizer(gamma=args.gamma))

    if vectorized:
        # Below this, no episode reaches `horizon` within a single update's rollout (falls no
        # longer end episodes early) -- mean_return/mean_len/died_frac/act_std all depend on a
        # completed episode and silently go NaN, even though training itself still proceeds (GAE
        # bootstraps fine without one). At least 2 full episodes per env keeps them populated.
        min_steps_per_env = 2 * args.horizon
        if args.rollout_steps // args.n_envs < min_steps_per_env:
            print(f"--rollout_steps {args.rollout_steps} / --n_envs {args.n_envs} = "
                 f"{args.rollout_steps // args.n_envs} steps/env, below 2*horizon="
                 f"{min_steps_per_env} -- raising rollout_steps to "
                 f"{min_steps_per_env * args.n_envs} so episode diagnostics aren't empty.")
            args.rollout_steps = min_steps_per_env * args.n_envs

    print(f"{'update':>6}{'mean_return':>14}{'mean_len':>10}{'died_frac':>11}{'tracking':>10}"
         f"{'true_fwd':>10}{'act_std_early':>15}{'act_std_late':>13}{'pi_loss':>10}{'v_loss':>9}"
         f"{'entropy':>9}")
    print("  'tracking' is the reward being optimised (the world model's own opinion in wm mode);\n"
         "  'true_fwd' is measured forward speed from physics and is NOT what the policy optimises,\n"
         "  so a reward the policy has learned to exploit cannot move it. Watch them diverge.")

    def snapshot(path):
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                   "obs_mean": obs_mean, "obs_std": obs_std, "args": vars(args)},
                  os.path.join(ROOT, path) if not os.path.isabs(path) else path)

    stem = (args.out[:-3] if args.out.endswith(".pt") else args.out) or ""
    best_true_fwd = -float("inf")
    for update in range(args.updates):
        if args.curriculum_updates > 0:
            scale = min(1.0, (update + 1) / args.curriculum_updates)
            env.set_goal_scale(scale)
        if vectorized:
            steps_per_env = max(1, args.rollout_steps // args.n_envs)
            data, diag = collect_rollout_vec(env, actor, critic, steps_per_env, obs_mean, obs_std,
                                             device, reward_norm, args.gamma, args.lam)
            adv, ret = data["adv"], data["ret"]
        else:
            data, diag = collect_rollout(env, actor, critic, args.rollout_steps, obs_mean, obs_std,
                                         device, reward_norm=reward_norm)
            adv, ret = gae(data["rew"], data["val"], data["done"], diag["last_val"], args.gamma, args.lam)

        obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=device)
        u_t = torch.as_tensor(data["u"], dtype=torch.float32, device=device)
        logp_old_t = torch.as_tensor(data["logp"], dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=device)

        n = len(obs_t)
        last_pi_loss = last_v_loss = last_ent = 0.0
        for _ in range(args.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, args.minibatch):
                mb = idx[start:start + args.minibatch]
                logp, ent = actor.log_prob_of(obs_t[mb], u_t[mb])
                ratio = (logp - logp_old_t[mb]).exp()
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * ent.mean()
                opt_a.zero_grad(); pi_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                opt_a.step()

                v = critic(obs_t[mb])
                v_loss = F.mse_loss(v, ret_t[mb])
                opt_c.zero_grad(); v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                opt_c.step()

                last_pi_loss, last_v_loss, last_ent = pi_loss.item(), v_loss.item(), ent.mean().item()

        print(f"{update:>6}{diag['mean_return']:>14.2f}{diag['mean_len']:>10.1f}"
             f"{diag['died_frac']:>11.2f}{diag['mean_tracking_reward']:>10.4f}"
             f"{diag['mean_true_fwd']:>10.4f}"
             f"{diag['action_std_early']:>15.4f}"
             f"{diag['action_std_late']:>13.4f}{last_pi_loss:>10.4f}{last_v_loss:>9.4f}"
             f"{last_ent:>9.3f}", flush=True)

        # Selection on MEASURED speed, never on the optimised reward: in wm mode that reward is the
        # model grading its own prediction, so a policy that has found its blind spot scores high
        # while standing still. Training is also not monotonic (a run has been observed peaking,
        # regressing by half, and partially recovering), so the final save is not reliably the best.
        if stem and args.save_every > 0:
            if diag["mean_true_fwd"] == diag["mean_true_fwd"] and diag["mean_true_fwd"] > best_true_fwd:
                best_true_fwd = diag["mean_true_fwd"]
                snapshot(f"{stem}_best.pt")
            if (update + 1) % args.save_every == 0:
                snapshot(f"{stem}_u{update + 1}.pt")

    if stem and args.save_every > 0 and best_true_fwd > -float("inf"):
        print(f"-> {stem}_best.pt  (best measured forward speed {best_true_fwd:.4f})")

    if args.out:
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                   "obs_mean": obs_mean, "obs_std": obs_std, "args": vars(args)}, args.out)
        print(f"-> {args.out}")

    print("\n`action_std_early` vs `action_std_late` collapsing toward 0 together with a rising "
         "`mean_return` is the myopia-trap signature F179 found (a frozen, degenerate stance that "
         "fakes a good return) -- check this before trusting any improving return curve.")

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
