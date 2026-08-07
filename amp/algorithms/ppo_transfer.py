'''
This PPO implementation is modified for transfer learning using a pre-trained AIRL discriminator reward network g(s).
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from common.buffer import RolloutBuffer
from common.base import Algorithm
from networks.actor import ActorNetworkPolicy
from networks.critic import CriticNetworkPolicy
from networks.discrim import AIRLDiscrim
import os


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)
    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):
    def __init__(self,
                 state_shape,
                 action_shape,
                 device,
                 seed,
                 gamma=0.995,
                 rollout_length=2048,
                 mix_buffer=20,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 units_actor=(64, 64),
                 units_critic=(64, 64),
                 epoch_ppo=10,
                 clip_eps=0.2,
                 lambd=0.97,
                 coef_ent=0.0,
                 max_grad_norm=10.0,
                 mini_batch_size=64,
                 lam_min=1.0,
                 lam_max=1.0,
                 lam_warmup=0,
                 lam_ramp=1,
                 g_clip=None,
                 g_center=True,
                 g_baseline=0.0,
                 disc_state_dim=None,
                 critic_context_dim=0,
                 reward_scale=1.0,
                 g_ood_coef=0.0,
                 ):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        self.device = device
        self.critic_context_dim = int(critic_context_dim)
        self.critic_state_shape = (state_shape[0] + self.critic_context_dim,)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
            critic_state_shape=self.critic_state_shape,
        )

        # Actor
        self.actor = ActorNetworkPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # actor_path = 'logs/RedMirror_66k_aug3c/ppo-transfer/20251115-1605/model/step250000/actor.pth'
        # self.actor.load_state_dict(torch.load(actor_path, weights_only=True, map_location=device))

        # Critic
        self.critic = CriticNetworkPolicy(
            state_shape=self.critic_state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        # critic_path = 'logs/RedMirror_66k_aug3c/ppo-transfer/20251115-1605/model/step250000/critic.pth'
        # self.critic.load_state_dict(torch.load(critic_path, weights_only=True, map_location=device))

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        # 训练超参 & 状态
        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size
        # command-reward weight schedule lambda(step): flat lam_min for lam_warmup steps, then a
        # convex (quadratic, slow-start) ramp to lam_max over lam_ramp steps, then hold. Defaults
        # (1,1) reproduce the old constant weight. Purpose: let the gait form (g dominant) before
        # the task/command weight rises. env_reward is bounded (track mode), so lam_max is a hard
        # ceiling on the command term -> it can never swamp the gait prior g again.
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.lam_warmup = lam_warmup
        self.lam_ramp = max(1, lam_ramp)
        # --- gait-reward shaping (revised 2026-08) ---
        # Measured the RAW discriminator on the actual EXPERT dataset it was trained on
        # (sim/env/expert_66k_aug3c_fcontact.csv, 66k frames): mean=2.91, std=1.55, p95=6.13,
        # max=7.23. The expert's OWN states routinely score g=4-7 -- a tight clip near 3.0
        # (the earlier design) was not near a gaming boundary, it was cutting off the top ~40%
        # of genuinely expert-quality variation, crushing the gait term to a ~0.5-wide range
        # that gave PPO almost no gradient to prefer good gait over mediocre gait.
        # On expert data, g vs per-frame displacement corr = -0.28, and even the highest-g bin
        # (6-8) has the same per-frame movement as the lowest bin: a real g=6-7 frame is normal
        # mid-stride motion, not necessarily a frozen pose.  The command term must therefore
        # provide its own useful directional gradient; see normalized_env_66k.py::_track_reward.
        # g_clip is now a loose SAFETY NET (default well above the expert's own max, e.g. 8.0),
        # not a routine constraint. g_baseline centers on the EXPERT MEAN (not the clip), so
        # below-expert states go negative and above-expert states go positive -- preserving the
        # full ~[-2, +4.3] informative range instead of squashing it to [-0.5, 0].
        self.g_clip = g_clip
        self.g_center = g_center
        self.g_baseline = g_baseline if g_center else 0.0
        self.reward_scale = float(reward_scale)
        # A ReLU discriminator has no trustworthy extrapolation outside the normalized expert
        # support.  The discriminator view is still clipped to its original [-1,1]^28 contract;
        # this factor smoothly removes only the OOD extrapolation incentive.  It is exactly 1.0
        # for every in-support state, so the requested reward remains raw g(s') there.
        self.g_ood_coef = float(g_ood_coef)
        if self.g_ood_coef < 0.0:
            raise ValueError("g_ood_coef must be non-negative")
        self._rollout_g = []
        self._rollout_task = []


        self.disc = None
        # our friend's frozen CPG gait prior (../discriminator.pth), overridable via AMP_DISC
        airl_disc_path = os.environ.get("AMP_DISC",
                                        os.path.join(os.path.dirname(__file__), "..", "discriminator.pth"))
        self.disc_state_dim = int(disc_state_dim or state_shape[0])
        if self.disc_state_dim > state_shape[0]:
            raise ValueError(
                f"disc_state_dim={self.disc_state_dim} exceeds policy state dim={state_shape[0]}"
            )
        self.disc = AIRLDiscrim(
            state_shape=(self.disc_state_dim,),
            gamma=gamma,
            hidden_units_r=(100,100),
            hidden_units_v=(100,100),
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
        ).to(device)

        state_dict = torch.load(airl_disc_path, map_location=device)
        self.disc.load_state_dict(state_dict)
        self.disc.eval()
        self._disc_ood_values = 0
        self._disc_total_values = 0
        self._disc_ood_states = 0
        self._disc_total_states = 0
        self._disc_support_sum = 0.0
        self._disc_support_total = 0
        self._last_disc_support = torch.ones(1, device=device)
        print(f"[INFO] Loaded AIRL discriminator from {airl_disc_path} "
              f"(disc obs={self.disc_state_dim}, policy obs={state_shape[0]}, "
              f"critic obs={self.critic_state_shape[0]})")

    def _critic_state(self, state, context):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        context = np.asarray(context, dtype=np.float32).reshape(-1)
        if state.size != self.state_shape[0]:
            raise RuntimeError(f"policy state has {state.size} values, expected {self.state_shape[0]}")
        if context.size != self.critic_context_dim:
            raise RuntimeError(
                f"critic context has {context.size} values, expected {self.critic_context_dim}"
            )
        return np.concatenate((state, context))

    def _disc_input(self, state):
        """Return the frozen discriminator's original, bounded observation view.

        The expert normalisation bounds map its training support to [-1, 1].  Policy rollouts can
        leave that support (especially orientation while spinning/rearing); feeding those values
        directly to a ReLU discriminator permits arbitrary high OOD extrapolation.  Clip only the
        discriminator view and record how often clipping was needed.  Actor/critic observations
        remain unclipped and may include extra reward-state features.
        """
        if isinstance(state, torch.Tensor):
            s = state.to(self.device, dtype=torch.float32)
        else:
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        s = s[..., :self.disc_state_dim]
        ood = s.abs() > 1.0
        self._disc_ood_values += int(ood.sum().item())
        self._disc_total_values += int(ood.numel())
        self._disc_ood_states += int(ood.any(dim=-1).sum().item())
        self._disc_total_states += int(s.shape[0])
        overflow = torch.relu(s.abs() - 1.0)
        support = torch.exp(-self.g_ood_coef * overflow.square().sum(dim=-1))
        self._disc_support_sum += float(support.sum().item())
        self._disc_support_total += int(support.numel())
        self._last_disc_support = support
        return s.clamp(-1.0, 1.0)

    def _pop_disc_ood_stats(self):
        value_frac = self._disc_ood_values / max(1, self._disc_total_values)
        state_frac = self._disc_ood_states / max(1, self._disc_total_states)
        self._disc_ood_values = self._disc_total_values = 0
        self._disc_ood_states = self._disc_total_states = 0
        return value_frac, state_frac

    def _pop_disc_support_stats(self):
        mean = self._disc_support_sum / max(1, self._disc_support_total)
        self._disc_support_sum = 0.0
        self._disc_support_total = 0
        return mean

    @torch.no_grad()
    def airl_reward_g(self, state):
        """
        use the learned AIRL discriminator to compute reward
        state: numpy array 或 torch tensor, shape = [obs_dim]
        """
        s = self._disc_input(state)
        r = float(self.disc.g(s).item())  # [1, 1]
        r *= float(self._last_disc_support.item())
        if self.g_clip is not None:
            r = min(r, self.g_clip)        # loose safety net; see the __init__ note -- not a
                                            # routine constraint, expert's own g reaches ~7.2
        return r

    def _lambda(self, step):
        """Command-reward weight at this training step (see __init__ schedule notes)."""
        if step <= self.lam_warmup:
            return self.lam_min
        p = min(1.0, (step - self.lam_warmup) / self.lam_ramp)
        return self.lam_min + (self.lam_max - self.lam_min) * (p * p)   # convex: slow start

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1
        critic_state = self._critic_state(state, env.get_critic_context())
        action, log_pi = self.explore(state)
        next_state, env_reward, done, truncated, info = env.step(action)
        next_context = (info['critic_context']
                        if isinstance(info, dict) and 'critic_context' in info
                        else env.get_critic_context())
        next_critic_state = self._critic_state(next_state, next_context)
        done = done or truncated
        mask = False if t == env._max_episode_steps else done
        # ==== reward = (centered gait prior)  +  lambda(step) * env_reward (bounded task/command) ====
        g = self.airl_reward_g(next_state)                              # clipped gait prior (logged raw)
        # Preserve the original transfer objective: raw learned gait reward on s' plus a
        # weighted task reward.  A positive global scale changes critic target magnitude but not
        # the gait/task trade-off (GAE is normalised before the actor update).
        reward = self.reward_scale * (
            (g - self.g_baseline) + self._lambda(step) * env_reward
        )
        self._rollout_g.append(g)
        self._rollout_task.append(env_reward)
        self.buffer.append(state, action, reward, mask, log_pi, next_state,
                           critic_state=critic_state, next_critic_state=next_critic_state)
        if done:
            t = 0
            reset_out = env.reset()
            if isinstance(reset_out, tuple):
                next_state, info = reset_out
            else:
                next_state = reset_out
            if isinstance(next_state, dict):
                next_state = next_state.get("observation", next_state)
        return next_state, t

    def update(self, writer=None, model_dir=None):
        self.learning_steps += 1
        if writer is not None and self._rollout_g:
            writer.add_scalar('reward/g_mean', float(np.mean(self._rollout_g)), self.learning_steps)
            writer.add_scalar('reward/task_raw_mean', float(np.mean(self._rollout_task)), self.learning_steps)
            writer.add_scalar('reward/task_weight', self._lambda(self.learning_steps * self.rollout_length),
                              self.learning_steps)
        self._rollout_g.clear()
        self._rollout_task.clear()
        (states, actions, rewards, dones, log_pis, next_states,
         critic_states, next_critic_states) = self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states,
                        critic_states, next_critic_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   critic_states, next_critic_states, writer=None):
        if writer is not None:
            writer.add_scalar('reward/train_mean', rewards.mean().item(), self.learning_steps)
            writer.add_scalar('reward/train_std', rewards.std().item(), self.learning_steps)
            writer.add_scalar('reward/train_min', rewards.min().item(), self.learning_steps)
            writer.add_scalar('reward/train_max', rewards.max().item(), self.learning_steps)
            if self.critic_context_dim:
                context = critic_states[..., -self.critic_context_dim:]
                writer.add_scalar('critic/context_mean', context.mean().item(), self.learning_steps)
                writer.add_scalar('critic/context_std', context.std().item(), self.learning_steps)
            value_frac, state_frac = self._pop_disc_ood_stats()
            writer.add_scalar('diagnostics/disc_ood_value_frac', value_frac, self.learning_steps)
            writer.add_scalar('diagnostics/disc_ood_state_frac', state_frac, self.learning_steps)
            writer.add_scalar('diagnostics/disc_support_mean', self._pop_disc_support_stats(),
                              self.learning_steps)

        with torch.no_grad():
            values = self.critic(critic_states)
            next_values = self.critic(next_critic_states)

        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)

        # 这里未做 mini-batch 打乱；如需小批量，可按需要切分 states 等张量
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(critic_states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer=None):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)
            writer.add_scalar(
                'gae/target_mean', targets.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/target_std', targets.std().item(), self.learning_steps)
            writer.add_scalar(
                'gae/value_mean', self.critic(states).mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/value_std', self.critic(states).std().item(), self.learning_steps)
            writer.add_scalar(
                'critic/grad_norm_preclip', float(grad_norm), self.learning_steps)
            with torch.no_grad():
                predicted = self.critic(states)
                target_var = torch.var(targets, unbiased=False)
                residual_var = torch.var(targets - predicted, unbiased=False)
                explained_variance = 1.0 - residual_var / (target_var + 1e-8)
            writer.add_scalar(
                'critic/explained_variance', explained_variance.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)
            writer.add_scalar(
                'stats/ratio', ratios.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/gae_mean', gaes.mean().item(), self.learning_steps)
            writer.add_scalar(
                'gae/gae_std', gaes.std().item(), self.learning_steps)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
