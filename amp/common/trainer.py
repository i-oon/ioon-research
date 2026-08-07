'''
This Trainer class manages the training and evaluation loop for a reinforcement learning algorithm.
'''


import os
from datetime import timedelta
from time import time, sleep
from torch.utils.tensorboard import SummaryWriter
import sys


class Trainer:
    def __init__(self, env, env_test, algo, log_dir, num_steps=10**7,
                 eval_interval=10**5, num_eval_episodes=5):
        self.env = env

        self.env_test = env_test

        self.algo = algo
        self.log_dir = log_dir

        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes


    def train(self):
        self.start_time = time()
        t = 0

        # Reset the environment properly.
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out
        if isinstance(state, dict):
            state = state.get("observation", state)
        print(f"Initial state shape: {state.shape}")

        for step in range(1, self.num_steps + 1):
            state, t = self.algo.step(self.env, state, t, step)
            if self.algo.is_update(step):
                print(f"Step {step} - Updating the algorithm", file=sys.__stdout__)
                self.algo.update(self.writer)
            if step % self.eval_interval == 0:
                print(f"Step {step} - Evaluating the algorithm", file=sys.__stdout__)
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f'step{step}'))
                # This project uses the same CoppeliaSim instance for training and evaluation.
                # evaluate() resets and advances that instance through five complete episodes,
                # so the local training `state` is stale afterwards.  Continuing with it creates
                # an impossible transition (old observation/action -> final evaluation state).
                # Explicitly reset and replace both state and episode time before resuming.
                if self.env_test is self.env:
                    reset_out = self.env.reset()
                    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    if isinstance(state, dict):
                        state = state.get("observation", state)
                    t = 0
        sleep(10)


    def evaluate(self, step):
        # return/test = env command reward only (windowed speed tracking in the current setup).
        # gait/g_eval = mean discriminator reward g(s') per step -> gait-quality metric
        # (is the frozen CPG prior actually satisfied, or is it just running fast+ugly?).
        has_g = hasattr(self.algo, 'airl_reward_g')
        env = self.env_test
        try:                                              # coppelia env -> also track posture/drift/stability
            head = env.sim.getObject('/head'); has_pose = True
        except Exception:
            has_pose = False
        mean_return = 0.0
        g_sum, g_steps = 0.0, 0
        x_sum, y_sum, pitch_sum, len_sum = 0.0, 0.0, 0.0, 0
        for _ in range(self.num_eval_episodes):
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            if isinstance(state, dict):
                state = state.get("observation", state)
            p0 = env.sim.getObjectPosition(head) if has_pose else None
            ep_pitch, ep_len = 0.0, 0
            episode_return = 0.0
            done = False
            while not done:
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                if has_g:
                    g_sum += self.algo.airl_reward_g(state)   # g on s' (matches training reward)
                    g_steps += 1
                if has_pose:
                    try:
                        ep_pitch += abs(env.get_bodyorientation()[1]); ep_len += 1   # |pitch| -> rearing
                    except Exception:
                        pass
            mean_return += episode_return / self.num_eval_episodes
            if has_pose:
                p1 = env.sim.getObjectPosition(head)
                x_sum += (p1[0] - p0[0])                       # forward distance
                y_sum += abs(p1[1] - p0[1])                    # lateral drift
                pitch_sum += ep_pitch / max(1, ep_len)         # mean |pitch|
                len_sum += ep_len                              # steps before fall = stability
        n = self.num_eval_episodes
        self.writer.add_scalar('return/test', mean_return, step)
        mean_g = (g_sum / g_steps) if g_steps else 0.0
        if has_g:
            self.writer.add_scalar('gait/g_eval', mean_g, step)
            if hasattr(self.algo, '_pop_disc_ood_stats'):
                value_frac, state_frac = self.algo._pop_disc_ood_stats()
                self.writer.add_scalar('diagnostics/eval_disc_ood_value_frac', value_frac, step)
                self.writer.add_scalar('diagnostics/eval_disc_ood_state_frac', state_frac, step)
                if hasattr(self.algo, '_pop_disc_support_stats'):
                    self.writer.add_scalar('diagnostics/eval_disc_support_mean',
                                           self.algo._pop_disc_support_stats(), step)
        if has_pose:
            self.writer.add_scalar('eval/x_dist', x_sum / n, step)     # forward reach (m)
            self.writer.add_scalar('eval/y_drift', y_sum / n, step)    # lateral drift (m) -> short's issue
            self.writer.add_scalar('eval/pitch_abs', pitch_sum / n, step)  # mean |pitch| -> long's rearing
            self.writer.add_scalar('eval/ep_len', len_sum / n, step)   # steps before fall -> stability
        print(f"Num steps: {step:<6}   Return: {mean_return:<7.1f}   g/step: {mean_g:<5.2f}   "
              f"x:{x_sum/n:+.1f}m  ydrift:{y_sum/n:.2f}m  |pitch|:{pitch_sum/n:.2f}  len:{len_sum/n:.0f}   {self.time}")


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
