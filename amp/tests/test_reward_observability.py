import os
import sys
import unittest

import numpy as np
import torch


AMP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AMP_ROOT)

from algorithms.ppo_transfer import PPO  # noqa: E402
from common.buffer import RolloutBuffer  # noqa: E402
from common.normalized_env_66k import CoppeliaSimEnv  # noqa: E402


class EnvironmentContractTest(unittest.TestCase):
    def test_policy_and_discriminator_observations_remain_28d(self):
        env = CoppeliaSimEnv(
            simulation=False,
            reward_mode="track",
            vx_target=0.45,
            track_sigma=0.15,
        )
        self.assertEqual(env.disc_observation_dim, 28)
        self.assertEqual(env.observation_space.shape, (28,))
        self.assertEqual(env.critic_context_dim, 1)
        env._set_critic_velocity_context(env.vx_target)
        np.testing.assert_array_equal(env.get_critic_context(), np.array([0.0], dtype=np.float32))
        env._set_critic_velocity_context(0.0)
        np.testing.assert_allclose(env.get_critic_context(), np.array([-0.75], dtype=np.float32))
        self.assertAlmostEqual(env._track_reward(env.vx_target), 1.0)
        self.assertGreater(env._track_reward(0.1 * env.vx_target), env._track_reward(0.0))
        self.assertLess(env._track_reward(-env.vx_target), 0.0)
        self.assertLess(env._track_reward(2.0 * env.vx_target), 1.0)

    def test_discriminator_clamps_ood_values_without_changing_width(self):
        algo = PPO(
            state_shape=(28,),
            action_shape=(18,),
            device=torch.device("cpu"),
            seed=0,
            rollout_length=8,
            epoch_ppo=1,
            disc_state_dim=28,
            critic_context_dim=1,
            g_clip=8.0,
            g_center=True,
            g_baseline=2.91,
            g_ood_coef=2.0,
        )
        ood_policy_obs = np.full(28, 2.0, dtype=np.float32)
        bounded_disc_obs = np.ones(28, dtype=np.float32)
        algo.airl_reward_g(ood_policy_obs)
        self.assertLess(algo._last_disc_support.item(), 1e-10)
        algo.airl_reward_g(bounded_disc_obs)
        self.assertEqual(algo._last_disc_support.item(), 1.0)
        value_frac, state_frac = algo._pop_disc_ood_stats()
        self.assertEqual(value_frac, 0.5)
        self.assertEqual(state_frac, 0.5)
        self.assertEqual(algo.actor.net[0].in_features, 28)
        self.assertEqual(algo.disc.g[0].in_features, 28)
        self.assertEqual(algo.critic.net[0].in_features, 29)
        critic_state = algo._critic_state(np.zeros(28, dtype=np.float32), np.array([0.25], dtype=np.float32))
        self.assertEqual(critic_state.shape, (29,))
        self.assertEqual(critic_state[-1], 0.25)

    def test_rollout_buffer_keeps_policy_and_critic_states_separate(self):
        buffer = RolloutBuffer(
            buffer_size=2,
            state_shape=(28,),
            action_shape=(18,),
            critic_state_shape=(29,),
            device=torch.device("cpu"),
            mix=1,
        )
        for i in range(2):
            state = np.full(28, i, dtype=np.float32)
            next_state = np.full(28, i + 1, dtype=np.float32)
            critic_state = np.concatenate((state, np.array([i / 4], dtype=np.float32)))
            next_critic_state = np.concatenate((next_state, np.array([(i + 1) / 4], dtype=np.float32)))
            buffer.append(
                state,
                np.zeros(18, dtype=np.float32),
                reward=float(i),
                done=False,
                log_pi=0.0,
                next_state=next_state,
                critic_state=critic_state,
                next_critic_state=next_critic_state,
            )
        items = buffer.get()
        self.assertEqual(len(items), 8)
        self.assertEqual(tuple(items[0].shape), (2, 28))
        self.assertEqual(tuple(items[6].shape), (2, 29))
        self.assertEqual(tuple(items[7].shape), (2, 29))
        self.assertAlmostEqual(items[6][1, -1].item(), 0.25)


if __name__ == "__main__":
    unittest.main()
