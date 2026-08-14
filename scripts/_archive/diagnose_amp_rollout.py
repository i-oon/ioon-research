"""Numerically diagnose an AMP checkpoint without rendering.

Reports task motion, frozen-discriminator reward, and the normalized observation
channels that leave the expert support [-1, 1].  The actor and discriminator both
retain their original 28-D inputs; this script only observes their rollout.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "amp"))

from common.normalized_env_66k import CoppeliaSimEnv  # noqa: E402
from networks.actor import ActorNetworkPolicy  # noqa: E402
from networks.discrim import AIRLDiscrim  # noqa: E402


CHANNELS = (
    ["body_z", "roll", "pitch", "yaw"]
    + [f"{leg}_{joint}" for leg in ("FL", "ML", "HL", "FR", "MR", "HR")
       for joint in ("ThC", "CTr", "FTi")]
    + [f"contact_{leg}" for leg in ("FL", "ML", "HL", "FR", "MR", "HR")]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--disc", default=os.path.join(ROOT, "amp", "discriminator.pth"))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--vx_target", type=float, default=0.45)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--track_window", type=int, default=25)
    args = ap.parse_args()

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    scene = os.path.abspath(args.scene)
    ckpt = args.ckpt if args.ckpt.endswith(".pth") else os.path.join(args.ckpt, "actor.pth")
    sim0 = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim0.stopSimulation()
    while sim0.getSimulationState() != sim0.simulation_stopped:
        time.sleep(0.1)
    sim0.loadScene(scene)

    env = CoppeliaSimEnv(
        port=args.port,
        OnTimeStep=True,
        reward_mode="track",
        vx_target=args.vx_target,
        track_sigma=args.sigma,
        track_window=args.track_window,
    )
    if env.observation_space.shape != (28,):
        raise RuntimeError(f"Expected the frozen 28-D contract, got {env.observation_space.shape}")

    actor = ActorNetworkPolicy(
        state_shape=(28,), action_shape=env.action_space.shape,
        hidden_units=(64, 64), hidden_activation=nn.Tanh(),
    )
    actor.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    actor.eval()

    disc = AIRLDiscrim(
        state_shape=(28,), gamma=0.995,
        hidden_units_r=(100, 100), hidden_units_v=(100, 100),
        hidden_activation_r=nn.ReLU(inplace=True),
        hidden_activation_v=nn.ReLU(inplace=True),
    )
    disc.load_state_dict(torch.load(args.disc, map_location="cpu", weights_only=True))
    disc.eval()

    state = env.reset()
    head = env.sim.getObject("/head")
    p0 = np.asarray(env.sim.getObjectPosition(head), dtype=float)
    states, task_rewards, velocities, g_raw, g_bounded = [], [], [], [], []
    terminated = truncated = False
    for _ in range(args.steps):
        with torch.no_grad():
            action = actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        state, task_reward, terminated, truncated, info = env.step(action)
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            g_raw.append(float(disc.g(s).item()))
            g_bounded.append(float(disc.g(s.clamp(-1.0, 1.0)).item()))
        states.append(np.asarray(state, dtype=float))
        task_rewards.append(float(task_reward))
        velocities.append(float(info["vx_avg"]))
        if terminated or truncated:
            break
    p1 = np.asarray(env.sim.getObjectPosition(head), dtype=float)
    env.stop()

    states = np.asarray(states)
    task_rewards = np.asarray(task_rewards)
    velocities = np.asarray(velocities)
    g_raw = np.asarray(g_raw)
    g_bounded = np.asarray(g_bounded)
    ood = np.abs(states) > 1.0
    rates = ood.mean(axis=0)
    overflow = np.maximum(np.abs(states) - 1.0, 0.0)
    order = np.argsort(rates)[::-1]

    print(f"checkpoint={ckpt}")
    print(f"steps={len(states)} terminated={terminated} truncated={truncated}")
    print(f"target={env.vx_target:.4f} sigma={env.track_sigma:.4f} "
          f"dx={p1[0]-p0[0]:+.4f} dy={p1[1]-p0[1]:+.4f}")
    print(f"vx_avg mean={velocities.mean():+.4f} median={np.median(velocities):+.4f} "
          f"last={velocities[-1]:+.4f}")
    print(f"task mean={task_rewards.mean():.6f} sum={task_rewards.sum():.3f}")
    print(f"g(clamped-input) mean={g_bounded.mean():.4f} max={g_bounded.max():.4f}; "
          f"g(raw-input) mean={g_raw.mean():.4f} max={g_raw.max():.4f}")
    print(f"OOD states={ood.any(axis=1).mean():.3%}; values={ood.mean():.3%}")
    print("top OOD channels (rate, mean overflow, max |normalized value|):")
    for idx in order:
        if rates[idx] <= 0:
            break
        print(f"  {idx:02d} {CHANNELS[idx]:12s} {rates[idx]:7.2%} "
              f"{overflow[:, idx].mean():9.4f} {np.abs(states[:, idx]).max():9.4f}")


if __name__ == "__main__":
    main()
