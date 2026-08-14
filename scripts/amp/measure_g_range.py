"""Measure the RAW (unclipped) discriminator g(s') over a deterministic rollout of a
trained checkpoint, so we can pick g_clip/lambda from real numbers instead of guessing.
We only ever log the CLIPPED g in training -> we don't actually know the natural spread.

  .venv/bin/python3 scripts/measure_g_range.py --port 23063 \
      --scene sim/env/medauroidea_stick_insect.ttt \
      --ckpt amp/logs/insect_long/20260804-1646/model/step260000
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "amp"))
from networks.actor import ActorNetworkPolicy        # noqa: E402
from networks.discrim import AIRLDiscrim              # noqa: E402
from common.normalized_env_66k import CoppeliaSimEnv   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True, help="dir containing actor.pth, or the actor.pth itself")
    ap.add_argument("--disc", default=os.path.join(ROOT, "amp", "discriminator.pth"))
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    import time
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    scene = os.path.abspath(args.scene)
    ckpt = args.ckpt if args.ckpt.endswith(".pth") else os.path.join(args.ckpt, "actor.pth")

    sim0 = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim0.stopSimulation()
    while sim0.getSimulationState() != 0:
        time.sleep(0.1)
    sim0.loadScene(scene)

    env = CoppeliaSimEnv(port=args.port, OnTimeStep=True)
    actor = ActorNetworkPolicy(state_shape=env.observation_space.shape,
                               action_shape=env.action_space.shape,
                               hidden_units=(64, 64), hidden_activation=nn.Tanh())
    actor.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    actor.eval()

    disc = AIRLDiscrim(state_shape=env.observation_space.shape, gamma=0.995,
                       hidden_units_r=(100, 100), hidden_units_v=(100, 100),
                       hidden_activation_r=nn.ReLU(inplace=True), hidden_activation_v=nn.ReLU(inplace=True))
    disc.load_state_dict(torch.load(args.disc, map_location="cpu"))
    disc.eval()

    state = env.reset()
    g_vals = []
    for t in range(args.steps):
        with torch.no_grad():
            a = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        next_state, reward, terminated, truncated, _ = env.step(a)
        with torch.no_grad():
            s = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
            g = float(disc.g(s).item())   # RAW, no clip
        g_vals.append(g)
        state = next_state
        if terminated or truncated:
            break
    env.stop()

    g = np.array(g_vals)
    print(f"\n{args.ckpt}")
    print(f"  n={len(g)}  min={g.min():.3f}  p5={np.percentile(g,5):.3f}  "
          f"mean={g.mean():.3f}  median={np.median(g):.3f}  p95={np.percentile(g,95):.3f}  max={g.max():.3f}  std={g.std():.3f}")


if __name__ == "__main__":
    main()
