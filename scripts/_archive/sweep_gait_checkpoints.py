"""Find the checkpoint whose temporal contact pattern is closest to the expert.

This is deliberately separate from discriminator g: the frozen 28-D g(s) is state-only and cannot
measure contact ordering.  Each checkpoint is rolled out with identical reset-noise seeds, its
average stride is phase-normalized to 100 bins, and the six-leg profile is compared directly with
the canonical long-body expert episode.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "amp"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "amp"))   # gait_report moved here
sys.path.insert(0, HERE)

from common.normalized_env_66k import CoppeliaSimEnv  # noqa: E402
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from gait_report import (CSV, average_cycle, coord_distance, duty, load_expert,
                         phase_offsets, stride_period)  # noqa: E402
from networks.actor import ActorNetworkPolicy  # noqa: E402
from networks.discrim import AIRLDiscrim  # noqa: E402


def rollout(env, actor, steps, seed):
    np.random.seed(seed)
    state = env.reset()
    head = env.sim.getObject("/head")
    p0 = np.asarray(env.sim.getObjectPosition(head), dtype=float)
    contacts, states, positions, velocities = [], [], [], []
    fell = False
    for _ in range(steps):
        with torch.no_grad():
            action = actor(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        state, _, terminated, truncated, info = env.step(action)
        contacts.append(env.get_contact())
        states.append(np.asarray(state, dtype=float))
        positions.append(np.asarray(env.sim.getObjectPosition(head), dtype=float))
        velocities.append(float(info["vx_avg"]))
        if terminated or truncated:
            fell = bool(terminated)
            break
    return (np.asarray(contacts), np.asarray(states), np.asarray(positions),
            np.asarray(velocities), p0, fell)


def discriminator_metrics(disc, states, coef):
    s = torch.as_tensor(states, dtype=torch.float32)
    overflow = torch.relu(s.abs() - 1.0)
    support = torch.exp(-coef * overflow.square().sum(dim=-1))
    with torch.no_grad():
        raw = disc.g(s.clamp(-1.0, 1.0)).squeeze(-1)
    gated = raw * support
    return float(gated.mean()), float(support.mean()), float((s.abs() > 1.0).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run_dir", required=True, help="directory containing model/stepN")
    ap.add_argument("--checkpoints", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--expert_ep", type=int, default=926)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vx_target", type=float, default=0.45)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--g_ood_coef", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    scene = os.path.abspath(args.scene)
    sim0 = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim0.stopSimulation()
    while sim0.getSimulationState() != sim0.simulation_stopped:
        time.sleep(0.1)
    sim0.loadScene(scene)

    env = CoppeliaSimEnv(port=args.port, OnTimeStep=True, reward_mode="track",
                         vx_target=args.vx_target, track_sigma=args.sigma,
                         track_window=25, track_direction_mix=0.5)
    actor = ActorNetworkPolicy(state_shape=(28,), action_shape=env.action_space.shape,
                               hidden_units=(64, 64), hidden_activation=torch.nn.Tanh())
    actor.eval()
    disc = AIRLDiscrim(state_shape=(28,), gamma=0.995,
                       hidden_units_r=(100, 100), hidden_units_v=(100, 100),
                       hidden_activation_r=torch.nn.ReLU(inplace=True),
                       hidden_activation_v=torch.nn.ReLU(inplace=True))
    disc.load_state_dict(torch.load(os.path.join(ROOT, "amp", "discriminator.pth"),
                                    map_location="cpu", weights_only=True))
    disc.eval()

    canonical_env = CoppeliaSimEnv(simulation=False)
    ec, estates, edt = load_expert(pd.read_csv(CSV), args.expert_ep, canonical_env)
    ep = stride_period(ec)
    eprofile, _ = average_cycle(ec, ep)
    eph, eduty = phase_offsets(ec, ep), duty(ec)
    efreq = 1.0 / (ep * edt)
    eg, _, _ = discriminator_metrics(disc, estates, args.g_ood_coef)

    rows = []
    for step in args.checkpoints:
        path = os.path.join(args.run_dir, "model", f"step{step}", "actor.pth")
        if not os.path.exists(path):
            print(f"skip missing {path}", flush=True)
            continue
        actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        episode_rows = []
        for episode in range(args.episodes):
            c, states, pos, vx, p0, fell = rollout(env, actor, args.steps, args.seed + episode)
            period = stride_period(c)
            if not period:
                continue
            profile, cycles = average_cycle(c, period)
            ph, du = phase_offsets(c, period), duty(c)
            freq = 1.0 / (period * env.dt)
            g, support, ood_value = discriminator_metrics(disc, states, args.g_ood_coef)
            dx, dy = pos[-1, 0] - p0[0], pos[-1, 1] - p0[1]
            episode_rows.append({
                "profile_mae": float(np.mean(np.abs(profile - eprofile))),
                "phase_distance": coord_distance(ph, eph),
                "duty_mae": float(np.mean(np.abs(du - eduty))),
                "freq_ratio": freq / efreq,
                "g_ratio": g / eg,
                "support": support,
                "ood_value": ood_value,
                "vx_ratio": float(np.mean(vx)) / env.vx_target,
                "dx": float(dx),
                "drift_ratio": abs(float(dy)) / max(abs(float(dx)), 1e-6),
                "survival": len(c) / args.steps,
                "fell": float(fell),
                "cycles": cycles,
                "active_legs": float(np.sum(du >= 0.20)),
            })
        if not episode_rows:
            continue
        row = {"step": step}
        for key in episode_rows[0]:
            row[key] = float(np.mean([x[key] for x in episode_rows]))
        # Temporal style is primary. Cadence is a separate real-time component; pose/support and
        # task viability are reported rather than hidden inside an arbitrary all-purpose score.
        row["temporal_error"] = row["profile_mae"] + 0.25 * abs(math.log(max(row["freq_ratio"], 1e-6)))
        rows.append(row)
        print(f"{step//1000:3d}k temporal={row['temporal_error']:.3f} "
              f"profile={row['profile_mae']:.3f} phase={row['phase_distance']:.3f} "
              f"duty={row['duty_mae']:.3f} rate={row['freq_ratio']:.2f}x "
              f"g={row['g_ratio']:.2f}x vx={row['vx_ratio']:.2f}x "
              f"drift={row['drift_ratio']:.2f} survival={row['survival']:.2f}", flush=True)

    env.stop()
    result = pd.DataFrame(rows).sort_values(["temporal_error", "profile_mae"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_csv(args.out, index=False)
    viable = result[(result.survival >= 0.8) & (result.vx_ratio >= 0.5)]
    print("\n=== TEMPORAL RANKING (lower is better) ===")
    print((viable if len(viable) else result).to_string(index=False))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
