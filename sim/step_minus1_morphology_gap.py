"""Step -1 — Morphology Gap Check.

Runs the SAME open-loop gait replay (main_script.py's CSV-driven joint
targets, identical regardless of body) on the short-leg and long-leg (base)
variants, logs body trajectory and foot height traces for each, and compares
them. Pass criterion (per direction_plan.md): visibly distinct behavior
between morphologies under identical commands.

Usage:
  python sim/step_minus1_morphology_gap.py --duration 10
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

VARIANTS = {
    "short (0.5x)": "/home/aria/ioon-research/sim/env/medauroidea_stick_insect_short.ttt",
    "long/base (1.0x)": "/home/aria/ioon-research/sim/env/medauroidea_stick_insect.ttt",
}
LEG_SUFFIXES = ["_FL", "_ML", "_HL", "_FR", "_MR", "_HR"]
SAMPLE_DT = 0.05  # seconds between samples


def run_variant(sim, path, duration):
    sim.loadScene(path)
    head = sim.getObject("/head")
    foot_handles = {s: sim.getObject(f"/foot{s}") for s in LEG_SUFFIXES}

    sim.startSimulation()
    t0 = time.time()
    samples = {"t": [], "head_pos": [], "foot_z": {s: [] for s in LEG_SUFFIXES}}
    while time.time() - t0 < duration:
        samples["t"].append(time.time() - t0)
        samples["head_pos"].append(sim.getObjectPosition(head, sim.handle_world))
        for s in LEG_SUFFIXES:
            z = sim.getObjectPosition(foot_handles[s], sim.handle_world)[2]
            samples["foot_z"][s].append(z)
        time.sleep(SAMPLE_DT)
    sim.stopSimulation()
    time.sleep(0.5)

    for k in ("t", "head_pos"):
        samples[k] = np.array(samples[k])
    for s in LEG_SUFFIXES:
        samples["foot_z"][s] = np.array(samples["foot_z"][s])
    return samples


def summarize(name, samples):
    head_pos = samples["head_pos"]
    forward_dist = np.linalg.norm(head_pos[-1, :2] - head_pos[0, :2])
    height_std = head_pos[:, 2].std()
    print(f"\n=== {name} ===")
    print(f"forward distance traveled (xy): {forward_dist:.4f} m")
    print(f"body height std (stability):    {height_std:.4f} m")
    for s in LEG_SUFFIXES:
        fz = samples["foot_z"][s]
        print(f"  foot{s}: z min={fz.min():.4f} max={fz.max():.4f} "
              f"swing_clearance={fz.max()-fz.min():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="step_minus1_comparison.png")
    args = ap.parse_args()

    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")

    results = {}
    for name, path in VARIANTS.items():
        print(f"running {name} ...")
        results[name] = run_variant(sim, path, args.duration)
        summarize(name, results[name])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, samples in results.items():
        hp = samples["head_pos"]
        dist = np.linalg.norm(hp[:, :2] - hp[0, :2], axis=1)
        axes[0].plot(samples["t"], dist, label=name)
    axes[0].set_ylabel("forward distance (m)")
    axes[0].set_title("Body forward progress under identical gait commands")
    axes[0].legend()

    for name, samples in results.items():
        axes[1].plot(samples["t"], samples["foot_z"]["_FL"], label=f"{name} foot_FL")
    axes[1].set_ylabel("foot_FL height (z, m)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Front-left foot height (swing/stance pattern)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
