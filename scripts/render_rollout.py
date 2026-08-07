"""Render a trained AMP policy rollout to an mp4 so we can eyeball the GAIT.

Scalar task return alone says that the policy walks forward, not that it walks
nicely or survives. This script loads a checkpoint actor, runs it deterministically,
films it with a FOLLOWING side-camera, and prints a compact numerical evaluation (the
scene's baked-in vjepa_cam is bolted in place for the short data-collection
clips, so a long walk would leave frame — we make our own tracking camera here).

Needs its OWN CoppeliaSim instance (the 3 training sims are busy), e.g. from the
CoppeliaSim_d copy on a spare port:
  cd /home/aria/CoppeliaSim_d && ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=23063

Then (repo root, one call renders all bodies it's pointed at):
  .venv/bin/python3 scripts/render_rollout.py --port 23063 \
      --scene sim/env/medauroidea_stick_insect.ttt \
      --ckpt amp/logs/insect_long/<ts>/model/step100000 \
      --out results/rollouts/long_step100000.mp4
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "amp"))          # actor lives in amp/networks
from networks.actor import ActorNetworkPolicy          # noqa: E402
from networks.discrim import AIRLDiscrim                # noqa: E402
from common.normalized_env_66k import CoppeliaSimEnv    # noqa: E402

# --- following side-camera framing (wider/closer than the dataset cam: we want
#     to SEE the legs, not compress perspective for V-JEPA) ---
DISTANCE = 3.5       # m from the robot
ELEVATION = 22.0     # deg above horizontal
AZIMUTH = 90.0       # deg -> pure side view (legs fully visible)
FOV = 34.0           # deg field of view
TARGET_Z = 0.10      # aim height
RES = 480            # output frame size (px); dataset uses 256, bigger = nicer video


def cam_offset():
    el, az = np.deg2rad(ELEVATION), np.deg2rad(AZIMUTH)
    horiz = DISTANCE * np.cos(el)
    return np.array([horiz * np.cos(az), horiz * np.sin(az), DISTANCE * np.sin(el)])


def look_at(cam_pos, target):
    """3x4 row-major pose; CoppeliaSim vision sensors view along their +Z."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return [v for r in range(3) for v in (x[r], y[r], z[r], cam_pos[r])]


def make_follow_cam(sim):
    options = 1 | 2 | 4                                  # explicitly-handled | perspective | hide frustum
    h = sim.createVisionSensor(options, [RES, RES, 0, 0],
                               [0.01, 30.0, np.deg2rad(FOV), 0.05, 0, 0, 0, 0, 0, 0, 0])
    sim.setObjectAlias(h, "render_cam")
    sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0xFFFF)
    return h


def place_cam(sim, cam, head_xy):
    target = np.array([head_xy[0], head_xy[1], TARGET_Z])
    sim.setObjectMatrix(cam, look_at(target + cam_offset(), target))


def grab(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23063)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True, help="dir containing actor.pth, or the actor.pth itself")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--disc", default=os.path.join(ROOT, "amp", "discriminator.pth"))
    ap.add_argument("--vx_target", type=float, default=0.45)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--track_window", type=int, default=25)
    ap.add_argument("--track_direction_mix", type=float, default=0.5)
    ap.add_argument("--g_ood_coef", type=float, default=2.0)
    args = ap.parse_args()

    import time
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    scene = os.path.abspath(args.scene)
    ckpt = args.ckpt if args.ckpt.endswith(".pth") else os.path.join(args.ckpt, "actor.pth")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # load scene before the env (headless starts empty)
    sim0 = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim0.stopSimulation()
    while sim0.getSimulationState() != 0:
        time.sleep(0.1)
    sim0.loadScene(scene)

    # Match the current training reward/context configuration. This does not change actor input:
    # actor and frozen discriminator remain exactly 28-D.
    env = CoppeliaSimEnv(port=args.port, OnTimeStep=True, reward_mode="track",
                         vx_target=args.vx_target, track_sigma=args.sigma,
                         track_window=args.track_window,
                         track_direction_mix=args.track_direction_mix)
    actor = ActorNetworkPolicy(state_shape=env.observation_space.shape,
                               action_shape=env.action_space.shape,
                               hidden_units=(64, 64), hidden_activation=torch.nn.Tanh())
    actor.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    actor.eval()

    disc = AIRLDiscrim(state_shape=(28,), gamma=0.995,
                       hidden_units_r=(100, 100), hidden_units_v=(100, 100),
                       hidden_activation_r=torch.nn.ReLU(inplace=True),
                       hidden_activation_v=torch.nn.ReLU(inplace=True))
    disc.load_state_dict(torch.load(args.disc, map_location="cpu", weights_only=True))
    disc.eval()
    print(f"loaded {ckpt}  obs{env.observation_space.shape} act{env.action_space.shape}")

    head = env.sim.getObject("/head")
    state = env.reset()   # first reset() is a HARD reset (real stop/start, reloads the
                           # scene) -> destroys any dynamically-created object made before
                           # it, incl. a vision sensor. Create the follow-cam AFTER reset.
    cam = make_follow_cam(env.sim)
    import cv2
    frames, fell = [], False
    p0 = np.asarray(env.sim.getObjectPosition(head), dtype=float)
    positions, orientations = [], []
    task_rewards, vx_avgs = [], []
    gait_scores, support_scores = [], []
    ood_states, ood_values = [], []
    for t in range(args.steps):
        with torch.no_grad():
            a = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        state, reward, terminated, truncated, info = env.step(a)
        hp = np.asarray(env.sim.getObjectPosition(head), dtype=float)
        orientation = np.asarray(env.get_bodyorientation(), dtype=float)

        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        overflow = torch.relu(s.abs() - 1.0)
        support = torch.exp(-args.g_ood_coef * overflow.square().sum(dim=-1))
        with torch.no_grad():
            g = disc.g(s.clamp(-1.0, 1.0)).item() * support.item()

        positions.append(hp)
        orientations.append(orientation)
        task_rewards.append(float(reward))
        vx_avgs.append(float(info["vx_avg"]))
        gait_scores.append(float(g))
        support_scores.append(float(support.item()))
        ood_states.append(bool((s.abs() > 1.0).any().item()))
        ood_values.append(float((s.abs() > 1.0).float().mean().item()))

        place_cam(env.sim, cam, hp[:2])
        img = grab(env.sim, cam)
        cv2.putText(img, f"t={t:3d}  x={hp[0]-p0[0]:+.2f}m  vx_avg={info['vx_avg']:+.2f}  g={g:.2f}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        frames.append(img)
        if terminated or truncated:
            fell = terminated
            break
    env.stop()

    import imageio
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=None)
    reached_requested_limit = len(frames) >= args.steps and not fell and not truncated
    tag = "FELL" if fell else ("env time limit" if truncated else
                                ("requested limit" if reached_requested_limit else "ended"))
    pos = np.asarray(positions)
    ori = np.asarray(orientations)
    task = np.asarray(task_rewards)
    vx = np.asarray(vx_avgs)
    gait = np.asarray(gait_scores)
    support = np.asarray(support_scores)
    n = len(frames)
    sim_seconds = n * env.dt
    dx, dy = pos[-1, 0] - p0[0], pos[-1, 1] - p0[1]
    probe = min(200, n) - 1
    min_height = max(0.02, 0.25 * env.stand_height)
    reasons = []
    if pos[-1, 2] <= min_height:
        reasons.append(f"body_z {pos[-1, 2]:.3f} <= {min_height:.3f}")
    if abs(ori[-1, 0]) >= 1.2:
        reasons.append(f"|roll| {abs(ori[-1, 0]):.3f} >= 1.2")
    if abs(ori[-1, 1]) >= 1.2:
        reasons.append(f"|pitch| {abs(ori[-1, 1]):.3f} >= 1.2")
    reason = ", ".join(reasons) if reasons else (
        "environment time limit" if truncated else
        ("requested step limit" if reached_requested_limit else "unknown")
    )

    print(f"{n} frames -> {args.out}   final x-dist={dx:+.2f}m   [{tag}]")
    print("\n=== rollout evaluation ===")
    print(f"checkpoint : {ckpt}")
    print(f"survival   : {n}/{args.steps} steps ({100*n/args.steps:.1f}%), "
          f"sim {sim_seconds:.2f}s; termination={reason}")
    print(f"trajectory : dx={dx:+.3f}m  |dy|={abs(dy):.3f}m  drift/x={abs(dy)/max(abs(dx),1e-6):.3f}")
    print(f"first 200  : dx={pos[probe,0]-p0[0]:+.3f}m  |dy|={abs(pos[probe,1]-p0[1]):.3f}m")
    print(f"velocity   : target={env.vx_target:.3f}  net={dx/max(sim_seconds,1e-6):+.3f}  "
          f"vx_avg mean={vx.mean():+.3f} std={vx.std():.3f} m/s")
    print(f"task       : mean={task.mean():+.4f}  sum={task.sum():+.2f}  "
          f"(bounded direction+precision reward)")
    print(f"gait prior : gated g mean={gait.mean():.3f} std={gait.std():.3f}; "
          f"support mean={support.mean():.3f}")
    print(f"OOD        : states={np.mean(ood_states):.1%}  values={np.mean(ood_values):.1%}")
    print(f"posture    : mean |roll|={np.abs(ori[:,0]).mean():.3f}  "
          f"|pitch|={np.abs(ori[:,1]).mean():.3f}; final z/r/p="
          f"{pos[-1,2]:.3f}/{ori[-1,0]:+.3f}/{ori[-1,1]:+.3f}")


if __name__ == "__main__":
    main()
