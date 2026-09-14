"""Evaluate a trained PPO policy (`b1_real_ppo.py`'s `--out`) in real physics, deterministically --
per this project's own standing rule (verify before asserting), a training curve alone is not
proof of a working controller.

    .venv/bin/python3 wm/policy/b1_ppo_eval.py \\
        --policy wm/runs/beh12_hinge_multistep_anchor_v2/b1_ppo_mujoco.pt \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1.pt

**Deterministic, not sampled.** Training uses the actor's stochastic Gaussian sample (for
exploration); reporting performance uses the mean action only -- the policy's actual intended
behaviour, not a noisy draw from it.

**Checks the REAL Froude, not just the proxy reward.** `info["true_froude"]` is measured directly
from MuJoCo's own physics (body_velocity/yaw_rate on the real qpos trajectory), completely
independent of the world-model reward the policy was trained against. If the policy is actually
tracking the vision-derived goal in real physics (not just exploiting some quirk of
`body_head(proj(action))`), true_froude should land close to the goal clip's own real motion.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_mujoco_env import B1MuJoCoEnv, OBS_DIM, ACTION_DIM  # noqa: E402
from b1_real_ppo import Actor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true",
                    help="open MuJoCo's interactive viewer and play the policy live, "
                         "roughly real-time. Needs a display -- won't work over plain SSH "
                         "without X forwarding. Reduces --episodes to 1 automatically unless "
                         "set explicitly, since watching is usually a single-episode thing.")
    ap.add_argument("--video", default=None,
                    help="write an offscreen mp4 of the first episode here. Unlike --render this "
                         "needs no display, which is the usual case over SSH -- and a policy is "
                         "not reported as walking until someone has watched one of these.")
    args = ap.parse_args()
    if args.render and "--episodes" not in sys.argv:
        args.episodes = 1

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    saved = torch.load(os.path.join(ROOT, args.policy), map_location="cpu", weights_only=False)
    actor = Actor(OBS_DIM, ACTION_DIM).to(device)
    actor.load_state_dict(saved["actor"])
    actor.eval()
    obs_mean, obs_std = saved["obs_mean"], saved["obs_std"]

    env = B1MuJoCoEnv(args.ckpt, horizon=args.horizon)
    print(f"goal (standardized Froude, fwd/lat/yaw): {env.goal_std}\n")

    viewer_ctx = None
    if args.render:
        import mujoco.viewer
        import time as _time
        viewer_ctx = mujoco.viewer.launch_passive(env.m, env.d)

    renderer, frames = None, []
    if args.video:
        import mujoco
        renderer = mujoco.Renderer(env.m, 480, 640)

    lens, survived, all_true_fr, fall_counts, heights, travels = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs = env.reset()
        if viewer_ctx is not None:
            viewer_ctx.sync()
        done, steps = False, 0
        ep_true_fr = []
        # **Count fall EVENTS, not the final-step flag.** Falls no longer end an episode (they cost
        # a penalty and reset the pose), so `fell` at the last step says nothing: a policy that
        # falls every two seconds still finishes with `fell=False` and scored "100% survival" until
        # this was fixed. Track the pose height too -- a collapsed crouch that lurches forward can
        # look like locomotion in the displacement number alone.
        n_falls, ep_h, prev_x = 0, [], float(env.d.qpos[0])
        ep_travel = 0.0
        while not done:
            obs_n = (obs - obs_mean) / obs_std
            obs_t = torch.as_tensor(obs_n, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mu, _ = actor(obs_t)         # deterministic: mean action, tanh-squashed
                action = torch.tanh(mu).squeeze(0).numpy()
            obs, reward, done, info = env.step(action)
            if viewer_ctx is not None:
                viewer_ctx.sync()
                _time.sleep(env.dt)          # roughly real-time playback
            if renderer is not None and ep == 0:
                renderer.update_scene(env.d, camera=-1)
                frames.append(renderer.render())
            ep_true_fr.append(info["true_froude"])
            ep_h.append(info["height"])
            if info["fell"]:
                n_falls += 1
            x = float(env.d.qpos[0])
            step_dx = x - prev_x
            if abs(step_dx) < 0.05:          # skip the teleport a fall-reset causes
                ep_travel += step_dx
            prev_x = x
            steps += 1
        lens.append(steps)
        survived.append(n_falls == 0)
        fall_counts.append(n_falls)
        heights.append(float(np.mean(ep_h)))
        travels.append(ep_travel)
        all_true_fr.append(np.mean(ep_true_fr, axis=0))
        print(f"episode {ep}: steps={steps}  falls={n_falls}  mean_height={np.mean(ep_h):.3f}  "
             f"travel={ep_travel:+.3f}m  mean_true_froude={np.mean(ep_true_fr, axis=0)}")

    all_true_fr = np.asarray(all_true_fr)
    print(f"\nepisodes with NO fall: {np.mean(survived):.1%}   mean falls per episode: "
         f"{np.mean(fall_counts):.2f}   mean height: {np.mean(heights):.3f} (nominal stand 0.56)")
    print(f"forward travel, teleports excluded: {np.mean(travels):+.3f} m "
         f"({np.mean(travels) / (args.horizon * 0.02):+.3f} m/s; a normal B1 walk is ~0.29 m/s)")
    print(f"mean episode length: {np.mean(lens):.1f}/{args.horizon}")
    print(f"mean true Froude across episodes (fwd/lat/yaw): {all_true_fr.mean(0)}")
    print(f"goal Froude was standardized -- compare the SIGN and RELATIVE magnitude of the "
         f"channels above, not raw units, since env.goal_std lives in body_head's normalized space.")

    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=int(round(1 / env.dt)))
        print(f"video: {args.video}")

    if viewer_ctx is not None:
        viewer_ctx.close()


if __name__ == "__main__":
    main()
