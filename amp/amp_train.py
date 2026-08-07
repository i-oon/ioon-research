"""AMP training for the stick insect — self-contained (copied from the lab repo; we don't touch it).

reward = scale * [disc.g(s') + w * task_reward].  Actor/discriminator keep the original 28-D
normalised observation; the asymmetric critic receives one private velocity-context scalar.

PREREQ: CoppeliaSim GUI (launched from the venv) with the insect scene OPEN on --port, e.g.
  sim/env/medauroidea_stick_insect.ttt (long/base body).

  python3 amp/amp_train.py --port 23000 --steps 2000000 --name insect_long

Stage B (walk/turn/stop) wraps this env with a command + task reward — added after walk validates.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CWD = os.getcwd()                           # where the user ran from (for --scene resolution)
sys.path.insert(0, HERE)
os.chdir(HERE)                                   # relative logs/ resolve inside amp/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--eval_interval", type=int, default=10_000)
    ap.add_argument("--rollout", type=int, default=1000)
    ap.add_argument("--name", type=str, default="insect_amp")
    ap.add_argument("--scene", type=str, default="",
                    help="scene .ttt to load before training (headless launches empty)")
    ap.add_argument("--disc", type=str, default=os.path.join(HERE, "discriminator.pth"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vx_coef", type=float, default=10.0,
                    help="[speed mode] velocity reward coefficient. reward = g + vx*vx_coef.")
    # --- bounded tracking task reward combined with the frozen raw AIRL g(s') ---
    ap.add_argument("--reward_mode", choices=["track", "speed"], default="track",
                    help="track = bounded direction+target-precision reward (default); speed = vx*vx_coef (legacy)")
    ap.add_argument("--vx_target", type=float, default=0.45,
                    help="REFERENCE (long-body) target fwd speed m/s; env auto-scales it by this body's leg length")
    ap.add_argument("--leg_ref", type=float, default=0.7717, help="reference (long) leg length m for target scaling")
    ap.add_argument("--sigma", type=float, default=0.15, help="tracking tolerance m/s (bigger = more forgiving)")
    ap.add_argument("--track_window", type=int, default=25,
                    help="[track mode] steps over which forward velocity is AVERAGED (net displacement/time). "
                         "Kills oscillation-gaming: rocking cancels over the window, only sustained travel pays. "
                         "Longer = harder to fake but slower reward onset.")
    ap.add_argument("--track_direction_mix", type=float, default=0.5,
                    help="mix of clipped directional progress in track reward; remainder is Gaussian precision")
    ap.add_argument("--lam_min", type=float, default=10.0,
                    help="task-reward weight. Default equals --lam_max, so weight is constant.")
    ap.add_argument("--lam_max", type=float, default=10.0,
                    help="final task-reward weight; keep equal to --lam_min for the original constant-w form.")
    ap.add_argument("--lam_warmup_frac", type=float, default=0.0,
                    help="optional fraction of --steps held at lam_min before ramping")
    ap.add_argument("--lam_ramp_frac", type=float, default=0.0,
                    help="optional fraction of --steps over which lambda ramps up")
    ap.add_argument("--g_clip", type=float, default=0.0,
                    help="optional upper clip on raw g output; default 0 follows the original unclipped g(s'). Scored "
                         "the raw discriminator on its own 66k-frame expert dataset -- expert g reaches "
                         "mean=2.91, std=1.55, p95=6.13, max=7.23, and high-g expert frames are normal "
                         "mid-stride motion (corr(g, per-frame displacement) = -0.28, NOT static poses). The "
                         "old clip=3.0 was silently discarding ~40%% of legitimate expert-quality variation, "
                         "leaving PPO almost no gradient to prefer good gait over mediocre gait.")
    ap.add_argument("--g_baseline", type=float, default=0.0,
                    help="optional constant subtracted from g; default 0 preserves raw g(s')")
    ap.add_argument("--g_center", type=int, default=0,
                    help="1 enables gait-reward centering; default 0 matches the original transfer objective")
    ap.add_argument("--g_ood_coef", type=float, default=2.0,
                    help="soft reliability decay for g only outside the expert normalized support; 0 disables")
    ap.add_argument("--reward_scale", type=float, default=0.05,
                    help="positive global scale applied after g + w*task; changes critic target scale, not balance")
    args = ap.parse_args()

    os.environ["AMP_DISC"] = args.disc               # ppo_transfer reads this for the frozen gait prior
    from algorithms.ppo_transfer import PPO
    from common.normalized_env_66k import CoppeliaSimEnv
    from common.trainer import Trainer

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.name, ts)    # tensorboard events land here
    os.makedirs(log_dir, exist_ok=True)
    print(f"AMP {args.name} @ {ts} | port {args.port} | disc={args.disc} | "
          f"reward_mode={args.reward_mode} g_clip={args.g_clip} (exact reward printed after env init)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.scene:                                   # headless starts empty -> load the body scene
        scene = args.scene if os.path.isabs(args.scene) else os.path.join(ORIG_CWD, args.scene)
        import time as _t
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        _sim = RemoteAPIClient("localhost", port=args.port).require("sim")
        _sim.stopSimulation()                        # loadScene requires a stopped sim
        while _sim.getSimulationState() != 0:
            _t.sleep(0.1)
        _sim.loadScene(scene)
        print("loaded scene:", scene)
    env = CoppeliaSimEnv(port=args.port, OnTimeStep=True, vx_coef=args.vx_coef,
                         reward_mode=args.reward_mode, vx_target=args.vx_target, track_sigma=args.sigma,
                         leg_ref=args.leg_ref, track_window=args.track_window,
                         track_direction_mix=args.track_direction_mix)
    lam_warmup = int(args.lam_warmup_frac * args.steps)
    lam_ramp = int(args.lam_ramp_frac * args.steps)
    if args.reward_mode == "track":
        print(f"actor/disc obs {env.observation_space.shape}  critic obs "
              f"({env.observation_space.shape[0] + env.critic_context_dim},)  action {env.action_space.shape}\n"
              f"reward = {args.reward_scale} * (support(s')*g(s') + lambda(step)*track(vx_avg{args.track_window}))  "
              f"| track={args.track_direction_mix}*clip(vx/target,-1,1)+{1-args.track_direction_mix}*Gaussian  "
              f"| raw in-support g (OOD coef={args.g_ood_coef}, center={bool(args.g_center)}, output_clip={args.g_clip or 'off'})  "
              f"| vx_avg = net head displacement over {args.track_window} steps / time (anti-rocking)  "
              f"| lambda {args.lam_min}->{args.lam_max}, warmup {lam_warmup}, ramp {lam_ramp} steps")
    else:
        print(f"obs {env.observation_space.shape}  action {env.action_space.shape}  "
              f"reward = g(s') + vx*{args.vx_coef}  (speed mode)")

    algo = PPO(state_shape=env.observation_space.shape,
               action_shape=env.action_space.shape,
               device=device, seed=args.seed, gamma=0.995,
               rollout_length=args.rollout, mix_buffer=1,
               lr_actor=3e-4, lr_critic=1e-4,
               units_actor=(64, 64), units_critic=(64, 64),
               epoch_ppo=50, clip_eps=0.2, lambd=0.97, coef_ent=0.01,
               max_grad_norm=10.0, mini_batch_size=64,
               lam_min=args.lam_min, lam_max=args.lam_max,
               lam_warmup=lam_warmup, lam_ramp=lam_ramp,
               g_clip=(args.g_clip if args.g_clip > 0 else None),
               g_center=bool(args.g_center), g_baseline=args.g_baseline,
               disc_state_dim=env.disc_observation_dim,
               critic_context_dim=env.critic_context_dim,
               reward_scale=args.reward_scale,
               g_ood_coef=args.g_ood_coef)

    trainer = Trainer(env=env, env_test=env, algo=algo, log_dir=log_dir,
                      num_steps=args.steps, eval_interval=args.eval_interval)
    trainer.train()
    algo.save_models(trainer.model_dir)
    print("done ->", trainer.model_dir)


if __name__ == "__main__":
    main()
