"""Does the latent carry the transition, or only the pose at time t?

The architecture says z = ITM(e_t, e_{t+1}) is a latent *action*: what happened between the
two frames. But the joint command at t is largely determined by where the legs already are,
which one frame shows. If z only re-encodes the pose, the model is a single-frame regressor
wearing a world model's clothes, and nothing it does is about dynamics.

The test replaces e_{t+1} with something that carries no information about the real next
frame and measures how much the reconstructed command moves:

  real        z = ITM(e_t, e_{t+1})           the model as trained

The command compared against follows the checkpoint's own cfg.action_lag, so a run trained on the
corrected target is scored on the corrected target.
  duplicate   z = ITM(e_t, e_t)               no transition at all
  shuffled    z = ITM(e_t, e_{t+k})           a real frame from the wrong time
  reversed    z = ITM(e_t, e_{t-1})           the transition backwards
  zero        z = 0                           the latent removed entirely

If duplicate and shuffled score near the real one, e_{t+1} contributed nothing and z is a
pose code. If they collapse toward the zero-latent number, z is carrying the transition.

  .venv/bin/python3 scripts/z_dynamics.py --ckpt wm/runs/m3d_cross/epoch008.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import decode, encode_clip  # noqa: E402
from wm.predict_actions import load_model  # noqa: E402


@torch.no_grad()
def latents_from(itm, e_t, e_next, chunk=2):
    parts = [itm(e_t[i:i + chunk], e_next[i:i + chunk]) for i in range(0, len(e_t), chunk)]
    return torch.cat(parts)


def rmse_deg(pred, gt):
    return float(np.sqrt((np.degrees(pred - gt) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--morph", default="", help="defaults to the held-out body")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--encode_device", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, itm, md, mean, std, epoch = load_model(args.ckpt, device)
    morph = args.morph or cfg.heldout_morph
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    paths = clip_paths(data_dir, (morph,))[:args.clips]
    if not paths:
        raise SystemExit(f"no clips for morph '{morph}' in {data_dir}")

    encoder = VJEPA2FrameEncoder(device=args.encode_device or str(device),
                                 dtype=torch.float32 if args.encode_device == "cpu" else torch.float16)
    rng = np.random.default_rng(args.seed)
    collected = {k: [] for k in ("real", "duplicate", "shuffled", "reversed", "zero")}
    truth = []

    for path in paths:
        clip = load_clip(path)
        frames = clip["frames"][cfg.frame_start:cfg.frame_stop or None]
        actions = clip["actions"][cfg.frame_start:cfg.frame_stop or None]
        e = encode_clip(encoder, frames, args.chunk).to(device)
        # the command asked for sits action_lag steps past t, so a transition is usable only
        # while both the next frame and that command exist
        lag = cfg.action_lag
        n = min(len(e) - 1, len(actions) - lag)
        e_t = e[:n]

        # e_{t-1}, with the first transition left pointing at itself since it has no past
        back = e[np.clip(np.arange(n) - 1, 0, None)]
        # a real frame from a random other time in the same clip, so appearance statistics
        # are unchanged and only the temporal relationship is destroyed
        perm = rng.permutation(n)
        variants = {
            "real": e[1:n + 1],
            "duplicate": e_t,
            "shuffled": e[1:n + 1][perm],
            "reversed": back,
        }
        for name, e_next in variants.items():
            z = latents_from(itm, e_t, e_next, args.chunk)
            collected[name].append(decode(md, e_t, z, args.chunk) * std + mean)
        z0 = torch.zeros_like(latents_from(itm, e_t[:1], e_t[:1], 1)).repeat(n, 1)
        collected["zero"].append(decode(md, e_t, z0, args.chunk) * std + mean)
        truth.append(actions[lag:lag + n])

    gt = np.concatenate(truth)
    real = np.concatenate(collected["real"])
    print(f"{args.ckpt}  epoch {epoch}")
    print(f"body '{morph}' ({'held out' if morph not in cfg.train_morphs else 'in training'}), "
          f"{len(paths)} clips, {len(gt)} transitions\n")
    print(f"{'what the ITM is given as e_t+1':<34} {'RMSE deg':>9} {'vs real':>9} "
          f"{'z moved':>9}")
    base = rmse_deg(real, gt)
    for name in ("real", "duplicate", "shuffled", "reversed", "zero"):
        pred = np.concatenate(collected[name])
        moved = float(np.sqrt(((pred - real) ** 2).mean()) / (np.abs(real).mean() + 1e-9))
        label = {"real": "e_t+1, the real next frame",
                 "duplicate": "e_t, no transition at all",
                 "shuffled": "a frame from a random other time",
                 "reversed": "e_t-1, the transition backwards",
                 "zero": "-- latent zeroed entirely --"}[name]
        print(f"{label:<34} {rmse_deg(pred, gt):9.2f} {rmse_deg(pred, gt) / base:9.2f}x "
              f"{moved:9.3f}")
    print("\n'z moved' is how far the command moved from the real one, relative to the command's "
          "own scale.\nNear zero means the substituted frame changed nothing.")


if __name__ == "__main__":
    main()
