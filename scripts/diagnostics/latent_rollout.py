"""Can the forward model actually roll the world forward, given the actions?

Everything measured so far asked the forward model the wrong question. It was scored on
whether it improves action reconstruction, and it does not (F23, F30). But that is not what
a world model is for. Its job is to answer "if I apply this action, what happens next" --
which is what makes planning possible, and which nothing so far has tested.

This closes the loop on its own output:

    e_t --[FTM, z_t]--> ê_{t+1} --[FTM, z_{t+1}]--> ê_{t+2} --> ... --> ê_{t+k}

and compares ê_{t+k} against the true e_{t+k}, on un-augmented frames, against two
baselines that require no learning at all:

  hold      ê_{t+k} = e_t                          nothing moves
  linear    ê_{t+k} = e_t + k (e_t - e_{t-1})      constant velocity in embedding space

The latents fed in are the real ones the ITM infers from the true transitions, so this
measures the forward model alone, not the latent's quality.

Reading it: if the rollout cannot beat `hold`, the forward model has learned nothing usable
and the case for dropping it is closed. If it beats both out to five or ten steps, then it
did learn dynamics and the current objective simply never asks for them -- which is a very
different conclusion, and the one that would keep the world-model framing alive.

  .venv/bin/python3 scripts/latent_rollout.py --ckpt wm/runs/m3d_cross/epoch008.pt
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

from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def load(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    ftm = ForwardTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    ftm.load_state_dict(checkpoint["ftm"])
    return cfg, itm, ftm, checkpoint.get("epoch", -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--morph", default="")
    ap.add_argument("--data_dir", default="",
                    help="override cfg.data_dir; a cross-embodiment checkpoint carries "
                         "stale single-morphology defaults there")
    ap.add_argument("--embodiment", default="",
                    help="required only for checkpoints trained with ftm_embodiment_channel")
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10])
    ap.add_argument("--encode_device", default="cpu")
    args = ap.parse_args()

    device = torch.device("cpu" if args.encode_device == "cpu" else "cuda")
    cfg, itm, ftm, epoch = load(args.ckpt, device)
    # a `center_embeddings` checkpoint saw a per-embodiment mean subtracted; feeding it
    # raw embeddings is a silent distribution shift, not an error
    offset = offset_for(torch.load(args.ckpt, map_location='cpu', weights_only=False),
                        args.embodiment or 'hexapod')
    offset = offset.to(device) if offset is not None else None
    morph = args.morph or cfg.heldout_morph
    raw = args.data_dir or cfg.data_dir
    data_dir = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
    paths = clip_paths(data_dir, (morph,))[:args.clips]
    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)

    horizons = sorted(args.horizons)
    scores = {name: {k: [] for k in horizons} for name in ("rollout", "hold", "linear")}

    for path in paths:
        clip = load_clip(path)
        e = encode_clip(encoder, clip["frames"], 2).to(device)
        if offset is not None:
            e = e - offset
        n = len(e)
        # the latent for each transition, inferred from the true frames
        z = torch.cat([itm(e[i:i + 1], e[i + 1:i + 2]) for i in range(n - 1)])

        for start in range(1, n - max(horizons) - 1):
            predicted = e[start:start + 1]
            velocity = e[start] - e[start - 1]
            for step in range(1, max(horizons) + 1):
                predicted = ftm(predicted, z[start + step - 1:start + step],
                                args.embodiment or None)
                if step in scores["rollout"]:
                    truth = e[start + step]
                    scores["rollout"][step].append(((predicted[0] - truth) ** 2).mean().item())
                    scores["hold"][step].append(((e[start] - truth) ** 2).mean().item())
                    scores["linear"][step].append(
                        ((e[start] + step * velocity - truth) ** 2).mean().item())

    print(f"{args.ckpt}  epoch {epoch}")
    print(f"body '{morph}', {len(paths)} clips, un-augmented frames, "
          f"{len(scores['rollout'][horizons[0]])} rollouts\n")
    print(f'{"steps ahead":>12} {"forward model":>14} {"hold still":>12} {"constant vel":>13} '
          f'{"vs hold":>9}')
    for k in horizons:
        roll = float(np.mean(scores["rollout"][k]))
        hold = float(np.mean(scores["hold"][k]))
        lin = float(np.mean(scores["linear"][k]))
        print(f'{k:>12} {roll:14.3f} {hold:12.3f} {lin:13.3f} {hold / roll:8.2f}x')

    print("\n'vs hold' above 1 means the forward model beats doing nothing. Below 1 means it is\n"
          "worse than assuming the world is frozen, and has learned nothing it can roll forward.")


if __name__ == "__main__":
    main()
