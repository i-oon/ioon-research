"""Score one trained checkpoint on several held-out bodies, without retraining anything.

A held-out body is never trained on, so any body absent from a run's `train_morphs` is a valid
test for it -- including bodies that live in a different dataset directory. Retraining once per
test body is not only wasteful, it is weaker: it changes the weights as well as the test, so two
results cannot be attributed to the body alone. One checkpoint against several bodies isolates it.

The case this was written for: `tib_cross` holds out `c10f10t06`, which has a 94.6 mm dead zone
against a 92.5 mm closest target and veers 0.40 m off course. Its collapse could be the
femur/tibia coverage gap or could be the degraded gait. Scoring the same checkpoint on
`c10f10t08` -- ratio 1.04, dead zone 11.8 mm, walks straight -- separates the two.

Reported per body, in degrees per joint and in standardised units where 1.00 is what predicting a
constant pose scores, with both decoder inputs ablated:

  z zeroed      does the answer depend on the latent
  frame zeroed  does the frame help or hurt. On a body outside the training range it has
                previously *hurt*, which is the pathway failure F18 named.

  .venv/bin/python3 scripts/score_body.py --ckpt wm/runs/tib_cross/best.pt \\
      --bodies c10f10t06:data/fwd_hex8body c10f10t08:data/fwd_bracket
"""
import argparse
import glob
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
from wm.data.dataset import EXCLUDED_BODIES  # noqa: E402
from wm.evaluate import encode_clip, offset_for, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def trained_bodies(cfg):
    """Bodies the run actually learned from, so a test body can be refused if it is one.

    Single-morphology runs name them in `train_morphs`. Cross-embodiment runs name a directory
    instead, so the set is whatever it globbed minus the bodies that were excluded for not walking
    and minus the ones deliberately held out.
    """
    if not cfg.sources:
        return set(cfg.train_morphs)
    trained = set()
    for spec in cfg.sources:
        _, _, data_dir = spec.partition("=")
        directory = data_dir if os.path.isabs(data_dir) else os.path.join(ROOT, data_dir)
        trained |= {os.path.basename(p).split("_")[0]
                    for p in glob.glob(os.path.join(directory, "*.npz"))}
    return trained - set(cfg.heldout_bodies or ()) - set(EXCLUDED_BODIES)


@torch.no_grad()
def score(itm, md, encoder, paths, mean, std, action_lag, chunk, head="default",
          offset=None):
    """Squared error per joint, standardised and in degrees, plus the two input ablations."""
    totals = {k: 0.0 for k in ("model", "zero_z", "zero_x", "constant")}
    degrees, count = 0.0, 0
    collected = []
    for path in paths:
        clip = np.load(path)
        # the encoder may run on the GPU; the trained modules are small and stay on the CPU, so
        # bring the embeddings back rather than moving a 1B-parameter model
        e = encode_clip(encoder, clip["frames"], chunk).cpu()
        if offset is not None:
            # a checkpoint trained with `center_embeddings` subtracted a per-embodiment
            # mean before anything saw the frames; scoring it on raw embeddings is a
            # silent distribution shift that reads as the model failing. It cost a
            # reported R^2 of -0.95 once.
            e = e - offset
        actions = clip["actions"].astype(np.float32)
        n = min(len(e) - 1, len(actions) - action_lag)
        target = torch.tensor((actions[action_lag:action_lag + n] - mean) / std)
        for s in range(0, n, 8):
            t = min(s + 8, n)
            e_t, e_next = e[s:t], e[s + 1:t + 1]
            expected = target[s:t]
            z = itm(e_t, e_next)
            pred = md(e_t, z, head)
            totals["model"] += float(((pred - expected) ** 2).sum())
            totals["zero_z"] += float(((md(e_t, torch.zeros_like(z), head) - expected) ** 2).sum())
            totals["zero_x"] += float(((md(torch.zeros_like(e_t), z, head) - expected) ** 2).sum())
            # the no-learning reference: the training set's mean pose, which in standardised
            # units is exactly zero, so this is the target's own energy
            totals["constant"] += float((expected ** 2).sum())
            degrees += float((((pred - expected) * torch.tensor(std)) ** 2).sum())
            count += expected.numel()
            collected.append(expected)
    out = {k: v / count for k, v in totals.items()}
    out["deg"] = float(np.rad2deg(np.sqrt(degrees / count)))
    # Two different constant baselines, and slide 8's R^2 claim uses the second:
    #   constant  the *training* mean pose, which is zero in these units. Beating it only says
    #             the model noticed the body is not an average body.
    #   own_mean  this body's *own* per-joint mean, the baseline R^2 is defined against. A model
    #             above this is worse than someone who had seen the body once and memorised its
    #             average posture, which is the claim that matters for transfer.
    target = torch.cat(collected)
    out["own_mean"] = float(((target - target.mean(dim=0)) ** 2).mean())
    out["r2"] = 1.0 - out["model"] / out["own_mean"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bodies", nargs="+", required=True,
                    help="body:data_dir, e.g. c10f10t08:data/fwd_bracket")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--embodiment", default="hexapod",
                    help="which head to score through, for cross-embodiment checkpoints")
    ap.add_argument("--encode_device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).eval()
    itm.load_state_dict(checkpoint["itm"])

    # A cross-embodiment checkpoint keys its heads and its action statistics by embodiment; a
    # single-morphology one has one of each. The held-out body is a hexapod either way, so a
    # Stage 2 model is scored through its hexapod head against the hexapod statistics.
    if "action_stats" in checkpoint:
        stats = checkpoint["action_stats"]
        md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).eval()
        mean, std = (np.asarray(v, dtype=np.float32) for v in stats[args.embodiment])
        trained_on = f"cross-embodiment {list(stats)}, scored through the '{args.embodiment}' head"
        head = args.embodiment
    else:
        md = MotionDecoder(cfg).eval()
        mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["action_std"], dtype=np.float32)
        trained_on = f"trained on {list(cfg.train_morphs)}"
        head = "default"
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))

    print(f"{args.ckpt}  epoch {checkpoint.get('epoch', -1)}")
    print(f"{trained_on}, action_lag {cfg.action_lag}, lambda_cross {cfg.lambda_cross}\n")

    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    print(f'{"body":12}{"deg":>8}{"model":>9}{"train mean":>12}{"own mean":>10}'
          f'{"R2":>7}{"zero_z":>9}{"zero_x":>9}')
    for spec in args.bodies:
        body, _, data_dir = spec.partition(":")
        if body in trained_bodies(cfg):
            print(f"{body:12}  SKIPPED: this body was trained on, so it is not a held-out test")
            continue
        directory = data_dir if os.path.isabs(data_dir) else os.path.join(ROOT, data_dir)
        paths = sorted(glob.glob(os.path.join(directory, f"{body}_ep*.npz")))[:args.clips]
        if not paths:
            print(f"{body:12}  no clips found in {data_dir}")
            continue
        r = score(itm, md, encoder, paths, mean, std, cfg.action_lag, args.chunk, head,
                  offset_for(checkpoint, args.embodiment))
        print(f'{body:12}{r["deg"]:8.2f}{r["model"]:9.3f}{r["constant"]:12.3f}'
              f'{r["own_mean"]:10.3f}{r["r2"]:+7.2f}{r["zero_z"]:9.3f}{r["zero_x"]:9.3f}')
    del encoder
    print("\n'train mean' is what predicting the training mean pose scores; beating it only says\n"
          "the model noticed this is not an average body. 'own mean' is this body's own per-joint\n"
          "mean, the baseline R2 is defined against -- a negative R2 means worse than someone who\n"
          "saw the body once and memorised its average posture. 'zero_x' below 'model' means the\n"
          "frame is actively harmful.")


if __name__ == "__main__":
    main()
