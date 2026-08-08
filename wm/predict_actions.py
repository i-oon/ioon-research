"""Reconstruct joint commands for a body the model never trained on.

The world model is an inverse model: z_t = ITM(e_t, e_{t+1}) and a_t = MD(e_t, z_t).
Both frames are ground truth, so this is action *reconstruction* from video, not a
controller -- nothing here chooses what the robot should do, it reads off what it did.

Predictions come back in radians (the checkpoint's action_mean/std undo the
standardisation) so they can be replayed in CoppeliaSim and compared against the IK
commands that generated the clip.

Run from the repository root:
  .venv/bin/python3 -m wm.predict_actions --ckpt wm/runs/<run>/epoch020.pt --clips 3
"""
import argparse
import os
import sys
from dataclasses import fields

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import decode, encode_clip, latents, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = ["TC", "CF", "FT"]
JOINT_NAMES = [f"{leg}_{seg}" for leg in LEGS for seg in SEG]


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in checkpoint["config"].items() if k in known})
    cfg.train_morphs = tuple(cfg.train_morphs)
    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    return cfg, itm, md, checkpoint["action_mean"], checkpoint["action_std"], checkpoint.get("epoch", -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--morph", default="", help="body to reconstruct; defaults to the held-out body")
    parser.add_argument("--clips", type=int, default=3, help="how many clips of that body")
    parser.add_argument("--chunk", type=int, default=2, help="frames per encoder forward")
    parser.add_argument("--encode_device", default="",
                        help="where to run V-JEPA2; set to cpu when a training run holds the GPU")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, itm, md, mean, std, epoch = load_model(args.ckpt, device)
    morph = args.morph or cfg.heldout_morph
    unseen = morph not in cfg.train_morphs

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    paths = clip_paths(data_dir, (morph,))[:args.clips]
    if not paths:
        raise SystemExit(f"no clips for morph '{morph}' in {data_dir}")

    encoder = VJEPA2FrameEncoder(device=args.encode_device or str(device),
                                 dtype=torch.float32 if args.encode_device == "cpu" else torch.float16)
    start, stop = cfg.frame_start, cfg.frame_stop
    out = {"pred": [], "gt": [], "forces": [], "clip": [], "lengths": []}
    for path in paths:
        clip = load_clip(path)
        frames = clip["frames"][start:stop or None]
        actions = clip["actions"][start:stop or None]
        forces = clip["forces"][start:stop or None]

        embeddings = encode_clip(encoder, frames, args.chunk).to(device)
        z = latents(itm, embeddings, args.chunk)
        # the action at t is what carried the robot from frame t to t+1, so the last
        # frame has no action to predict
        pred = decode(md, embeddings[:-1], z, args.chunk) * std + mean

        out["pred"].append(pred.astype(np.float32))
        out["gt"].append(actions[:-1])
        out["forces"].append(forces[:-1])
        out["clip"].append(os.path.basename(path))
        out["lengths"].append(len(pred))

    pred = np.concatenate(out["pred"])
    gt = np.concatenate(out["gt"])
    error = np.degrees(pred - gt)
    print(f"{args.ckpt}  epoch {epoch}")
    print(f"body '{morph}' ({'held out' if unseen else 'seen in training'}), "
          f"{len(paths)} clips, {len(pred)} transitions")
    print(f"RMSE over all joints: {np.sqrt((error ** 2).mean()):.2f} deg")

    out_path = args.out or os.path.join(
        ROOT, "results", "wm", "predictions",
        f"{os.path.basename(os.path.dirname(args.ckpt))}_{os.path.splitext(os.path.basename(args.ckpt))[0]}_{morph}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        pred=pred, gt=gt,
        forces=np.concatenate(out["forces"]),
        lengths=np.array(out["lengths"]),
        clips=np.array(out["clip"]),
        joints=np.array(JOINT_NAMES),
        morph=morph, held_out=unseen, epoch=epoch,
        train_morphs=np.array(list(cfg.train_morphs)),
        ckpt=args.ckpt,
    )
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
