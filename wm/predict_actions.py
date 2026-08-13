"""Reconstruct joint commands for a body the model never trained on.

The world model is an inverse model: z_t = ITM(e_t, e_{t+1}) and a_hat = MD(e_t, z_t).
Both frames are ground truth, so this is action *reconstruction* from video, not a
controller -- nothing here chooses what the robot should do, it reads off what it did.

Which command a_hat is compared against follows cfg.action_lag. The collector captures
frames[t] after applying actions[t], so the command that caused frames[t] -> frames[t+1] is
actions[t+1], and that is what a latent describing the transition should decode to.

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

from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import (decode, encode_clip, latents, offset_for,  # noqa: E402
                         upgrade_decoder_state)
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = ["TC", "CF", "FT"]
JOINT_NAMES = [f"{leg}_{seg}" for leg in LEGS for seg in SEG]


def load_model(ckpt_path, device, embodiment="hexapod"):
    """Rebuild the trained modules, for either a single-morphology or a cross-embodiment run.

    A Stage 2 checkpoint keys its output heads and action statistics by embodiment, so it needs
    the head names to construct the decoder and the right statistics to un-standardise with.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    if "action_stats" in checkpoint:
        stats = checkpoint["action_stats"]
        md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).to(device).eval()
        mean, std = stats[embodiment]
        head = embodiment
    else:
        md = MotionDecoder(cfg).to(device).eval()
        mean, std = checkpoint["action_mean"], checkpoint["action_std"]
        head = "default"
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    return cfg, itm, md, mean, std, checkpoint.get("epoch", -1), head, checkpoint



def identity_basis_for(itm, encoder, checkpoint, args, cfg, device, n_dirs=8):
    """The directions in `z` that carry embodiment identity, peeled off one at a time.

    Same construction as `scripts/z_identity_ablation.py`: fit a linear probe for which
    embodiment a latent came from, remove its direction, refit, repeat. Needs both embodiments
    present, so it encodes a few clips of each.
    """
    import glob
    from sklearn.linear_model import LogisticRegression
    from wm.evaluate import training_bodies

    groups = []
    for spec in cfg.sources:
        name, _, data_dir = spec.partition("=")
        directory = data_dir if os.path.isabs(data_dir) else os.path.join(ROOT, data_dir)
        if name == "hexapod":
            bodies = training_bodies(cfg)
            paths = [p for b in bodies
                     for p in sorted(glob.glob(os.path.join(directory, f"{b}_ep*.npz")))[:2]]
        else:
            paths = sorted(glob.glob(os.path.join(directory, "*.npz")))[:4]
        groups.append((name, paths))

    Z, label = [], []
    for i, (name, paths) in enumerate(groups):
        for path in paths:
            # only frames are needed here, and `load_clip` is hexapod-specific -- the B1 clips
            # key their commands as `action`, not `actions`
            with np.load(path, allow_pickle=True) as data:
                clip_frames = data["frames"]
            e = encode_clip(encoder, clip_frames, args.chunk).to(device)
            off = offset_for(checkpoint, name)
            if off is not None:
                e = e - off.to(device)
            z = latents(itm, e, args.chunk).cpu().numpy()
            Z.append(z); label += [i] * len(z)
    Z = np.concatenate(Z); label = np.array(label)

    work, basis = Z.copy(), []
    for _ in range(n_dirs):
        w = LogisticRegression(max_iter=3000).fit(work, label).coef_[0]
        for b in basis:
            w = w - (w @ b) * b
        norm = np.linalg.norm(w)
        if norm < 1e-8:
            break
        w = w / norm
        basis.append(w)
        work = work - np.outer(work @ w, w)
    return np.stack(basis)


def project_identity(z, basis):
    """The component of `z` lying in the identity subspace, to be subtracted."""
    B = torch.tensor(basis, dtype=z.dtype, device=z.device)
    return (z @ B.T) @ B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--morph", default="", help="body to reconstruct; defaults to the held-out body")
    parser.add_argument("--clips", type=int, default=3, help="how many clips of that body")
    parser.add_argument("--chunk", type=int, default=2, help="frames per encoder forward")
    parser.add_argument("--encode_device", default="",
                        help="where to run V-JEPA2; set to cpu when a training run holds the GPU")
    parser.add_argument("--out", default="")
    parser.add_argument("--data_dir", default="",
                        help="override cfg.data_dir; a cross-embodiment checkpoint carries stale "
                             "single-morphology defaults there")
    parser.add_argument("--embodiment", default="hexapod",
                        help="which output head to decode through, for Stage 2 checkpoints")
    parser.add_argument("--ablate", default="none",
                        choices=("none", "zero_z", "zero_x", "no_identity"),
                        help="what to remove before decoding, so the *behaviour* cost of an "
                             "ablation can be replayed rather than only scored")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, itm, md, mean, std, epoch, head, checkpoint = load_model(
        args.ckpt, device, args.embodiment)
    morph = args.morph or cfg.heldout_morph
    unseen = morph not in (cfg.train_morphs or ())

    raw_dir = args.data_dir or cfg.data_dir
    data_dir = raw_dir if os.path.isabs(raw_dir) else os.path.join(ROOT, raw_dir)
    paths = clip_paths(data_dir, (morph,))[:args.clips]
    if not paths:
        raise SystemExit(f"no clips for morph '{morph}' in {data_dir}")

    encoder = VJEPA2FrameEncoder(device=args.encode_device or str(device),
                                 dtype=torch.float32)
    # float32 throughout: every diagnostic (score_body, the ablations, the probes) encodes in
    # float32, and predictions meant to be compared against them have to match. Encoding this
    # clip set in float16 read 14.51 deg where score_body reads 3.85 on the same checkpoint.
    start, stop = cfg.frame_start, cfg.frame_stop
    basis = None
    if args.ablate == "no_identity":
        basis = identity_basis_for(itm, encoder, checkpoint, args, cfg, device)
        print(f"removing {len(basis)} identity directions from z before decoding")

    out = {"pred": [], "gt": [], "forces": [], "clip": [], "lengths": []}
    for path in paths:
        clip = load_clip(path)
        frames = clip["frames"][start:stop or None]
        actions = clip["actions"][start:stop or None]
        forces = clip["forces"][start:stop or None]

        embeddings = encode_clip(encoder, frames, args.chunk).to(device)
        offset = offset_for(checkpoint, args.embodiment)
        if offset is not None:
            embeddings = embeddings - offset.to(device)
        lag = max(1, cfg.action_lag)
        n = len(embeddings) - lag
        z = latents(itm, embeddings, args.chunk)[:n]
        e_in = embeddings[:n]

        # The point of these is to be *replayed*, not scored. A number says the latent costs
        # 7.63x; a video says what that looks like as a walk. Ajan Blink asked for the second.
        if args.ablate == "zero_z":
            z = torch.zeros_like(z)
        elif args.ablate == "zero_x":
            e_in = torch.zeros_like(e_in)
        elif args.ablate == "no_identity":
            z = z - project_identity(z, basis)

        pred = decode(md, e_in, z, args.chunk, head) * std + mean

        out["pred"].append(pred.astype(np.float32))
        out["gt"].append(actions[cfg.action_lag:cfg.action_lag + n])
        out["forces"].append(forces[cfg.action_lag:cfg.action_lag + n])
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
