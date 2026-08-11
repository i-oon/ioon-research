"""Validate the trained latent action space (roadmap steps 1d and 1f).

Held-out transfer: the frozen ITM and motion decoder see only the held-out body's frames
and predict its joint commands; the recorded IK actions are used solely to score them.

Two-sided probe: a useful z_t should raise cross-morphology behaviour transfer while
lowering morphology decodability, both measured against raw e_t as the baseline.

Run from the repository root:
  .venv/bin/python3 -m wm.evaluate --ckpt wm/runs/stage1_6ep_clipped/best.pt
"""
import argparse
import json
import os
import sys
from dataclasses import fields

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config, from_checkpoint  # noqa: E402
from wm.data.dataset import clip_paths, contact_labels, load_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

TOP_PATTERNS = 8


def upgrade_decoder_state(state):
    """Checkpoints predating the multi-embodiment refactor store a single `head.*`; the
    decoder now keys its output layers by embodiment."""
    prefix = "head."
    if not any(k.startswith(prefix) for k in state):
        return state
    return {
        f"heads.default.{k[len(prefix):]}" if k.startswith(prefix) else k: v
        for k, v in state.items()
    }


def encode_clip(encoder, frames, chunk=16):
    embeddings = []
    for start in range(0, len(frames), chunk):
        embeddings.append(encoder.encode(list(frames[start:start + chunk])).float())
    return torch.cat(embeddings)


def offset_for(checkpoint, embodiment):
    """The appearance offset a `center_embeddings` run was trained with, or None.

    A model trained on centred embeddings and then scored on raw ones sees a shifted input
    distribution and reports quietly wrong numbers rather than failing, so every script that
    encodes frames for a checkpoint has to ask for this.
    """
    offsets = checkpoint.get("embedding_offsets")
    return offsets[embodiment] if offsets else None


@torch.no_grad()
def latents(itm, embeddings, chunk):
    total = len(embeddings) - 1
    parts = []
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        parts.append(itm(embeddings[start:stop], embeddings[start + 1:stop + 1]))
    return torch.cat(parts)


@torch.no_grad()
def decode(md, embeddings, z, chunk):
    parts = [md(embeddings[i:i + chunk], z[i:i + chunk]) for i in range(0, len(z), chunk)]
    return torch.cat(parts).cpu().numpy()


@torch.no_grad()
def collect(encoder, itm, md, paths, mean, std, device, chunk=8, seed=0, frame_range=(0, 0),
            action_lag=1):
    """Latents, motion predictions and probe features for every transition in `paths`.

    Alongside the real prediction, the decoder is also run with a zeroed and a shuffled
    latent. If those score nearly as well, the decoder is reading the joint command off the
    frame rather than from z, and the motion loss is not grounding the latent action.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = ("e", "z", "pred", "pred_zero", "pred_shuffled", "target", "contact")
    records = {key: [] for key in keys}
    records["morph"] = []

    start, stop = frame_range
    for path in paths:
        clip = load_clip(path)
        # score the same frame range the model was trained on, or the mismatch is measured
        # instead of the model
        frames = clip["frames"][start:stop or None]
        actions = clip["actions"][start:stop or None]
        forces = clip["forces"][start:stop or None]
        embeddings = encode_clip(encoder, frames, chunk).to(device)
        e_t = embeddings[:len(embeddings) - max(1, action_lag)]

        z = latents(itm, embeddings, chunk)[:len(e_t)]
        permutation = torch.randperm(len(z), generator=generator).to(device)

        records["pred"].append(decode(md, e_t, z, chunk))
        records["pred_zero"].append(decode(md, e_t, torch.zeros_like(z), chunk))
        records["pred_shuffled"].append(decode(md, e_t, z[permutation], chunk))
        records["e"].append(e_t.mean(dim=1).cpu().numpy())
        records["z"].append(z.cpu().numpy())
        # the command that caused the transition sits action_lag steps past t; z is defined as
        # that transition, so this is what the decoder is asked to recover
        n = len(embeddings) - max(1, action_lag)
        records["target"].append((actions[action_lag:action_lag + n] - mean) / std)
        records["contact"].append(contact_labels(forces[action_lag:action_lag + n]))
        records["morph"] += [clip["morph"]] * len(z)

    out = {key: np.concatenate(records[key]) for key in keys}
    out["morph"] = np.array(records["morph"])
    return out


JOINT_TYPES = ("TC", "CF", "FT")


def per_joint_type(pred, target):
    """Split the motion error by joint type, leg-major order (FL TC/CF/FT, ML ...).

    The aggregate hides which joints transferred. Measured on the held-out medium body,
    thorax-coxa scored 0.006 while coxa-femur scored 0.382 -- worse than predicting that
    joint's own mean -- yet the average across all 18 read 0.208 and looked healthy.
    Thorax-coxa swings the leg fore and aft by a similar angle whatever the leg length;
    the two distal joints set how high and how far the foot goes, which is what leg length
    changes, so they are the ones a cross-morphology claim rests on.

    The baseline is that joint's own mean over the clip, not the training mean, so a score
    above 1.0 means the prediction is worse than a constant.
    """
    scores = {}
    for offset, name in enumerate(JOINT_TYPES):
        index = list(range(offset, pred.shape[1], len(JOINT_TYPES)))
        error = ((pred[:, index] - target[:, index]) ** 2).mean()
        constant = ((target[:, index].mean(axis=0) - target[:, index]) ** 2).mean()
        scores[name] = {
            "mse": float(error),
            "constant_baseline": float(constant),
            "times_better_than_constant": float(constant / max(error, 1e-9)),
        }
    return scores


def behaviour_labels(contact):
    codes = np.array(["".join(map(str, row)) for row in contact])
    values, counts = np.unique(codes, return_counts=True)
    keep = set(values[np.argsort(-counts)[:TOP_PATTERNS]])
    mask = np.array([c in keep for c in codes])
    return codes, mask


def decode_accuracy(features, labels):
    scaled = StandardScaler().fit_transform(features)
    return float(cross_val_score(LogisticRegression(max_iter=2000), scaled, labels, cv=3).mean())


def structure(features, labels):
    """Presence and dominance are different questions: a probe can read a signal out at high
    accuracy while that signal explains almost none of the variance. Reporting only one of
    them gives the wrong answer, so both are returned."""
    scaled = StandardScaler().fit_transform(features)
    total = scaled.var(axis=0).sum()
    within = sum(
        (labels == value).mean() * scaled[labels == value].var(axis=0).sum()
        for value in np.unique(labels)
    )
    return {
        "decode": decode_accuracy(features, labels),
        # subsampled: silhouette is O(n^2) in pairwise distances
        "silhouette": float(silhouette_score(scaled, labels, sample_size=5000, random_state=0)),
        "between_class_variance": float(1.0 - within / total),
    }


def transfer_f1(features, labels, morph, held_out):
    train = morph != held_out
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(features[train]), labels[train])
    predicted = model.predict(scaler.transform(features[~train]))
    return float(f1_score(labels[~train], predicted, average="macro"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="results/wm")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(Config)}
    cfg = from_checkpoint(checkpoint["config"])
    cfg.train_morphs = tuple(cfg.train_morphs)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    mean, std = checkpoint["action_mean"], checkpoint["action_std"]

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    encoder = VJEPA2FrameEncoder(device=str(device))
    morphs = tuple(cfg.train_morphs) + (cfg.heldout_morph,)
    paths = clip_paths(data_dir, morphs)
    data = collect(encoder, itm, md, paths, mean, std, device,
                   frame_range=(cfg.frame_start, cfg.frame_stop))

    variants = {"with_z": "pred", "zero_z": "pred_zero", "shuffled_z": "pred_shuffled"}
    results = {"trained_on": list(cfg.train_morphs), "held_out": cfg.heldout_morph,
               "motion_mse": {}, "motion_mse_per_joint_type": {}}
    for body in morphs:
        mask = data["morph"] == body
        target = data["target"][mask]
        scores = {name: float(((data[key][mask] - target) ** 2).mean()) for name, key in variants.items()}
        # standardisation uses the training bodies' statistics, so predicting the training
        # mean costs 1.0 only on those bodies; on a held-out body the trivial score is
        # whatever that body's own posture happens to cost, and it is the honest baseline
        scores["predict_training_mean"] = float((target ** 2).mean())
        scores["predict_this_body_mean"] = float(((target.mean(axis=0) - target) ** 2).mean())
        results["motion_mse"][body] = scores
        if target.shape[1] % len(JOINT_TYPES) == 0:
            results["motion_mse_per_joint_type"][body] = per_joint_type(data["pred"][mask], target)

    codes, keep = behaviour_labels(data["contact"])
    results["morphology_structure"] = {
        "from_e": structure(data["e"], data["morph"]),
        "from_z": structure(data["z"], data["morph"]),
        "decode_chance": 1.0 / len(morphs),
    }
    results["behaviour_transfer_macro_f1"] = {
        body: {
            "from_e": transfer_f1(data["e"][keep], codes[keep], data["morph"][keep], body),
            "from_z": transfer_f1(data["z"][keep], codes[keep], data["morph"][keep], body),
        }
        for body in morphs
    }

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
