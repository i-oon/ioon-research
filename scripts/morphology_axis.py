"""Where does a held-out body land between the two training bodies, at each stage of the pipeline?

Two training bodies define a line. Any third body sits somewhere on it, and every stage of the
pipeline places it somewhere: the frozen embedding, the latent, and the decoded joint command.
Comparing those positions to where the body actually belongs says which stage loses the
morphology signal.

Position is a scalar projection onto the axis between the two training bodies' means:

    t = <q - a, b - a> / <b - a, b - a>        t = 0 at body a, t = 1 at body b

Read positions from different spaces as separate statements, not as one decaying quantity: the
axis in embedding space and the axis in joint space are different axes, so 0.465 in one is not
"more" than 0.301 in the other. What is directly comparable is the decoder's position in joint
space against the held-out body's true position in that same space.

The decisive comparison is the ridge probe. It is fitted on the training bodies only, from the
same latent the decoder reads, and it is linear, so it cannot memorise two plateaus. If it
generalises better than the decoder, capacity in the decoder is the problem rather than the
representation.

Run from the repository root:
  .venv/bin/python3 scripts/morphology_axis.py --ckpt wm/runs/<run>/epoch020.pt --clips 3
  .venv/bin/python3 scripts/morphology_axis.py --ckpt ... --encode_device cpu   # GPU busy
"""
import argparse
import json
import os
import sys
from dataclasses import fields

import numpy as np
import torch
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import decode, encode_clip, latents, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

JOINT_TYPES = ("TC", "CF", "FT")
MIN_SEPARATION_DEG = 2.0  # joints the two training bodies barely separate carry no axis


def axis_position(query, ref_a, ref_b):
    """Scalar projection of `query` onto the ref_a -> ref_b axis, all given as mean vectors."""
    direction = ref_b - ref_a
    return float(np.dot(query - ref_a, direction) / np.dot(direction, direction))


def joint_axis_position(query, ref_a, ref_b, offset):
    """Per-joint version, restricted to one joint type and to joints the bodies separate."""
    index = [j for j in range(len(ref_a)) if j % len(JOINT_TYPES) == offset]
    a, b, q = ref_a[index], ref_b[index], query[index]
    usable = np.abs(b - a) > MIN_SEPARATION_DEG
    if not usable.any():
        return float("nan")
    return float(np.mean(((q - a) / (b - a))[usable]))


def rmse_by_joint_type(pred, target):
    out = {}
    for offset, name in enumerate(JOINT_TYPES):
        index = [j for j in range(pred.shape[1]) if j % len(JOINT_TYPES) == offset]
        out[name] = float(np.sqrt(((pred[:, index] - target[:, index]) ** 2).mean()))
    out["all"] = float(np.sqrt(((pred - target) ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", type=int, default=3, help="clips per body, matched by episode")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--encode_device", default="")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty for the probe")
    ap.add_argument("--cache", default="results/wm/axis_embeddings.npz")
    ap.add_argument("--out", default="results/wm/morphology_axis.json")
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in checkpoint["config"].items()
                    if k in {f.name for f in fields(Config)}})
    cfg.train_morphs = tuple(cfg.train_morphs)
    if len(cfg.train_morphs) != 2:
        raise SystemExit(f"an axis needs exactly two training bodies, got {cfg.train_morphs}")
    ref_a, ref_b = cfg.train_morphs
    held = cfg.heldout_morph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    mean, std = checkpoint["action_mean"], checkpoint["action_std"]

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    start, stop = cfg.frame_start, cfg.frame_stop
    bodies = [ref_a, ref_b, held]

    # episodes shared by all three bodies, so the axis compares matched timesteps
    per_body = {b: clip_paths(data_dir, (b,))[:args.clips] for b in bodies}
    episodes = [load_clip(p)["episode"] for p in per_body[held]]

    encoder = VJEPA2FrameEncoder(
        device=args.encode_device or str(device),
        dtype=torch.float32 if args.encode_device == "cpu" else torch.float16)
    E, Z, A, P = {}, {}, {}, {}
    for body in bodies:
        embeds, lat, acts, preds = [], [], [], []
        for episode in episodes:
            path = os.path.join(data_dir, f"{body}_ep{episode}.npz")
            clip = load_clip(path)
            frames = clip["frames"][start:stop or None]
            emb = encode_clip(encoder, frames, args.chunk).to(device)
            z = latents(itm, emb, args.chunk)
            embeds.append(emb[:-1].mean(dim=1).cpu().numpy())
            lat.append(z.cpu().numpy())
            acts.append(np.degrees(clip["actions"][start:stop or None][:-1]))
            preds.append(np.degrees(decode(md, emb[:-1], z, args.chunk) * std + mean))
        E[body], Z[body] = np.concatenate(embeds), np.concatenate(lat)
        A[body], P[body] = np.concatenate(acts), np.concatenate(preds)
        print(f"encoded {body}: {len(A[body])} transitions", flush=True)
    del encoder
    torch.cuda.empty_cache()

    if args.cache:
        cache = args.cache if os.path.isabs(args.cache) else os.path.join(ROOT, args.cache)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **{f"e_{k}": v for k, v in E.items()},
                 **{f"z_{k}": v for k, v in Z.items()})

    positions = {
        "embedding_e": axis_position(E[held].mean(0), E[ref_a].mean(0), E[ref_b].mean(0)),
        "latent_z": axis_position(Z[held].mean(0), Z[ref_a].mean(0), Z[ref_b].mean(0)),
    }
    for offset, name in enumerate(JOINT_TYPES):
        positions[f"decoder_{name}"] = joint_axis_position(
            P[held].mean(0), A[ref_a].mean(0), A[ref_b].mean(0), offset)
        positions[f"correct_{name}"] = joint_axis_position(
            A[held].mean(0), A[ref_a].mean(0), A[ref_b].mean(0), offset)

    latent_train = np.concatenate([Z[ref_a], Z[ref_b]])
    action_train = np.concatenate([A[ref_a], A[ref_b]])
    probe = Ridge(alpha=args.alpha).fit(latent_train, action_train)
    probe_pred = {b: probe.predict(Z[b]) for b in bodies}
    positions["probe_TC"] = joint_axis_position(
        probe_pred[held].mean(0), A[ref_a].mean(0), A[ref_b].mean(0), 0)
    positions["probe_CF"] = joint_axis_position(
        probe_pred[held].mean(0), A[ref_a].mean(0), A[ref_b].mean(0), 1)

    results = {
        "ckpt": args.ckpt, "epoch": int(checkpoint.get("epoch", -1)),
        "trained_on": [ref_a, ref_b], "held_out": held, "episodes": episodes,
        "axis_positions": positions,
        "rmse_deg": {
            body: {"decoder": rmse_by_joint_type(P[body], A[body]),
                   "linear_probe_on_z": rmse_by_joint_type(probe_pred[body], A[body])}
            for body in bodies
        },
        "baseline_mean_of_training_bodies_deg":
            rmse_by_joint_type((A[ref_a] + A[ref_b]) / 2, A[held]),
    }

    print(f"\naxis: 0 = {ref_a}, 1 = {ref_b}; held out '{held}'")
    for key, value in positions.items():
        print(f"  {key:<16} {value: .3f}")
    print(f"\nRMSE deg per joint, '{held}' is never seen by either predictor")
    print(f"{'body':<20}{'decoder':>10}{'probe on z':>13}")
    for body in bodies:
        row = results["rmse_deg"][body]
        flag = " (held out)" if body == held else ""
        print(f"{body + flag:<20}{row['decoder']['all']:>10.2f}{row['linear_probe_on_z']['all']:>13.2f}")
    print(f"{'mean of training':<20}{'':>10}"
          f"{results['baseline_mean_of_training_bodies_deg']['all']:>13.2f}  (baseline on held out)")

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
