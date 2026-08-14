"""Is the latent still a behaviour representation after lambda_cross, or has it been hollowed out?

Stage 2 rests on z carrying behaviour across bodies. lambda_cross drives the decoder onto the
frame so completely that removing z costs it only 2.2-3.2x, against 21x in the control. Two
read-outs decide whether that means z is empty or merely no longer carrying the body:
foot-contact pattern decodable from z, and how z's variance splits between gait and body.

The bodies and the dataset come from each checkpoint's own config. They used to be a literal list
in this file, and that list still held `c10f10t06` and `c06f10t06` long after both were found to
veer 0.35-0.40 m off course -- so the variance split quoted on slide 6 was computed partly on
robots that do not walk. A body list written into a script is correct only until the next run
changes its split, and nothing warns you when it stops being (FINDINGS.md F42).

  .venv/bin/python3 scripts/z_content.py --ckpt wm/runs/m3d_cross/best.pt \\
      wm/runs/m3d_bracketed/best.pt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.bodies import contact_labels  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import available_episodes, clip_paths, load_clip  # noqa: E402
from wm.evaluate import behaviour_labels, encode_clip, latents  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def run_bodies(ckpt_path):
    """The training bodies and dataset of one checkpoint, read off the checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    cfg.train_morphs = tuple(cfg.train_morphs)
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    return checkpoint, cfg, sorted(cfg.train_morphs), data_dir


def shared_episodes(data_dir, bodies, count):
    """Episodes every body has, so the variance split compares like with like.

    The decomposition needs each body at the same timestep of the same expert episode; without
    that, the "gait" term absorbs whatever the bodies happened to be doing at different moments.
    """
    common = None
    for body in bodies:
        eps = set(available_episodes(data_dir, (body,)))
        common = eps if common is None else common & eps
    return sorted(common)[:count]


def encode(encoder, data_dir, bodies, episodes, chunk):
    """Frozen embeddings and contact labels per body, clips kept separate.

    Concatenating clips first would create a transition across each clip boundary, which is not a
    transition at all.
    """
    embeddings, contacts = {}, {}
    for body in bodies:
        paths = [p for p in clip_paths(data_dir, (body,))
                 if load_clip(p)["episode"] in episodes]
        embeddings[body] = [encode_clip(encoder, load_clip(p)["frames"], chunk).float().cpu()
                            for p in paths]
        contacts[body] = [contact_labels(load_clip(p)["forces"][:-1]) for p in paths]
    return embeddings, contacts


def latents_for(checkpoint, cfg, bodies, embeddings, contacts, device, chunk):
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    Z, labels, body_id = [], [], []
    with torch.no_grad():
        for i, body in enumerate(bodies):
            for clip_e, clip_c in zip(embeddings[body], contacts[body]):
                n = min(len(clip_e) - 1, len(clip_c))
                z = latents(itm, clip_e[:n + 1].to(device), chunk)[:n]
                Z.append(z.cpu().numpy())
                labels.append(clip_c[:n])
                body_id += [i] * n
    del itm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(Z), np.concatenate(labels), np.array(body_id)


def variance_split(Z, n_bodies):
    """Two-way split of z's variance into a body term, a gait-phase term and the remainder.

    Bodies are paired timestep by timestep, so the row mean is "what this body does on average"
    and the column mean is "what every body does at this phase".
    """
    per = len(Z) // n_bodies
    stack = Z[:per * n_bodies].reshape(n_bodies, per, -1)
    centred = stack - stack.reshape(-1, Z.shape[-1]).mean(0)
    body = (centred.mean(1) ** 2).sum() * per
    phase = (centred.mean(0) ** 2).sum() * n_bodies
    rest = ((centred - centred.mean(1)[:, None] - centred.mean(0)[None]) ** 2).sum()
    total = body + phase + rest
    return 100 * phase / total, 100 * body / total, 100 * rest / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=3,
                        help="how many shared expert episodes per body")
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--encode_device", default="",
                        help="set to cpu when a training run holds the GPU")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # every checkpoint has to be scored on the same bodies and clips, or the numbers are not
    # comparable; take the first checkpoint's set and refuse if another disagrees
    checkpoint, cfg, bodies, data_dir = run_bodies(args.ckpt[0])
    for other in args.ckpt[1:]:
        _, _, other_bodies, other_dir = run_bodies(other)
        if other_bodies != bodies or other_dir != data_dir:
            raise SystemExit(
                f"{other} trained on {other_bodies} in {os.path.basename(other_dir)}, "
                f"{args.ckpt[0]} on {bodies} in {os.path.basename(data_dir)}. "
                f"Scoring them together would compare two different measurements.")

    episodes = shared_episodes(data_dir, bodies, args.episodes)
    print(f"bodies {bodies}")
    print(f"data   {os.path.relpath(data_dir, ROOT)}, episodes {episodes}")

    encoder = VJEPA2FrameEncoder(device=args.encode_device or str(device), dtype=torch.float32)
    embeddings, contacts = encode(encoder, data_dir, bodies, episodes, args.chunk)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("encoded", flush=True)

    for path in args.ckpt:
        checkpoint, cfg, _, _ = run_bodies(path)
        Z, labels, body_id = latents_for(checkpoint, cfg, bodies, embeddings, contacts,
                                         device, args.chunk)
        codes, keep = behaviour_labels(labels)
        n_classes = len(set(codes[keep]))
        majority = max(np.bincount(np.unique(codes[keep], return_inverse=True)[1])) / keep.sum()
        behaviour = cross_val_score(LogisticRegression(max_iter=3000),
                                    Z[keep], codes[keep], cv=5).mean()
        body = cross_val_score(LogisticRegression(max_iter=3000), Z, body_id, cv=5).mean()
        phase_pct, body_pct, rest_pct = variance_split(Z, len(bodies))

        name = os.path.relpath(path, os.path.join(ROOT, "wm", "runs"))
        print(f"\n--- {name}  epoch {checkpoint.get('epoch', '?')}  "
              f"lambda_cross {cfg.lambda_cross} ---")
        print(f"  behaviour from z : {behaviour:.4f}   over {n_classes} contact patterns, "
              f"majority class {majority:.3f}")
        print(f"  body from z      : {body:.4f}   (chance {1 / len(bodies):.3f})")
        print(f"  variance of z    : gait {phase_pct:.1f}%   body {body_pct:.1f}%   "
              f"rest {rest_pct:.1f}%")


if __name__ == "__main__":
    main()
