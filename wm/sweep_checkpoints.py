"""Score every saved checkpoint of a run on the full held-out body.

The in-loop held-out probe is small so that it costs nothing per epoch, which leaves it
noisy: at 80 transitions the estimate swung between 0.14 and 0.34 across consecutive epochs.
This re-scores the periodic snapshots against every held-out clip, giving a curve that can be
read rather than guessed at.

Selecting a checkpoint on this curve would leak the held-out body. It is for reporting only.

Run from the repository root:
  .venv/bin/python3 -m wm.sweep_checkpoints --run wm/runs/stage1_100ep_clean
"""
import argparse
import glob
import json
import os
import re
import sys
from dataclasses import fields

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import clip_paths, load_clip  # noqa: E402
from wm.evaluate import upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def load_config(state):
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in state["config"].items() if k in known})
    cfg.train_morphs = tuple(cfg.train_morphs)
    return cfg


@torch.no_grad()
def embed_heldout(encoder, cfg, data_dir, device, max_clips, chunk=8):
    """Encode the held-out body once; every checkpoint is scored against the same frames."""
    start, stop = cfg.frame_start, cfg.frame_stop
    clips = []
    for path in clip_paths(data_dir, (cfg.heldout_morph,))[:max_clips]:
        clip = load_clip(path)
        frames = clip["frames"][start:stop or None]
        parts = [encoder.encode(list(frames[i:i + chunk])).float() for i in range(0, len(frames), chunk)]
        clips.append((
            torch.cat(parts).to(device),
            torch.tensor(clip["actions"][start:stop or None][:-1], device=device),
        ))
    return clips


@torch.no_grad()
def score(itm, md, clips, mean, std, device, chunk=8):
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    errors, ablated, n = 0.0, 0.0, 0
    for embeddings, actions in clips:
        target = (actions - mean_t) / std_t
        total = len(embeddings) - 1
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            e_t, e_next = embeddings[start:stop], embeddings[start + 1:stop + 1]
            expected = target[start:stop]
            z = itm(e_t, e_next)
            size = stop - start
            errors += ((md(e_t, z) - expected) ** 2).mean().item() * size
            ablated += ((md(e_t, torch.zeros_like(z)) - expected) ** 2).mean().item() * size
            n += size
    return errors / n, ablated / n, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory containing epoch*.pt")
    parser.add_argument("--max_clips", type=int, default=40)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    paths = sorted(glob.glob(os.path.join(run_dir, "epoch*.pt")))
    if not paths:
        raise SystemExit(f"no epoch*.pt snapshots in {run_dir}")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    cfg = load_config(first)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)

    encoder = VJEPA2FrameEncoder(device=str(device))
    clips = embed_heldout(encoder, cfg, data_dir, device, args.max_clips)
    del encoder
    torch.cuda.empty_cache()

    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg).to(device).eval()

    rows = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        itm.load_state_dict(state["itm"])
        md.load_state_dict(upgrade_decoder_state(state["md"]))
        mse, ablated, n = score(itm, md, clips, state["action_mean"], state["action_std"], device)
        epoch = int(re.search(r"epoch(\d+)", os.path.basename(path)).group(1))
        rows.append({"epoch": epoch, "with_z": mse, "zero_z": ablated})
        print(f"epoch {epoch:>3}: with_z {mse:.4f}  zero_z {ablated:.4f}  (n={n})")

    out = args.out or os.path.join(run_dir, "heldout_sweep.json")
    with open(out, "w") as handle:
        json.dump({"heldout_morph": cfg.heldout_morph, "n_pairs": rows and n, "curve": rows}, handle, indent=2)
    best = min(rows, key=lambda r: r["with_z"])
    print(f"\nlowest at epoch {best['epoch']} ({best['with_z']:.4f}); final "
          f"{rows[-1]['with_z']:.4f} -> {out}")


if __name__ == "__main__":
    main()
