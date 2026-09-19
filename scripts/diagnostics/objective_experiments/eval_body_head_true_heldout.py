"""Score a fitted body_head against the REAL, external held-out set -- not fit_body_head.py's own
internal --val_frac split, which silently measures something else entirely.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt \\
        --embodiment b1 \\
        --train_dir data/egocentric/beh12_b1_ego_flat_cleantrain \\
        --heldout_dir data/egocentric/beh12_b1_ego_flat_cleanheldout \\
        --also_embodiment hexapod \\
        --also_train_dir data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \\
        --also_heldout_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout

**`--conditions`**: restrict scoring to a comma-separated subset of conditions, in both `--data`
and `--also` pools. Added to compare a checkpoint trained on a bigger condition set (e.g. beh24)
against one trained on a smaller one (beh12) using only the conditions they both have -- otherwise
the two checkpoints' overall ratios are answering different-difficulty questions (F223's addendum),
not a fair head-to-head.

**Why this script exists and is not just `fit_body_head.py --epochs 0`.** `fit_body_head.py`'s own
`report()` splits "held out" by taking `--val_frac` (default 0.2) of whichever `--data`/`--also`
directory it was GIVEN -- a random, non-stratified subset of *whatever pool it's pointed at*. Point
it at a train-only directory (as the clean retrain's stage 4 does, on purpose, since the real
held-out check is meant to happen separately against `scripts/dataset/make_clean_split.py`'s own
stratified `_cleanheldout` set) and its "held out" number silently becomes an internal split of the
TRAIN pool -- not the real held-out set at all. F222 (doc/FINDINGS.md) reported that number as the
real held-out result; it wasn't. This script never splits anything: every clip in `--train_dir` is
scored as train, every clip in `--heldout_dir` (a directory that must never appear in any training
call for this checkpoint) is scored as held out, full stop.

**Reusable for a clip-count sweep, which is the actual next question.** Run this after each
`wm.adapt`/`wm.fit_body_head` pair built on a different number of adaptation clips, holding
`--heldout_dir` fixed throughout -- the only way the sweep's held-out number stays comparable
across points, since a resweep that also moves the test set would confound "more data" with
"different questions asked."

Diagnosis only. No tuning, no retraining.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.config import from_checkpoint            # noqa: E402
from wm.data.embodiment import REGISTRY, load     # noqa: E402
from wm.evaluate import encode_clip               # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder     # noqa: E402


def embed_and_target(encoder, itm, paths, spec, channels, chunk, cache_path):
    """Every (e_t, e_t+1) -> ITM -> z, paired with that transition's true body_motion. One row of
    output per transition, not per clip -- matches fit_body_head.py's own unit exactly, so the two
    scripts' numbers are comparable."""
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    zs, ys = [], []
    device = next(itm.parameters()).device
    with torch.no_grad():
        for path in paths:
            clip = load(path, spec)
            if path not in cache:
                cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu().half()
            e = cache[path].float().to(device)
            motion = np.asarray(clip["body_motion"])[:, channels]
            n = min(len(e) - 1, len(motion) - 1)
            z = torch.cat([itm(e[t:t + 1], e[t + 1:t + 2]) for t in range(n)]).cpu()
            zs.append(z)
            ys.append(torch.tensor(motion[:n], dtype=torch.float32))
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return torch.cat(zs), torch.cat(ys)


def score(md, z, y, mean, std, device):
    y_std = ((y - mean) / std).to(device)
    with torch.no_grad():
        pred = md.body(None, z.to(device))
    err = torch.nn.functional.mse_loss(pred, y_std).item()
    base = torch.nn.functional.mse_loss(y_std.mean(0, keepdim=True).expand_as(y_std), y_std).item()
    # raw-unit (actual Froude, not standardized) MSE too -- the ratio's denominator uses THIS
    # checkpoint's own std, which differs across checkpoints trained on different-width target
    # distributions (see --conditions docstring); raw MSE is checkpoint-independent and is what
    # settles whether a ratio gap reflects a real absolute-error difference or just a shrunk
    # denominator on a narrow subgroup.
    pred_raw = (pred.cpu() * std) + mean
    raw_mse = torch.nn.functional.mse_loss(pred_raw, y).item()
    return err, base, err / max(base, 1e-9), raw_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--heldout_dir", required=True)
    ap.add_argument("--also_embodiment", default="")
    ap.add_argument("--also_train_dir", default="")
    ap.add_argument("--also_heldout_dir", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--cache_dir", default="results/wm/cache")
    ap.add_argument("--conditions", default="",
                    help="comma-separated condition names to restrict scoring to (both --data "
                         "and --also pools) -- e.g. for a subset comparison between two datasets "
                         "that only share some conditions. Empty means no filtering.")
    args = ap.parse_args()
    conditions = set(c for c in args.conditions.split(",") if c) or None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
    std = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)

    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)

    action_dim = int(load(glob.glob(os.path.join(ROOT, args.train_dir, "*.npz"))[0],
                         REGISTRY[args.embodiment])["actions"].shape[1])
    md = MotionDecoder(cfg, {args.embodiment: action_dim}).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    if md.body_head is None:
        raise SystemExit("this checkpoint has no body_head")

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    def filter_conditions(paths):
        if conditions is None:
            return paths
        kept = []
        for p in paths:
            with np.load(p, allow_pickle=True) as d:
                if str(d["condition"]) in conditions:
                    kept.append(p)
        return kept

    def run(embodiment, train_dir, heldout_dir):
        spec = REGISTRY[embodiment]
        train_paths = filter_conditions(sorted(glob.glob(os.path.join(ROOT, train_dir, "*.npz"))))
        held_paths = filter_conditions(sorted(glob.glob(os.path.join(ROOT, heldout_dir, "*.npz"))))
        overlap = set(os.path.basename(p) for p in train_paths) & \
                 set(os.path.basename(p) for p in held_paths)
        if overlap:
            raise SystemExit(f"{embodiment}: train/heldout overlap, refusing to score: {overlap}")
        cache = os.path.join(ROOT, args.cache_dir, f"eval_true_heldout_{embodiment}.pt")
        z_tr, y_tr = embed_and_target(encoder, itm, train_paths, spec, channels, args.chunk, cache)
        z_he, y_he = embed_and_target(encoder, itm, held_paths, spec, channels, args.chunk, cache)
        err_tr, base_tr, ratio_tr, raw_tr = score(md, z_tr, y_tr, mean, std, device)
        err_he, base_he, ratio_he, raw_he = score(md, z_he, y_he, mean, std, device)
        print(f"{embodiment:<10} train ({len(train_paths):>2} clips, {len(z_tr):>4} transitions)"
             f"  MSE {err_tr:.4f}  mean {base_tr:.4f}  ratio {ratio_tr:.3f}  raw-unit MSE {raw_tr:.5f}")
        print(f"{embodiment:<10} TRUE HELD-OUT ({len(held_paths):>2} clips, {len(z_he):>4} "
             f"transitions)  MSE {err_he:.4f}  mean {base_he:.4f}  ratio {ratio_he:.3f}"
             f"  raw-unit MSE {raw_he:.5f}")
        return ratio_tr, ratio_he

    print(f"checkpoint: {args.ckpt}\n")
    run(args.embodiment, args.train_dir, args.heldout_dir)
    if args.also_embodiment:
        print()
        run(args.also_embodiment, args.also_train_dir, args.also_heldout_dir)

    del encoder
    print("\nratio is against predicting the target's mean. Below 1.0 on TRUE HELD-OUT is the")
    print("only line that means anything -- train and fit_body_head.py's own internal 'held out'")
    print("column are both optimistic by construction (train: seen directly; internal held-out:")
    print("drawn from the same pool the fit was calibrated on, not a separate directory).")


if __name__ == "__main__":
    main()
