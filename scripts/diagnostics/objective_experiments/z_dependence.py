"""Does the shared Froude head use z? Head error with z shuffled across pairs vs with the true z.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/z_dependence.py --ckpt wm/runs/X/best.pt \\
        --embodiment hexapod --dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt

Every stride-k pair (e_t, e_t+k) of every clip in --dir from its first_pair on; z = ITM(e_t, e_t+k), target =
mean body_motion[t:t+k] on the checkpoint's body channels, standardised with its body_stats (the training
label, wm/data/dataset.py). Pairs are taken in fixed random batches of 64; within each batch z is rolled
by one, so every frame is paired with another pair's z (the frame stays). ratio = MSE(shuffled) / MSE(true):
~1 means the head ignores z (reads the frame only); a z-only head (body_sees_frame False) is also scored,
since it ignores the frame instead. Embeddings come from --cache (the eval_suite selection caches, keyed by
path); clips missing from it are encoded (GPU).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from wm.config import from_checkpoint  # noqa: E402
from wm.data.emb_cache import load_cache, note, save_cache  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.data.strided import first_pair_of, stride_of  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(dev).eval(); itm.load_state_dict(ck["itm"])
    md = MotionDecoder(cfg, {}).to(dev).eval(); md.load_state_dict(ck["md"], strict=False)
    if md.body_head is None:
        raise SystemExit("no body head in this checkpoint")
    ch = [int(c) for c in cfg.body_channels]
    mean, std = [np.asarray(x, dtype=np.float32).ravel()[:len(ch)] for x in ck["body_stats"]]
    off = offset_for(ck, args.embodiment)
    k = stride_of(cfg)

    cache_path = os.path.join(ROOT, args.cache)
    emb = load_cache(cache_path)
    paths = sorted(glob.glob(os.path.join(ROOT, args.dir, "*.npz")))
    encoder, dirty = None, False
    X, Z, Y = [], [], []
    for p in paths:
        clip = load(p, REGISTRY[args.embodiment], lazy_frames=True)
        if p not in emb:
            if encoder is None:
                from vjepa2_encoder import VJEPA2FrameEncoder
                encoder = VJEPA2FrameEncoder(dtype=torch.float32)
            emb[p] = encode_clip(encoder, load(p, REGISTRY[args.embodiment])["frames"], 2).cpu().half()
            note(emb, p); dirty = True
        e = emb[p].float()
        if off is not None:
            e = e - off.float().reshape(1, *e.shape[1:])
        bm = np.asarray(clip["body_motion"], dtype=np.float32)[:, ch]
        s0 = first_pair_of(clip)
        n = min(len(e), len(bm)) - k - s0
        if n <= 0:
            continue
        ts = np.arange(s0, s0 + n)
        for i in range(0, n, 32):
            t = ts[i:i + 32]
            et = e[t].to(dev)
            Z.append(itm(et, e[t + k].to(dev)).float().cpu())
            X.append(e[t].half())
        Y.append(torch.as_tensor((np.stack([bm[t:t + k].mean(0) for t in ts]) - mean) / std))
    if dirty:
        save_cache(emb, cache_path)
    X, Z, Y = torch.cat(X), torch.cat(Z), torch.cat(Y)
    order = torch.randperm(len(Z), generator=torch.Generator().manual_seed(0))
    se_true = se_shuf = 0.0
    for i in range(0, len(order), args.batch):
        idx = order[i:i + args.batch]
        if len(idx) < 2:
            continue
        x, z, y = X[idx].float().to(dev), Z[idx].to(dev), Y[idx].to(dev)
        se_true += ((md.body(x, z) - y) ** 2).mean(-1).sum().item()
        se_shuf += ((md.body(x, z.roll(1, dims=0)) - y) ** 2).mean(-1).sum().item()
    n = len(order)
    head = "head(e_t, z)" if md.body_sees_frame else "head(z)"
    print(f"z-dependence {args.embodiment}: {head} error z-shuffled/true {se_shuf / max(se_true, 1e-12):.3f}x "
          f"(MSE {se_true / n:.4f} -> {se_shuf / n:.4f}, {n} pairs; ~1 = head ignores z)")


if __name__ == "__main__":
    main()
