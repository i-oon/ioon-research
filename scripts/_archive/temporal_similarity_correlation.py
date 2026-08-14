"""Step 0, Check 3 — confound-free version.

Instead of labeling patches as "robot" vs "background" (which turned out to
be confounded — see temporal_similarity_quantified.py results), this directly
correlates real per-patch pixel motion against per-patch embedding change.
If the encoder is genuinely motion-sensitive, patches with more actual pixel
change should show more embedding change — regardless of what's in them.

Usage:
  python scripts/temporal_similarity_correlation.py --video data/removebg_forward_walk.mp4
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder

CROP = 256
RESIZE = 292  # matches VJEPA2VideoProcessor: shortest_edge=292, center crop 256
CROP_OFFSET = (RESIZE - CROP) // 2
PATCH = 16
GRID = CROP // PATCH  # 16


def load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def aligned_crop(frame_rgb):
    """Same shortest-edge resize + center-crop the processor applies, so pixel
    space lines up with the patch grid (handles non-square frames correctly)."""
    h, w = frame_rgb.shape[:2]
    scale = RESIZE / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(frame_rgb, (new_w, new_h))
    y0 = (new_h - CROP) // 2
    x0 = (new_w - CROP) // 2
    return resized[y0:y0 + CROP, x0:x0 + CROP]


def pixel_motion_grid(frame_t, frame_t1):
    a = aligned_crop(frame_t).astype(np.float32)
    b = aligned_crop(frame_t1).astype(np.float32)
    diff = np.abs(a - b).mean(axis=-1)  # (256,256)
    grid = np.zeros((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            grid[i, j] = diff[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH].mean()
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--n-pairs", type=int, default=15)
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument("--out", type=str, default="motion_vs_embedding.png")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(args.margin, total - args.margin - args.stride, args.n_pairs).astype(int)

    encoder = VJEPA2FrameEncoder()

    all_pixel_motion, all_embed_change = [], []
    for idx in starts:
        f_t = load_frame(cap, idx)
        f_t1 = load_frame(cap, idx + args.stride)
        if f_t is None or f_t1 is None:
            continue

        motion_grid = pixel_motion_grid(f_t, f_t1)  # (16,16) real pixel-space motion

        e = encoder.encode([f_t, f_t1])
        sim = torch.nn.functional.cosine_similarity(e[0].float(), e[1].float(), dim=-1)
        embed_change_grid = (1 - sim.reshape(GRID, GRID).cpu().numpy())  # (16,16)

        all_pixel_motion.append(motion_grid.flatten())
        all_embed_change.append(embed_change_grid.flatten())
        r, _ = pearsonr(motion_grid.flatten(), embed_change_grid.flatten())
        print(f"[frame {idx:4d}] per-frame-pair correlation(pixel_motion, embed_change) = {r:+.3f}")

    cap.release()

    x = np.concatenate(all_pixel_motion)
    y = np.concatenate(all_embed_change)

    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)
    print(f"\n=== pooled over {len(starts)} frame pairs, {len(x)} patches total ===")
    print(f"Pearson  r={pear_r:.4f}  p={pear_p:.2e}")
    print(f"Spearman r={spear_r:.4f}  p={spear_p:.2e}")
    if pear_p < 0.05 and pear_r > 0:
        print("=> SIGNIFICANT positive correlation: embedding change tracks real pixel motion.")
    else:
        print("=> no significant positive relationship detected.")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xlabel("per-patch pixel motion |Δpixel|")
    ax.set_ylabel("per-patch embedding change (1 - cos sim)")
    ax.set_title(f"Pearson r={pear_r:.3f} (p={pear_p:.1e})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved scatter plot: {args.out}")


if __name__ == "__main__":
    main()
