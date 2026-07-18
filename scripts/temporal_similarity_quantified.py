"""Step 0, Check 3 — quantified version.

Samples multiple frame pairs across a clip, auto-derives a robot-vs-background
mask from pixel color (valid for the white-background removebg footage), and
compares mean per-patch cosine similarity between the two regions across
pairs, with a paired significance test.

Usage:
  python scripts/temporal_similarity_quantified.py --video data/removebg_forward_walk.mp4
"""
import argparse

import cv2
import numpy as np
import torch
from scipy.stats import wilcoxon

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


def robot_patch_mask(frame_rgb, white_thresh=30, patch_frac_thresh=0.05):
    """Replicate the processor's resize+center-crop, then threshold non-white
    pixels and pool into a 16x16 boolean mask of robot-containing patches."""
    resized = cv2.resize(frame_rgb, (RESIZE, RESIZE))
    cropped = resized[CROP_OFFSET:CROP_OFFSET + CROP, CROP_OFFSET:CROP_OFFSET + CROP]
    non_white = (255 - cropped.astype(np.int16)).sum(axis=-1) > white_thresh  # (256,256) bool
    mask = np.zeros((GRID, GRID), dtype=bool)
    for i in range(GRID):
        for j in range(GRID):
            patch = non_white[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]
            mask[i, j] = patch.mean() > patch_frac_thresh
    return mask, cropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--n-pairs", type=int, default=15)
    ap.add_argument("--margin", type=int, default=60, help="skip this many frames at start/end")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(args.margin, total - args.margin - args.stride, args.n_pairs).astype(int)

    encoder = VJEPA2FrameEncoder()

    robot_means, bg_means = [], []
    for idx in starts:
        f_t = load_frame(cap, idx)
        f_t1 = load_frame(cap, idx + args.stride)
        if f_t is None or f_t1 is None:
            continue

        mask, _ = robot_patch_mask(f_t)
        if mask.sum() == 0:
            print(f"[frame {idx}] no robot pixels detected, skipping")
            continue

        e = encoder.encode([f_t, f_t1])
        sim = torch.nn.functional.cosine_similarity(e[0].float(), e[1].float(), dim=-1)
        sim_grid = sim.reshape(GRID, GRID).cpu().numpy()

        robot_mean = sim_grid[mask].mean()
        bg_mean = sim_grid[~mask].mean()
        robot_means.append(robot_mean)
        bg_means.append(bg_mean)
        print(f"[frame {idx:4d}] robot patches={mask.sum():3d}  robot_sim={robot_mean:.3f}  bg_sim={bg_mean:.3f}  diff={bg_mean - robot_mean:+.3f}")

    cap.release()

    robot_means = np.array(robot_means)
    bg_means = np.array(bg_means)
    n = len(robot_means)
    print(f"\n=== summary over {n} frame pairs (stride={args.stride}) ===")
    print(f"robot region: mean={robot_means.mean():.4f} std={robot_means.std():.4f}")
    print(f"bg region:    mean={bg_means.mean():.4f} std={bg_means.std():.4f}")
    print(f"mean(bg - robot) = {(bg_means - robot_means).mean():+.4f}")

    stat, p = wilcoxon(bg_means, robot_means, alternative="greater")
    print(f"\nWilcoxon signed-rank (H1: bg similarity > robot similarity): stat={stat:.2f}, p={p:.5f}")
    if p < 0.05:
        print("=> SIGNIFICANT: background is consistently more temporally stable than the robot region.")
    else:
        print("=> NOT significant at p<0.05 with this sample size.")


if __name__ == "__main__":
    main()
