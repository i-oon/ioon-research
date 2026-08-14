"""Step 0, Check 3 — temporal similarity of frozen V-JEPA2 embeddings.

Merged tool (was temporal_similarity_{heatmap,quantified,correlation}.py). Encodes two
frames independently and probes per-patch cosine similarity. Pick a mode:

  # per-patch cos-sim heatmap of two consecutive frames, overlaid on the frame
  python scripts/temporal_similarity.py --mode heatmap --video clip.mp4 --frame-idx 0 --out heatmap.png
  python scripts/temporal_similarity.py --mode heatmap --demo --out demo_heatmap.png

  # robot-vs-background temporal stability across pairs + paired Wilcoxon (white-bg footage)
  python scripts/temporal_similarity.py --mode quantified --video data/removebg_forward_walk.mp4

  # confound-free: correlate real per-patch pixel motion vs per-patch embedding change
  python scripts/temporal_similarity.py --mode correlation --video data/removebg_forward_walk.mp4 --out motion_vs_embedding.png
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder

CROP = 256
RESIZE = 292  # matches VJEPA2VideoProcessor: shortest_edge=292, center crop 256
CROP_OFFSET = (RESIZE - CROP) // 2
PATCH = 16
GRID = CROP // PATCH  # 16


# --------------------------------------------------------------------------- shared
def load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def cos_sim_grid(encoder, frame_t, frame_t1):
    """Per-patch cosine similarity between the two frames' frozen embeddings -> (16,16)."""
    e = encoder.encode([frame_t, frame_t1])          # (2, 256, 1408)
    sim = torch.nn.functional.cosine_similarity(e[0].float(), e[1].float(), dim=-1)
    return sim.reshape(GRID, GRID).cpu().numpy()


# --------------------------------------------------------------------------- heatmap
def make_demo_frame_pair(size=256):
    """Static noisy background + a moving square (stand-in for a robot leg)."""
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 100, (size, size, 3), dtype=np.uint8)

    def with_square(cx, cy, s=40):
        frame = bg.copy()
        frame[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2] = [220, 40, 40]
        return frame

    return with_square(90, 130), with_square(150, 130)


def load_frame_pair_from_video(path, frame_idx, stride):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + stride); ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise RuntimeError(f"could not read frames {frame_idx},{frame_idx+stride} from {path}")
    return cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)


def save_overlay(frame_t, heatmap, out_path):
    size = frame_t.shape[0]
    heatmap_up = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(frame_t); axes[0].set_title("frame t"); axes[0].axis("off")
    axes[1].imshow(frame_t)
    im = axes[1].imshow(heatmap_up, cmap="RdYlGn", vmin=-1, vmax=1, alpha=0.55)
    axes[1].set_title("cos-sim(e_t, e_t+1)\ngreen=high sim (bg) / red=low sim (motion)")
    axes[1].axis("off"); fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def run_heatmap(args, encoder):
    stride = args.stride if args.stride is not None else 1
    if args.demo or not args.video:
        print("no --video (or --demo) -> synthetic moving-square self-test")
        frame_t, frame_t1 = make_demo_frame_pair()
    else:
        frame_t, frame_t1 = load_frame_pair_from_video(args.video, args.frame_idx, stride)
    heatmap = cos_sim_grid(encoder, frame_t, frame_t1)
    print(f"similarity grid min={heatmap.min():.3f} max={heatmap.max():.3f} mean={heatmap.mean():.3f}")
    save_overlay(frame_t, heatmap, args.out or "heatmap.png")


# --------------------------------------------------------------------------- quantified
def robot_patch_mask(frame_rgb, white_thresh=30, patch_frac_thresh=0.05):
    """Processor-aligned resize+center-crop, then threshold non-white pixels and pool
    into a 16x16 boolean mask of robot-containing patches (white-bg footage)."""
    resized = cv2.resize(frame_rgb, (RESIZE, RESIZE))
    cropped = resized[CROP_OFFSET:CROP_OFFSET + CROP, CROP_OFFSET:CROP_OFFSET + CROP]
    non_white = (255 - cropped.astype(np.int16)).sum(axis=-1) > white_thresh
    mask = np.zeros((GRID, GRID), dtype=bool)
    for i in range(GRID):
        for j in range(GRID):
            mask[i, j] = non_white[i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH].mean() > patch_frac_thresh
    return mask


def run_quantified(args, encoder):
    from scipy.stats import wilcoxon
    stride = args.stride if args.stride is not None else 15
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(args.margin, total - args.margin - stride, args.n_pairs).astype(int)
    robot_means, bg_means = [], []
    for idx in starts:
        f_t, f_t1 = load_frame(cap, idx), load_frame(cap, idx + stride)
        if f_t is None or f_t1 is None:
            continue
        mask = robot_patch_mask(f_t)
        if mask.sum() == 0:
            print(f"[frame {idx}] no robot pixels detected, skipping"); continue
        sim_grid = cos_sim_grid(encoder, f_t, f_t1)
        r, b = sim_grid[mask].mean(), sim_grid[~mask].mean()
        robot_means.append(r); bg_means.append(b)
        print(f"[frame {idx:4d}] robot patches={mask.sum():3d}  robot_sim={r:.3f}  bg_sim={b:.3f}  diff={b-r:+.3f}")
    cap.release()
    robot_means, bg_means = np.array(robot_means), np.array(bg_means)
    print(f"\n=== summary over {len(robot_means)} frame pairs (stride={stride}) ===")
    print(f"robot region: mean={robot_means.mean():.4f} std={robot_means.std():.4f}")
    print(f"bg region:    mean={bg_means.mean():.4f} std={bg_means.std():.4f}")
    print(f"mean(bg - robot) = {(bg_means - robot_means).mean():+.4f}")
    stat, p = wilcoxon(bg_means, robot_means, alternative="greater")
    print(f"\nWilcoxon (H1: bg sim > robot sim): stat={stat:.2f}, p={p:.5f}")
    print("=> SIGNIFICANT: background more temporally stable than robot region." if p < 0.05
          else "=> NOT significant at p<0.05 with this sample size.")


# --------------------------------------------------------------------------- correlation
def aligned_crop(frame_rgb):
    """Processor-aligned resize (shortest edge) + center crop, so pixel space lines up
    with the patch grid (handles non-square frames)."""
    h, w = frame_rgb.shape[:2]
    scale = RESIZE / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(frame_rgb, (new_w, new_h))
    y0, x0 = (new_h - CROP) // 2, (new_w - CROP) // 2
    return resized[y0:y0 + CROP, x0:x0 + CROP]


def pixel_motion_grid(frame_t, frame_t1):
    a = aligned_crop(frame_t).astype(np.float32)
    b = aligned_crop(frame_t1).astype(np.float32)
    diff = np.abs(a - b).mean(axis=-1)
    grid = np.zeros((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            grid[i, j] = diff[i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH].mean()
    return grid


def run_correlation(args, encoder):
    from scipy.stats import pearsonr, spearmanr
    stride = args.stride if args.stride is not None else 15
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = np.linspace(args.margin, total - args.margin - stride, args.n_pairs).astype(int)
    all_motion, all_change = [], []
    for idx in starts:
        f_t, f_t1 = load_frame(cap, idx), load_frame(cap, idx + stride)
        if f_t is None or f_t1 is None:
            continue
        motion = pixel_motion_grid(f_t, f_t1)
        change = 1 - cos_sim_grid(encoder, f_t, f_t1)
        all_motion.append(motion.flatten()); all_change.append(change.flatten())
        r, _ = pearsonr(motion.flatten(), change.flatten())
        print(f"[frame {idx:4d}] corr(pixel_motion, embed_change) = {r:+.3f}")
    cap.release()
    x, y = np.concatenate(all_motion), np.concatenate(all_change)
    pear_r, pear_p = pearsonr(x, y); spear_r, spear_p = spearmanr(x, y)
    print(f"\n=== pooled over {len(starts)} pairs, {len(x)} patches ===")
    print(f"Pearson  r={pear_r:.4f}  p={pear_p:.2e}")
    print(f"Spearman r={spear_r:.4f}  p={spear_p:.2e}")
    print("=> SIGNIFICANT positive correlation: embedding change tracks pixel motion."
          if (pear_p < 0.05 and pear_r > 0) else "=> no significant positive relationship.")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xlabel("per-patch pixel motion |Delta pixel|")
    ax.set_ylabel("per-patch embedding change (1 - cos sim)")
    ax.set_title(f"Pearson r={pear_r:.3f} (p={pear_p:.1e})")
    fig.tight_layout(); fig.savefig(args.out or "motion_vs_embedding.png", dpi=150)
    print(f"saved scatter plot: {args.out or 'motion_vs_embedding.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["heatmap", "quantified", "correlation"])
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=None, help="default: 1 for heatmap, 15 otherwise")
    ap.add_argument("--n-pairs", type=int, default=15)
    ap.add_argument("--margin", type=int, default=60)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.mode in ("quantified", "correlation") and not args.video:
        ap.error(f"--mode {args.mode} requires --video")
    encoder = VJEPA2FrameEncoder()
    {"heatmap": run_heatmap, "quantified": run_quantified, "correlation": run_correlation}[args.mode](args, encoder)


if __name__ == "__main__":
    main()
