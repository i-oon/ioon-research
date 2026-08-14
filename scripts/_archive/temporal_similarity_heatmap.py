"""Step 0, Check 3 — Temporal Similarity Heatmap.

Encodes two consecutive frames independently, computes per-patch cosine
similarity between them, and visualizes it as a 16x16 heatmap overlaid on
the frame. Prediction: background patches (unchanged) -> high similarity;
robot/leg patches (moved) -> low similarity.

Usage:
  # against real footage (e.g. a recorded B1 walk clip)
  python scripts/temporal_similarity_heatmap.py --video path/to/clip.mp4 --frame-idx 0 --out heatmap.png

  # self-test with a synthetic moving-square clip (no real footage needed yet)
  python scripts/temporal_similarity_heatmap.py --demo --out demo_heatmap.png
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


def load_frame_pair_from_video(path, frame_idx, stride=1):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok1, f1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + stride)
    ok2, f2 = cap.read()
    cap.release()
    if not (ok1 and ok2):
        raise RuntimeError(f"could not read frames {frame_idx},{frame_idx+stride} from {path}")
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
    return f1, f2


def make_demo_frame_pair(size=256):
    """Static noisy background + a moving square (stand-in for a robot leg)."""
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 100, (size, size, 3), dtype=np.uint8)  # static background

    def with_square(cx, cy, s=40):
        frame = bg.copy()
        frame[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2] = [220, 40, 40]
        return frame

    frame_t = with_square(90, 130)
    frame_t1 = with_square(150, 130)  # square moved right
    return frame_t, frame_t1


def compute_similarity_heatmap(encoder, frame_t, frame_t1, grid=16):
    e = encoder.encode([frame_t, frame_t1])  # (2, 256, 1408)
    e_t, e_t1 = e[0].float(), e[1].float()
    sim = torch.nn.functional.cosine_similarity(e_t, e_t1, dim=-1)  # (256,)
    return sim.reshape(grid, grid).cpu().numpy()


def save_overlay(frame_t, heatmap, out_path):
    size = frame_t.shape[0]
    heatmap_up = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(frame_t)
    axes[0].set_title("frame t")
    axes[0].axis("off")

    axes[1].imshow(frame_t)
    im = axes[1].imshow(heatmap_up, cmap="RdYlGn", vmin=-1, vmax=1, alpha=0.55)
    axes[1].set_title("cos-sim(e_t, e_t+1)\ngreen=high sim (bg) / red=low sim (motion)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", type=str, default="heatmap.png")
    args = ap.parse_args()

    if args.demo or not args.video:
        print("no --video given (or --demo set) -> running synthetic moving-square self-test")
        frame_t, frame_t1 = make_demo_frame_pair()
    else:
        frame_t, frame_t1 = load_frame_pair_from_video(args.video, args.frame_idx, args.stride)

    encoder = VJEPA2FrameEncoder()
    heatmap = compute_similarity_heatmap(encoder, frame_t, frame_t1)

    print(f"similarity grid min={heatmap.min():.3f} max={heatmap.max():.3f} mean={heatmap.mean():.3f}")
    save_overlay(frame_t, heatmap, args.out)


if __name__ == "__main__":
    main()
