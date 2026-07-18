"""Whole-frame e_t UMAP across different renderers/domains, same behavior (walking).

All 3 clips show walking locomotion but come from different rendering setups
(background, lighting, engine). If the pooled embedding is dominated by
rendering style rather than anything behavior-relevant, frames should cluster
tightly by source video. If the representation is more style-robust, frames
should mix.

Usage:
  python scripts/umap_domain_check.py
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap

from vjepa2_encoder import VJEPA2FrameEncoder

VIDEOS = {
    "removebg (white bg)": "data/removebg_forward_walk.mp4",
    "isaac play-step-0 (grid bg)": "data/play-step-0_realtime.mp4",
    "mujoco light-ood (checker bg)": "data/light_ood_mujoco.mp4",
}
N_FRAMES_PER_VIDEO = 40
MARGIN = 60


def sample_frames(path, n, margin=MARGIN):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(margin, total - margin, n).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    encoder = VJEPA2FrameEncoder()

    all_vecs = []
    all_labels = []
    for label, path in VIDEOS.items():
        frames = sample_frames(path, N_FRAMES_PER_VIDEO)
        print(f"{label}: {len(frames)} frames sampled from {path}")
        # batch through the encoder in chunks to keep memory sane
        pooled = []
        chunk = 8
        for i in range(0, len(frames), chunk):
            batch = frames[i:i + chunk]
            e = encoder.encode(batch)  # (B, 256, 1408)
            pooled.append(e.mean(dim=1).float().cpu())  # mean-pool over patches -> (B, 1408)
        pooled = torch.cat(pooled, dim=0).numpy()
        all_vecs.append(pooled)
        all_labels += [label] * len(pooled)

    X = np.concatenate(all_vecs, axis=0)
    labels = np.array(all_labels)
    print(f"\nTotal frames: {X.shape[0]}, embedding dim: {X.shape[1]}")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    emb2d = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label in VIDEOS:
        mask = labels == label
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], label=label, s=25, alpha=0.7)
    ax.set_title("UMAP of whole-frame e_t (mean-pooled), same behavior (walk),\ndifferent rendering domains")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = "domain_umap.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
