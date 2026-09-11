"""Checks whether a trained `convgru_full_retrain.py` checkpoint collapsed to predicting near-zero
delta-Froude regardless of the action -- the same failure mode this codebase calls `moves` ratio
everywhere else (`near 0 is the collapse where the model copies its input, which scores well on
MSE and cannot control anything`). `convgru_full_retrain.py` did not print this, which is a real
gap: a training loss near 0.0000 is consistent with genuine convergence OR with exactly this
collapse, and the kill-gate's real-vs-mean-action gap cannot tell them apart on its own (a
collapsed model also produces a near-zero gap, for a different reason than "recurrence doesn't
help").

    .venv/bin/python3 scripts/diagnostics/objective_experiments/convgru_collapse_check.py \\
        --ckpt wm/runs/convgru_full/convgru_full.pt
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

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.convgru_ftm import SpatialRecurrentModel  # noqa: E402

BODY_CHANNELS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--data", default="")
    ap.add_argument("--n_clips", type=int, default=10)
    ap.add_argument("--seq_len", type=int, default=8)
    args = ap.parse_args()

    data_dir = args.data or {
        "hexapod": "data/egocentric/beh12_c10f10t10_ego_flat",
        "b1": "data/egocentric/beh12_b1_ego_flat",
    }[args.embodiment]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    print("checkpoint args:", ck.get("args"))
    print("checkpoint embodiments:", ck.get("embodiments"))
    model = SpatialRecurrentModel(ck["action_dims"], grid=16, body_dim=3)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz")))[:args.n_clips]
    reg = REGISTRY[args.embodiment]

    pred_norms, true_norms, pred_all, true_all = [], [], [], []
    with torch.no_grad():
        for p in paths:
            clip = load(p, reg)
            e = encode_clip(encoder, clip["frames"], 2).float().to(device)
            bm = torch.tensor(np.asarray(clip["body_motion"])[:, BODY_CHANNELS], dtype=torch.float32)
            mean, std = bm.mean(0), bm.std(0).clamp_min(1e-6)
            bm = ((bm - mean) / std).to(device)
            actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32).to(device)
            n = min(len(e), len(bm), len(actions))
            if n < args.seq_len + 2:
                continue
            h = model.init_hidden(1, device)
            for t in range(1, args.seq_len + 1):
                h, pred = model.step(h, e[t:t + 1], actions[t:t + 1], args.embodiment)
                true_delta = bm[t + 1:t + 2] - bm[t:t + 1]
                pred_norms.append(pred.norm(dim=1).cpu())
                true_norms.append(true_delta.norm(dim=1).cpu())
                pred_all.append(pred.cpu()); true_all.append(true_delta.cpu())

    pred_norms = torch.cat(pred_norms); true_norms = torch.cat(true_norms)
    pred_all = torch.cat(pred_all); true_all = torch.cat(true_all)
    moves = (pred_norms.mean() / true_norms.mean().clamp_min(1e-9)).item()

    print(f"\npred delta norm, mean: {pred_norms.mean().item():.4f}  std: {pred_norms.std().item():.4f}")
    print(f"true delta norm, mean: {true_norms.mean().item():.4f}  std: {true_norms.std().item():.4f}")
    print(f"moves ratio (pred/true): {moves:.4f}")
    print(f"pred std per channel: {pred_all.std(0).tolist()}")
    print(f"true std per channel: {true_all.std(0).tolist()}")
    print("\nnear 0 moves ratio AND near-0 pred std per channel -> the model collapsed to predicting "
         "a near-constant delta regardless of input, and the kill-gate's FAIL means nothing about "
         "recurrence -- it means the model never learned to move at all. A moves ratio and pred std "
         "closer to the true numbers means the FAIL is a real, trustworthy null.")


if __name__ == "__main__":
    main()
