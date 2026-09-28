"""Does the margin fine-tune's z-separation gain generalize to held-out clips, or is it a
training-set shortcut?

`margin_finetune_itm.py` pushed the within/across ceiling ratio from 0.983 to 0.276 on the SAME
clips it was fine-tuned on -- side_L/side_R even beat the ground-truth Froude ceiling (0.183/0.154),
which is a red flag: z cannot legitimately separate conditions better than the real physics does,
so that specific result is suspicious of a shortcut (camera seed, room texture, timing) rather than
recovered Froude signal.

This fits a fresh ridge probe z -> true Froude on TRAIN clips only, and scores it on HELD-OUT clips
the fine-tune never saw, for both the baseline and margin checkpoints. If the margin checkpoint's
held-out R^2/correlation is genuinely higher, the gain is real. If it is flat or worse despite the
training-set ceiling ratio improving, the margin loss overfit a shortcut.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/margin_heldout_check.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/best.pt \\
        --train_data data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \\
        --heldout_data data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

CH = ["forward", "lateral", "yaw"]


def gather(ckpt_path, data_dir, embodiment, encoder):
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    reg = REGISTRY[embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz")))
    Z, Y = [], []
    for p in paths:
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        with torch.no_grad():
            z = itm(e[:-1], e[1:])
        Z.append(z.cpu().numpy())
        Y.append(np.asarray(clip["body_motion"], dtype=np.float64)[: len(z)])
    return np.concatenate(Z), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", default="wm/runs/beh12_hinge_cleansplit/best.pt")
    ap.add_argument("--margin_ckpt", default="wm/runs/beh12_margin_itm/best.pt")
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--heldout_data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    args = ap.parse_args()

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    for tag, ckpt in (("baseline", args.baseline_ckpt), ("margin fine-tune", args.margin_ckpt)):
        Z_tr, Y_tr = gather(ckpt, args.train_data, args.embodiment, encoder)
        Z_te, Y_te = gather(ckpt, args.heldout_data, args.embodiment, encoder)
        mu, sd = Y_tr.mean(0), Y_tr.std(0) + 1e-9
        Y_tr_s, Y_te_s = (Y_tr - mu) / sd, (Y_te - mu) / sd
        model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Z_tr, Y_tr_s)
        pred = model.predict(Z_te)
        print(f"\n=== {tag} ({ckpt}) ===")
        print(f"{'channel':>10}  {'R2':>8}  {'pearson r':>10}")
        for j, ch in enumerate(CH):
            r2 = r2_score(Y_te_s[:, j], pred[:, j])
            r = 0.0 if pred[:, j].std() < 1e-9 else float(np.corrcoef(pred[:, j], Y_te_s[:, j])[0, 1])
            print(f"{ch:>10}  {r2:>8.3f}  {r:>10.3f}")
        overall_r2 = r2_score(Y_te_s, pred)
        print(f"{'overall':>10}  {overall_r2:>8.3f}")
    del encoder


if __name__ == "__main__":
    main()
