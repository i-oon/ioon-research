"""Does giving FTM a short ACTION history (z_{t-1} alongside z_t), rather than a FRAME history,
move the action-lever gap (F173/F180's real-z-vs-mean-z metric, found tiny: +0.055 cosine)?

**Why this is a different test from the one already run and killed.** The addendum in F180's
architecture arc fed the frozen FTM two consecutive FRAMES (e_{t-1}, e_t) and found no change
(hexapod -0.051 -> -0.056, B1 +0.042 -> +0.044). That tested visual history. Rereading Egocentric
VSM's actual `ResNet_RNN.forward` shows its own real temporal signal is NOT visual: the "5-frame"
LSTM input is five COPIES of the SAME single frame's feature (`torch.cat(5*[x.unsqueeze(0)])`,
`ResNet_RNN.py:122`) -- no real image history at all. The genuine history lives on the ACTION side:
`input_pre_a=True` concatenates the current action with the PREVIOUS one (24-D vs 12-D,
`ResNet_RNN.py:95-97`) into a plain MLP. This script tests that axis instead: z_{t-1} concatenated
with z_t, not e_{t-1} concatenated with e_t.

**Method**, mirroring `ftm_froude_conditioning_check.py` exactly (same data, same recipe, same
lever): freeze the ITM/encoder from an existing checkpoint. Train two fresh ForwardTransitionModel
copies from scratch on the same (x_t, action) data and next-embedding-MSE objective -- one
conditioned on z_t (64-D, baseline), the other on concat(z_{t-1}, z_t) (128-D, history). Score both
on held-out data with the real-z-vs-mean-z action-lever.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/ftm_action_history_lever_check.py
"""
import argparse
import dataclasses
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import HEXAPOD, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

TRAIN_DIR = "data/egocentric/beh12_c10f10t10_ego_flat_cleantrain"
HELDOUT_DIR = "data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout"
EPOCHS = 100
LR = 3e-4
BATCH = 64


def collect(spec, data_dir, itm, encoder, offset, device):
    """Per clip: x_t, e_{t+1}, z_t = ITM(e_t,e_{t+1}), and z_{t-1} = ITM(e_{t-1},e_t) (zero
    vector for the clip's first transition, which has no predecessor)."""
    x_ts, e_nexts, z_nows, z_prevs = [], [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz"))):
        clip = load(f, spec)
        with np.load(f, allow_pickle=True) as d:
            frames = d["frames"]
        e = encode_clip(encoder, frames, 4).float().to(device)
        if offset is not None:
            e = e - offset.to(device)
        n = len(e) - 1
        with torch.no_grad():
            z_now = itm(e[:n], e[1:n + 1])          # z_t for t=0..n-1
            z_prev_valid = itm(e[:n - 1], e[1:n])    # z_{t-1} for t=1..n-1
        z_prev = torch.cat([torch.zeros_like(z_now[:1]), z_prev_valid], dim=0)
        x_ts.append(e[:n].cpu())
        e_nexts.append(e[1:n + 1].cpu())
        z_nows.append(z_now.cpu())
        z_prevs.append(z_prev.cpu())
        print(f"  encoded {os.path.basename(f)} n={n}")
    return (torch.cat(x_ts), torch.cat(e_nexts), torch.cat(z_nows), torch.cat(z_prevs))


def fit_ftm(cfg, x_t, code, e_next, device, epochs=EPOCHS, lr=LR, batch=BATCH):
    ftm = ForwardTransitionModel(cfg).to(device)
    opt = torch.optim.Adam(ftm.parameters(), lr=lr, weight_decay=1e-4)
    n = len(x_t)
    last = None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            pred = ftm(x_t[idx].to(device), code[idx].to(device))
            loss = F.mse_loss(pred, e_next[idx].to(device))
            loss.backward(); opt.step()
            last = float(loss.item())
    ftm.eval()
    return ftm, last


def action_lever_gap(ftm, x_t, code, e_next, device, batch=BATCH):
    """F173/F180's metric: median cosine(pred_disp, true_disp) with the REAL code for this
    transition, minus the same with the held-out set's own MEAN code (action-blind)."""
    mean_code = code.mean(0, keepdim=True)
    real_cos, mean_cos = [], []
    with torch.no_grad():
        for start in range(0, len(x_t), batch):
            sl = slice(start, start + batch)
            xt = x_t[sl].to(device)
            true_disp = F.normalize((e_next[sl].to(device) - xt).flatten(1), dim=1)
            for tag, cd in (("real", code[sl].to(device)),
                           ("mean", mean_code.expand(xt.shape[0], -1).to(device))):
                pred_disp = F.normalize((ftm(xt, cd) - xt).flatten(1), dim=1)
                cos = F.cosine_similarity(pred_disp, true_disp, dim=1).cpu().numpy()
                (real_cos if tag == "real" else mean_cos).extend(cos.tolist())
    return float(np.median(real_cos)), float(np.median(mean_cos))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hinge_cleansplit/best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).eval().to(device)
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    offset = offset_for(ck, "hexapod")
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    print("--- collecting train ---")
    x_tr, en_tr, znow_tr, zprev_tr = collect(HEXAPOD, TRAIN_DIR, itm, encoder, offset, device)
    print("--- collecting held out ---")
    x_he, en_he, znow_he, zprev_he = collect(HEXAPOD, HELDOUT_DIR, itm, encoder, offset, device)

    code_tr_base, code_he_base = znow_tr, znow_he
    code_tr_hist = torch.cat([zprev_tr, znow_tr], dim=-1)
    code_he_hist = torch.cat([zprev_he, znow_he], dim=-1)
    cfg_hist = dataclasses.replace(cfg, z_dim=cfg.z_dim * 2)

    print(f"\n--- fitting baseline FTM (z_t only, {cfg.z_dim}-D) ---")
    ftm_base, loss_base = fit_ftm(cfg, x_tr, code_tr_base, en_tr, device)
    print(f"  final train loss (MSE) {loss_base:.6f}")

    print(f"--- fitting action-history FTM (z_t-1 + z_t, {cfg_hist.z_dim}-D) ---")
    ftm_hist, loss_hist = fit_ftm(cfg_hist, x_tr, code_tr_hist, en_tr, device)
    print(f"  final train loss (MSE) {loss_hist:.6f}")

    print("\n=== action-lever gap, held out ===")
    real_b, mean_b = action_lever_gap(ftm_base, x_he, code_he_base, en_he, device)
    real_h, mean_h = action_lever_gap(ftm_hist, x_he, code_he_hist, en_he, device)
    print(f"  baseline (z_t only):        real {real_b:.4f}  mean {mean_b:.4f}  "
         f"gap {real_b - mean_b:+.4f}")
    print(f"  action-history (z_t-1,z_t): real {real_h:.4f}  mean {mean_h:.4f}  "
         f"gap {real_h - mean_h:+.4f}")
    print(f"\n  history gap - baseline gap = {(real_h - mean_h) - (real_b - mean_b):+.4f}")
    print("  positive and clearly bigger than baseline's own gap -> action-history is a real,")
    print("  previously-untested lever (frame-history was tested and killed; this is different).")
    print("  same or smaller -> action-history doesn't help FTM's action-sensitivity either.")


if __name__ == "__main__":
    main()
