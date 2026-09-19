"""Does giving FTM an explicit ground-truth Froude signal, ALONGSIDE the full z (not replacing it,
unlike `froude_bottleneck_action_ceiling.py`'s ruled-out proposal), make its predicted direction
more sensitive to the real action -- the "action-lever" gap F173/F180 measured and found tiny
(+0.055 cosine at one step)?

**Why this doesn't cost new information, but might still help.** `froude_t = body_head(z_t)` is
already fully derivable from `z_t` -- concatenating it on top adds nothing FTM couldn't already
reach in principle. What it might change is optimization: FTM has to extract whatever's
Froude-relevant from a 64-D bottleneck shaped by five other losses at once (F215's own finding);
handing it an explicit, pre-extracted 3-D signal removes that extraction burden. This uses
GROUND-TRUTH `body_motion[t]`, not `body_head`'s own (noisy) prediction of it -- the best case for
the proposal, same convention as the bottleneck-ceiling check.

**Method.** Freeze the ITM/encoder from an existing, already-fit checkpoint (`--ckpt`, default the
clean hexapod pretrain). Freshly train TWO ForwardTransitionModel copies from scratch on the exact
same (x_t, action) data and next-embedding-MSE objective FTM is normally trained on -- one takes
`z_t` (64-D) as its condition, the other `concat(z_t, froude_t)` (67-D), otherwise identical
architecture and recipe. Then score both on HELD OUT data with the real-z-vs-mean-z action-lever
(F173/F180's own metric): does the FTM's predicted displacement, given the real code for this
transition, point closer to the TRUE displacement than it does given a code that's blind to which
action was taken (the held-out set's own mean code)?

**Reading.** If the augmented variant's held-out gap is meaningfully larger than the baseline's,
the extraction-burden story holds and this is worth carrying into a real retrain. If the gap is the
same or smaller, giving FTM the signal explicitly didn't help it use that signal any better --
consistent with F180's own finding that the wall here is FTM/state-head's basic action-sensitivity,
not the loss target.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/ftm_froude_conditioning_check.py
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


def collect(spec, data_dir, itm, channels, encoder, offset, device):
    x_ts, e_nexts, z_reals, froudes = [], [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz"))):
        clip = load(f, spec)
        with np.load(f, allow_pickle=True) as d:
            frames = d["frames"]
        e = encode_clip(encoder, frames, 4).float().to(device)
        if offset is not None:
            e = e - offset.to(device)
        motion = np.asarray(clip["body_motion"])[:, channels]
        n = min(len(e) - 1, len(motion))
        with torch.no_grad():
            z = itm(e[:n], e[1:n + 1])
        x_ts.append(e[:n].cpu())
        e_nexts.append(e[1:n + 1].cpu())
        z_reals.append(z.cpu())
        froudes.append(torch.tensor(motion[:n], dtype=torch.float32))
        print(f"  encoded {os.path.basename(f)} n={n}")
    return (torch.cat(x_ts), torch.cat(e_nexts), torch.cat(z_reals), torch.cat(froudes))


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
    channels = [int(c) for c in cfg.body_channels]
    offset = offset_for(ck, "hexapod")
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    print("--- collecting train ---")
    x_tr, en_tr, z_tr, fr_tr = collect(HEXAPOD, TRAIN_DIR, itm, channels, encoder, offset, device)
    print("--- collecting held out ---")
    x_he, en_he, z_he, fr_he = collect(HEXAPOD, HELDOUT_DIR, itm, channels, encoder, offset, device)

    code_tr_base, code_he_base = z_tr, z_he
    code_tr_aug = torch.cat([z_tr, fr_tr], dim=-1)
    code_he_aug = torch.cat([z_he, fr_he], dim=-1)
    cfg_aug = dataclasses.replace(cfg, z_dim=cfg.z_dim + len(channels))

    print(f"\n--- fitting baseline FTM (z only, {cfg.z_dim}-D) ---")
    ftm_base, loss_base = fit_ftm(cfg, x_tr, code_tr_base, en_tr, device)
    print(f"  final train loss (MSE) {loss_base:.6f}")

    print(f"--- fitting augmented FTM (z + ground-truth Froude, {cfg_aug.z_dim}-D) ---")
    ftm_aug, loss_aug = fit_ftm(cfg_aug, x_tr, code_tr_aug, en_tr, device)
    print(f"  final train loss (MSE) {loss_aug:.6f}")

    print("\n=== action-lever gap, held out ===")
    real_b, mean_b = action_lever_gap(ftm_base, x_he, code_he_base, en_he, device)
    real_a, mean_a = action_lever_gap(ftm_aug, x_he, code_he_aug, en_he, device)
    print(f"  baseline (z only):          real {real_b:.4f}  mean {mean_b:.4f}  "
         f"gap {real_b - mean_b:+.4f}")
    print(f"  augmented (z + froude):     real {real_a:.4f}  mean {mean_a:.4f}  "
         f"gap {real_a - mean_a:+.4f}")
    print(f"\n  augmented gap - baseline gap = {(real_a - mean_a) - (real_b - mean_b):+.4f}")
    print("  positive and clearly bigger than the baseline's own gap -> the extraction-burden")
    print("  story holds, worth a real retrain. Same or smaller -> explicit Froude didn't help")
    print("  FTM use the signal any better, consistent with F180's action-sensitivity wall.")


if __name__ == "__main__":
    main()
