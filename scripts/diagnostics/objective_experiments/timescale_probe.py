"""Which side of the timescale mismatch is wrong: the latent's (single step) or the target's
(Froude smoothed over a ~1 s stride)?

2x2, no training, held-out by clip: RidgeCV from z to a body-motion target, scored on
`_cleanheldout`, for
    z       single-step `ITM(e_t, e_t+1)`   vs   z moving-averaged over a W-step window
    target  smoothed Froude (`clip["body_motion"]`, what selection is graded on)
            vs   per-step Froude (same body-frame velocity + yaw rate, no stride smoothing)

Reading: if averaging z over a stride lifts R2 on the smoothed target a lot, the latent's
timescale is the problem (a stride-level pretrain is justified). If single-step z predicts the
per-step target much better than the smoothed one, z is fine at its own timescale and the
averaged target is the mismatch.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/timescale_probe.py
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
from wm.data.embodiment import G, HEXAPOD_DT, REGISTRY, forward_axis, heading, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def per_step_froude(path, embodiment):
    """Same quantities as `body_velocity` + `yaw_rate`, minus the stride-window smoothing."""
    with np.load(path, allow_pickle=True) as d:
        pos = d["head"].astype(np.float64)
        quat = d["body_quat"]
        dt = float(d["dt"]) if "dt" in d.files else HEXAPOD_DT
    height = float(np.median(pos[:, 2]))
    scale = np.sqrt(G * max(height, 1e-6))
    v = np.gradient(pos[:, :2], dt, axis=0)
    f = forward_axis(quat, embodiment)
    left = np.stack([-f[:, 1], f[:, 0]], axis=1)
    fwd = (v * f).sum(1) / scale
    lat = (v * left).sum(1) / scale
    omega = np.gradient(np.unwrap(heading(quat, embodiment)), dt) * np.sqrt(max(height, 1e-6) / G)
    return np.stack([fwd, lat, omega], axis=1)


def moving_average(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], axis=1)


def gather(paths, itm, encoder, reg, device):
    out = []
    for p in paths:
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        with torch.no_grad():
            z = itm(e[:-1], e[1:]).cpu().numpy()
        n = len(z)
        smooth = np.asarray(clip["body_motion"], dtype=np.float64)[:n]
        raw = per_step_froude(p, "hexapod")[:n]
        out.append((z, smooth, raw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh24_hinge_cleansplit/best.pt")
    ap.add_argument("--train", default="data/egocentric/beh24_c10f10t10_ego_flat_cleantrain")
    ap.add_argument("--heldout", default="data/egocentric/beh24_c10f10t10_ego_flat_cleanheldout")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 5, 11, 21])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    reg = REGISTRY["hexapod"]
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    tr = gather(sorted(glob.glob(os.path.join(ROOT, args.train, "*.npz"))), itm, encoder, reg, device)
    he = gather(sorted(glob.glob(os.path.join(ROOT, args.heldout, "*.npz"))), itm, encoder, reg, device)
    del encoder

    print(f"{len(tr)} train clips, {len(he)} held-out clips\n")
    print(f"{'z window':>9} | {'R2 vs smoothed Froude':>22} | {'R2 vs per-step Froude':>22}")
    for w in args.windows:
        row = []
        for ti in (1, 2):  # 1 = smoothed target, 2 = per-step target
            Ztr = np.concatenate([moving_average(c[0], w) for c in tr])
            Ytr = np.concatenate([c[ti] for c in tr])
            Zte = np.concatenate([moving_average(c[0], w) for c in he])
            Yte = np.concatenate([c[ti] for c in he])
            mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Ztr, (Ytr - mu) / sd)
            row.append(r2_score((Yte - mu) / sd, m.predict(Zte)))
        print(f"{w:>9} | {row[0]:>22.3f} | {row[1]:>22.3f}")


if __name__ == "__main__":
    main()
