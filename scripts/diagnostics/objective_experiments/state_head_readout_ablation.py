"""Is the state head's own readout (pooling + additive z-combination) leaving signal on the table?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_readout_ablation.py

**What this tests, and why now.** F191 found `real_z` alone (64-D, from ITM on true transitions)
separates the four speed conditions at 80% / R2 0.78 -- well above what the actual trained state
head achieves downstream (F187: 68% win rate, low exact-pair accuracy). The state head's forward
pass is `head(pool(delta - offset) + z_proj(z))` (`wm/models/state_head.py`): a NAIVE MEAN-POOL over
patch tokens, an ADDITIVE combination with z, and a fixed frozen per-embodiment OFFSET subtracted
before any of it. Three design choices, each independently testable offline with a strong probe, no
retraining:

    1. does z ALONE already predict forward speed better than the current combined design implies
    2. does mean-pooling throw away spatial structure a different pooling operator would keep
    3. does the frozen offset subtraction help, hurt, or do nothing to FINE (not coarse) accuracy

Same data, same held-out-by-clip split, same kNN/regression probes as F190/F191, so every number
here reads directly against those. Predicts forward speed specifically (channel 0), the exact
channel F187 found the trained head confusing between adjacent conditions.

Diagnosis only; trains nothing, touches no checkpoint beyond reading `teacher_ego.pt` and the
already-fit offset in `wm/runs/beh12_state/best_state.pt`.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

SPEED_CONDITIONS = ["speed_c5.8", "speed_c7.1", "speed_c8.15", "speed_c8.8"]


def score(name, X_train, s_train, X_test, s_test):
    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    reg = KNeighborsRegressor(n_neighbors=15).fit(Xs_train, s_train)
    pred = reg.predict(Xs_test)
    r2 = 1 - np.sum((pred - s_test) ** 2) / max(np.sum((s_test - s_train.mean()) ** 2), 1e-9)
    rho, _ = spearmanr(pred, s_test)
    print(f"  {name:<32} R2 {r2:+.3f}   spearman rho {rho:+.3f}   ({X_train.shape[1]}-D)")
    return r2, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--offset_ckpt", default="wm/runs/beh12_state/best_state.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c10f10t10_more_ego_flat")
    ap.add_argument("--test_clips", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.teacher), map_location=device, weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    offset_ck = torch.load(os.path.join(ROOT, args.offset_ckpt), map_location=device,
                           weights_only=False)
    offset = offset_ck["state"]["offset_hexapod"].to(device)
    print(f"offset_hexapod norm: {offset.norm().item():.3f}\n")

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    cand = {c: [] for c in SPEED_CONDITIONS}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
        if c in cand:
            cand[c].append(p)

    rng = np.random.default_rng(args.seed)
    feats = {k: ([], [], [], []) for k in
            ("z", "mean_pool_raw", "mean_pool_offset", "std_pool", "z+mean_pool_offset",
             "z+std_pool")}
    # each: (X_train, s_train, X_test, s_test)

    for ci, cond in enumerate(SPEED_CONDITIONS):
        paths = cand[cond]
        order = rng.permutation(len(paths))
        test_idx = set(order[:args.test_clips].tolist())
        for i, p in enumerate(paths):
            clip = load(p, REGISTRY["hexapod"])
            with np.load(p, allow_pickle=True) as z:
                frames = z["frames"]
            with torch.no_grad():
                e = encoder.encode(list(frames)).float().to(device)   # (T, tokens, dim)
            n = len(e) - lag
            if n < 2:
                continue
            with torch.no_grad():
                e_t, e_next = e[:n], e[lag:lag + n]
                d = e_next - e_t                                     # (n, tokens, dim), per-token
                z_real = itm(e_t, e_next).cpu().numpy()
                mean_pool_raw = d.mean(1).cpu().numpy()
                mean_pool_offset = (d.mean(1) - offset).cpu().numpy()
                std_pool = d.std(1).cpu().numpy()                    # spatial VARIANCE across
                                                                       # tokens, discarded by mean
            fwd = np.asarray(clip["body_motion"])[:n, 0]
            bucket = "test" if i in test_idx else "train"

            def add(key, X):
                Xtr, str_, Xte, ste = feats[key]
                (Xtr if bucket == "train" else Xte).append(X)
                (str_ if bucket == "train" else ste).append(fwd)

            add("z", z_real)
            add("mean_pool_raw", mean_pool_raw)
            add("mean_pool_offset", mean_pool_offset)
            add("std_pool", std_pool)
            add("z+mean_pool_offset", np.concatenate([z_real, mean_pool_offset], axis=1))
            add("z+std_pool", np.concatenate([z_real, std_pool], axis=1))

    print("forward-speed regression, held-out by clip, kNN k=15 (matches F190/F191 exactly):\n")
    for key in ("z", "mean_pool_raw", "mean_pool_offset", "std_pool",
               "z+mean_pool_offset", "z+std_pool"):
        Xtr, str_, Xte, ste = feats[key]
        Xtr, str_ = np.concatenate(Xtr), np.concatenate(str_)
        Xte, ste = np.concatenate(Xte), np.concatenate(ste)
        score(key, Xtr, str_, Xte, ste)

    print("\nz vs mean_pool_*          -> is z alone already better than what pooled delta gives")
    print("mean_pool_offset vs _raw  -> does the frozen offset help, hurt, or do nothing (fine-grained)")
    print("std_pool vs mean_pool_*   -> does a variance-preserving pool beat plain averaging")
    print("z+X vs X alone / z alone  -> does combining help, or does the current design's delta")
    print("                             term drag z's already-good signal down")


if __name__ == "__main__":
    main()
