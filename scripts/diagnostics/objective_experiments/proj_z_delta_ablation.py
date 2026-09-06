"""Does combining delta into the readout hurt on proj(action)-z too, not just the oracle ITM-z?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/proj_z_delta_ablation.py

**Why this is the test that actually decides the fix, not F192 alone.** F192 showed z + delta
underperforms z alone -- but that z came from `ITM(e_t, e_t+1)`, the true observed transition, an
oracle unavailable at planning time. Ranking uses `proj(action)` instead (F191's "projected z").
proj_z alone already reads R2 0.791 (F191) -- close to real_z's 0.781. What was never checked is
whether COMBINING proj_z with delta hurts the same way combining real_z with delta did. If it does,
the state-head fix (drop delta, keep z) is confirmed for the path that is actually deployed. If
proj_z + delta does NOT show the same drop, the F192 finding may be specific to the oracle z and
would not justify the architecture change on its own.

Same data, same held-out-by-clip split, same kNN probe as F190/F191/F192.

Diagnosis only; trains nothing.
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
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

SPEED_CONDITIONS = ["speed_c5.8", "speed_c7.1", "speed_c8.15", "speed_c8.8"]


def score(name, X_train, s_train, X_test, s_test):
    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    reg = KNeighborsRegressor(n_neighbors=15).fit(Xs_train, s_train)
    pred = reg.predict(Xs_test)
    r2 = 1 - np.sum((pred - s_test) ** 2) / max(np.sum((s_test - s_train.mean()) ** 2), 1e-9)
    rho, _ = spearmanr(pred, s_test)
    print(f"  {name:<28} R2 {r2:+.3f}   spearman rho {rho:+.3f}   ({X_train.shape[1]}-D)")


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
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
    proj.load_state_dict(ck["projector"])
    for m in (itm, proj):
        for p in m.parameters():
            p.requires_grad_(False)
    offset_ck = torch.load(os.path.join(ROOT, args.offset_ckpt), map_location=device,
                           weights_only=False)
    offset = offset_ck["state"]["offset_hexapod"].to(device)

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    cand = {c: [] for c in SPEED_CONDITIONS}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
        if c in cand:
            cand[c].append(p)

    rng = np.random.default_rng(args.seed)
    feats = {k: ([], [], [], []) for k in ("proj_z", "mean_pool_offset", "proj_z+mean_pool")}

    for ci, cond in enumerate(SPEED_CONDITIONS):
        paths = cand[cond]
        order = rng.permutation(len(paths))
        test_idx = set(order[:args.test_clips].tolist())
        for i, p in enumerate(paths):
            clip = load(p, REGISTRY["hexapod"])
            with np.load(p, allow_pickle=True) as z:
                frames = z["frames"]
            with torch.no_grad():
                e = encoder.encode(list(frames)).float().to(device)
            n = len(e) - lag
            if n < 2:
                continue
            with torch.no_grad():
                e_t, e_next = e[:n], e[lag:lag + n]
                d = e_next - e_t
                mean_pool_offset = (d.mean(1) - offset).cpu().numpy()
                actions = torch.as_tensor(clip["actions"][lag:lag + n], dtype=torch.float32,
                                          device=device)
                pz = proj(actions, "hexapod").cpu().numpy()
            fwd = np.asarray(clip["body_motion"])[:n, 0]
            bucket = "test" if i in test_idx else "train"

            def add(key, X):
                Xtr, str_, Xte, ste = feats[key]
                (Xtr if bucket == "train" else Xte).append(X)
                (str_ if bucket == "train" else ste).append(fwd)

            add("proj_z", pz)
            add("mean_pool_offset", mean_pool_offset)
            add("proj_z+mean_pool", np.concatenate([pz, mean_pool_offset], axis=1))

    print("forward-speed regression, held-out by clip, kNN k=15 -- deployment-relevant z:\n")
    for key in ("proj_z", "mean_pool_offset", "proj_z+mean_pool"):
        Xtr, str_, Xte, ste = feats[key]
        Xtr, str_ = np.concatenate(Xtr), np.concatenate(str_)
        Xte, ste = np.concatenate(Xte), np.concatenate(ste)
        score(key, Xtr, str_, Xte, ste)

    print("\nproj_z alone vs proj_z+mean_pool -> does combining hurt on the DEPLOYED z too,")
    print("                                      or was F192's drop specific to the oracle ITM-z")


if __name__ == "__main__":
    main()
