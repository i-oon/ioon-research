"""Where does the fine-speed signal degrade, if it does -- raw latent, real z, or the projector?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/pipeline_speed_ceiling.py

**F190 found the raw frozen embedding delta carries the fine speed-magnitude signal clearly** (kNN
~50% on 4-way, chance 25%; R2 0.40). That rules out the encoder as a hard ceiling. It does not rule
out the signal being lost one or two stages later, before a scorer ever sees it. Two links, tested
here with the exact same probes as F190 so the numbers read side by side:

    real z       = ITM(e_t, e_t+1), from TRUE observed transitions. Tests whether the 64-D
                   bottleneck itself discards fine magnitude that the raw delta still has.
    projected z  = proj(action), from the RECORDED ACTION alone, no next frame. This is what
                   ranking actually uses at test time (the next frame is the thing being decided),
                   so this is the one link every prior check in this chain has skipped.

**The read.** If `real z` matches F190's raw-delta numbers, the ITM is not the bottleneck. If
`projected z` matches `real z`, the projector is not either -- and the residual gap in F186/F187
is not explained by any of these three, pointing back to the state head's own readout or the
95-97% oracle ceiling already established. If `projected z` reads clearly WORSE than `real z`, the
projector is where fine magnitude dies, independent of how much training data the state head gets.

Same data, same held-out-by-clip split, same three probes (kNN raw, kNN PCA-50, MLP) as F190, so
the comparison is apples to apples.

Diagnosis only; trains nothing, touches no checkpoint file.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
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


def probe_report(name, X_train, y_train, s_train, X_test, y_test, s_test, seed):
    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    print(f"\n--- {name} ({X_train.shape[1]}-D) ---")
    best = 0.0
    for k in (5, 15):
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xs_train, y_train)
        te = clf.score(Xs_test, y_test)
        best = max(best, te)
        print(f"  kNN k={k}, raw          test acc {te:.1%}  (chance 25%)")
    if X_train.shape[1] > 50:
        pca = PCA(n_components=50, random_state=seed).fit(Xs_train)
        Xp_train, Xp_test = pca.transform(Xs_train), pca.transform(Xs_test)
        for k in (5, 15):
            clf = KNeighborsClassifier(n_neighbors=k).fit(Xp_train, y_train)
            te = clf.score(Xp_test, y_test)
            best = max(best, te)
            print(f"  kNN k={k}, PCA-50       test acc {te:.1%}")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=seed,
                        early_stopping=True).fit(Xs_train, y_train)
    te = mlp.score(Xs_test, y_test)
    best = max(best, te)
    print(f"  MLP classifier            test acc {te:.1%}")

    reg = KNeighborsRegressor(n_neighbors=15).fit(Xs_train, s_train)
    pred = reg.predict(Xs_test)
    r2 = 1 - np.sum((pred - s_test) ** 2) / max(np.sum((s_test - s_train.mean()) ** 2), 1e-9)
    rho, _ = spearmanr(pred, s_test)
    print(f"  kNN k=15 regressor        R2 {r2:+.3f}   spearman rho {rho:+.3f}")
    return best, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_ego/teacher_ego.pt")
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
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    lag = max(1, cfg.action_lag)

    cand = {c: [] for c in SPEED_CONDITIONS}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
        if c in cand:
            cand[c].append(p)
    for c, paths in cand.items():
        print(f"  {c}: {len(paths)} clips")

    rng = np.random.default_rng(args.seed)
    sets = {"raw_delta": ([], [], [], [], [], []), "real_z": ([], [], [], [], [], []),
           "proj_z": ([], [], [], [], [], [])}
    # each tuple: (X_train, y_train, s_train, X_test, y_test, s_test)

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
                raw = (e_next.mean(1) - e_t.mean(1)).cpu().numpy()
                rz = itm(e_t, e_next).cpu().numpy()
                actions = torch.as_tensor(clip["actions"][lag:lag + n], dtype=torch.float32,
                                          device=device)
                pz = proj(actions, "hexapod").cpu().numpy()
            fwd = np.asarray(clip["body_motion"])[:n, 0]

            bucket = "test" if i in test_idx else "train"
            for name, feat in (("raw_delta", raw), ("real_z", rz), ("proj_z", pz)):
                Xtr, ytr, str_, Xte, yte, ste = sets[name]
                (Xtr if bucket == "train" else Xte).append(feat)
                (ytr if bucket == "train" else yte).append(np.full(len(feat), ci))
                (str_ if bucket == "train" else ste).append(fwd)

    print(f"\nteacher: {args.teacher}   action_lag: {lag}")
    for name in ("raw_delta", "real_z", "proj_z"):
        Xtr, ytr, str_, Xte, yte, ste = sets[name]
        Xtr, ytr, str_ = np.concatenate(Xtr), np.concatenate(ytr), np.concatenate(str_)
        Xte, yte, ste = np.concatenate(Xte), np.concatenate(yte), np.concatenate(ste)
        probe_report(name, Xtr, ytr, str_, Xte, yte, ste, args.seed)

    print("\nreal_z ~ raw_delta -> the ITM's 64-D bottleneck is not where signal is lost")
    print("proj_z ~ real_z    -> the projector is not where signal is lost either")
    print("proj_z << real_z   -> the projector is the bottleneck: ranking uses proj(action), which")
    print("                      is exactly the link every prior check in this chain has skipped")


if __name__ == "__main__":
    main()
