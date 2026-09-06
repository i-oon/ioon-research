"""Is fine speed-magnitude even IN the raw V-JEPA2 latent, independent of the FTM/predictor?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/embedding_speed_ceiling.py

**What this isolates.** F187 found the state head confuses adjacent speeds (speed_c7.1 vs c8.8,
etc.) on well-separated candidates. F188 found two architecture fixes to the FTM/predictor both
failed to help. Both of those measure the PIPELINE (encoder -> ITM/FTM -> head). This measures the
ENCODER ALONE: raw, frozen V-JEPA2 embeddings, no ITM, no FTM, no trained head of any kind except
the offline probe fit here. If a strong probe cannot separate the four physically distinct speeds
(forward 0.124/0.152/0.186/0.192, verified real and monotonic by --separability) from the raw
embedding delta, the information is not there to begin with and no predictor fix reaches it. If a
strong probe CAN separate them, the ceiling is not the encoder -- it is something downstream not
using what is already present.

**The quantity probed is `e_t+1 - e_t`, raw and frozen, pooled over patch tokens** -- not a single
static frame. A still image does not obviously encode instantaneous speed; the frame-to-frame change
does, and this is the same delta `action_lever_vs_horizon.py` and the state head both read, just
before any trained model touches it.

**Three probes, not one, and the point is redundancy.** A single weak probe reading null is
ambiguous -- weak probe or absent signal, indistinguishable. kNN (k=5, k=15) on raw standardised
features, kNN on a 50-component PCA (curse-of-dimensionality control), and a small MLP classifier
(finds nonlinear structure neither kNN variant assumes) are fit and read together: if ALL THREE read
near chance, that is a real null. If any one clears it clearly, the information is present.

**Split by clip, not by frame** -- F76's leak (consecutive frames of one clip are near-duplicates)
applies here exactly as it does everywhere else in this project.

Classification: 4-way, chance 25%. Regression: predicts each clip's own achieved forward speed
(from `body_motion`, not the nominal `--cycles` label), read by held-out R2 and Spearman rank
correlation -- the more honest test, since it asks whether FINE, CONTINUOUS magnitude is recoverable,
not just whether four discrete bins are.

Diagnosis only; trains nothing beyond the offline probes themselves, and touches no checkpoint.
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
from wm.data.embodiment import REGISTRY, load  # noqa: E402

SPEED_CONDITIONS = ["speed_c5.8", "speed_c7.1", "speed_c8.15", "speed_c8.8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/egocentric/beh12_c10f10t10_more_ego_flat")
    ap.add_argument("--test_clips", type=int, default=5, help="held-out clips per condition")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    # ---- gather clips per condition, split by clip ---------------------------------------------
    cand = {c: [] for c in SPEED_CONDITIONS}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
        if c in cand:
            cand[c].append(p)
    for c, paths in cand.items():
        print(f"  {c}: {len(paths)} clips")

    rng = np.random.default_rng(args.seed)
    X_train, y_train, s_train = [], [], []   # features, condition label, achieved forward speed
    X_test, y_test, s_test = [], [], []

    for ci, cond in enumerate(SPEED_CONDITIONS):
        paths = cand[cond]
        order = rng.permutation(len(paths))
        test_idx = set(order[:args.test_clips].tolist())
        for i, p in enumerate(paths):
            clip = load(p, REGISTRY["hexapod"])
            with np.load(p, allow_pickle=True) as z:
                frames = z["frames"]
            e = encoder.encode(list(frames)).float()          # (T, tokens, dim), frozen, raw
            pooled = e.mean(1).cpu().numpy()                  # (T, dim)
            delta = pooled[1:] - pooled[:-1]                  # (T-1, dim), raw e_t+1 - e_t
            fwd = np.asarray(clip["body_motion"])[:len(delta), 0]  # per-frame forward speed, already
                                                                    # Froude-scaled and world-frame-corrected
            (X_test if i in test_idx else X_train).append(delta)
            (y_test if i in test_idx else y_train).append(np.full(len(delta), ci))
            (s_test if i in test_idx else s_train).append(fwd)

    X_train, y_train, s_train = np.concatenate(X_train), np.concatenate(y_train), np.concatenate(s_train)
    X_test, y_test, s_test = np.concatenate(X_test), np.concatenate(y_test), np.concatenate(s_test)
    print(f"\ntrain: {len(X_train)} transitions, test: {len(X_test)} transitions "
          f"(held out {args.test_clips} clips/condition)\n")

    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

    print(f"{'probe':<28}{'train acc':>11}{'test acc':>11}{'chance':>9}")
    results = {}
    for k in (5, 15):
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xs_train, y_train)
        tr, te = clf.score(Xs_train, y_train), clf.score(Xs_test, y_test)
        results[f"kNN k={k}, raw {Xs_train.shape[1]}-D"] = te
        print(f"{'kNN k=' + str(k) + ', raw':<28}{tr:>11.1%}{te:>11.1%}{0.25:>9.1%}")

    pca = PCA(n_components=50, random_state=args.seed).fit(Xs_train)
    Xp_train, Xp_test = pca.transform(Xs_train), pca.transform(Xs_test)
    print(f"  (PCA-50 explains {pca.explained_variance_ratio_.sum():.1%} of variance)")
    for k in (5, 15):
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xp_train, y_train)
        tr, te = clf.score(Xp_train, y_train), clf.score(Xp_test, y_test)
        results[f"kNN k={k}, PCA-50"] = te
        print(f"{'kNN k=' + str(k) + ', PCA-50':<28}{tr:>11.1%}{te:>11.1%}{0.25:>9.1%}")

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=2000, random_state=args.seed,
                        early_stopping=True).fit(Xs_train, y_train)
    tr, te = mlp.score(Xs_train, y_train), mlp.score(Xs_test, y_test)
    results["MLP classifier"] = te
    print(f"{'MLP classifier':<28}{tr:>11.1%}{te:>11.1%}{0.25:>9.1%}")

    print("\nconfusion matrix (test), best probe by test accuracy:")
    best_name = max(results, key=results.get)
    print(f"  best: {best_name} ({results[best_name]:.1%})")
    if "MLP" in best_name:
        pred = mlp.predict(Xs_test)
    elif "PCA" in best_name:
        k = int(best_name.split("k=")[1].split(",")[0])
        pred = KNeighborsClassifier(n_neighbors=k).fit(Xp_train, y_train).predict(Xp_test)
    else:
        k = int(best_name.split("k=")[1].split(",")[0])
        pred = KNeighborsClassifier(n_neighbors=k).fit(Xs_train, y_train).predict(Xs_test)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_test.astype(int), pred.astype(int)):
        cm[t, p] += 1
    header = "true vs pred"
    print(f"  {header:<14}" + "".join(f"{c:>12}" for c in SPEED_CONDITIONS))
    for i, cond in enumerate(SPEED_CONDITIONS):
        print(f"  {cond:<14}" + "".join(f"{cm[i,j]:>12}" for j in range(4)))

    # ---- regression: continuous magnitude, not just 4 bins --------------------------------------
    print("\ncontinuous forward-speed regression (held-out, per transition):")
    for name, reg in (("kNN k=15 regressor", KNeighborsRegressor(n_neighbors=15)),
                      ("MLP regressor", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=2000,
                                                     random_state=args.seed, early_stopping=True))):
        reg.fit(Xs_train, s_train)
        pred = reg.predict(Xs_test)
        r2 = 1 - np.sum((pred - s_test) ** 2) / max(np.sum((s_test - s_train.mean()) ** 2), 1e-9)
        rho, _ = spearmanr(pred, s_test)
        print(f"  {name:<22} R2 {r2:+.3f}   spearman rho {rho:+.3f}")

    print("\nif every probe reads near chance (25% / R2~0 / rho~0): the information is not in the")
    print("raw latent and no predictor fix reaches it -- coarse-only is the honest scope.")
    print("if any probe clears chance clearly: the ceiling is not the encoder.")


if __name__ == "__main__":
    main()
