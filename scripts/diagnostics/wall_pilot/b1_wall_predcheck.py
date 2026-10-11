"""Wall PREDICTION CHECK (protocol locked 2026-10-09): from held-out start states near a wall, does the clearance read from
the FTM's imagined embeddings rank the 24 candidate commands like their TRUE future clearance?

Data `data/wall_predcheck/b1/{fit,test}` (`b1_wall_predcheck_data.py`): per start state 24 futures, each command HELD
CONSTANT 2 s (constant-command test, not the closed loop); fit / test differ in start state, episode and room look.
Imagination exactly as the planner's rollout: z_k = projector(the command chunk the future executed, stride 5), e^_k =
FTM(e^_{k-1}, z_k) from the real start embedding e_0; horizons k = 1 / 2 / 4 (0.25 / 0.5 / 1 s) and 8 (2 s, beyond every
trained chain -- reported separately).
Readouts (ridge on mean-pooled embeddings -> footprint signed clearance), both fitted on the FIT set only:
  R_real   fitted on REAL frames (all fit futures, every 5th frame) and applied unchanged to imagined embeddings:
           does the existing pipeline's output carry clearance in the form real frames do?
  R_imag   fitted on the FIT set's IMAGINED embeddings at the same horizon: is the information decodable at all?
References: real future (R_real on the true frame at 5k: the ceiling of reading), copy (R_real on e_0: identical for all
24 commands, so it cannot rank -- its error is the distance-only baseline).
Metrics on TEST, per horizon: within-state Spearman across the 24 commands (pred vs true clearance at frame 5k), MAE (m),
unsafe recall / precision (unsafe = true min clearance over [0, 5k] < 0.15 m; predicted unsafe = predicted < 0.15 m).

    .venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_predcheck.py --ckpt NAME=results/eval/NAME/ckpt/b1.pt [...]
    -> results/check/wall_predcheck/predcheck_<NAME>.txt
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
DATA = os.path.join(ROOT, "data/wall_predcheck/b1")
CHK = os.path.join(ROOT, "results/check/wall_predcheck")
HZ = (1, 2, 4, 8)
UNSAFE = 0.15


def spearman(a, b):
    """rank correlation with ties averaged (scipy); NaN when either side is constant (cannot rank)"""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) < 1e-9 or np.ptp(b) < 1e-9:
        return np.nan
    return float(spearmanr(a, b).correlation)


@torch.no_grad()
def collect(m, enc, split, K):
    from wm.data.embodiment import REGISTRY, load
    from wm.policy.planner import action_chunk_at
    from wm.evaluate import encode_clip
    groups = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(DATA, split, "*.npz"))):
        groups[os.path.basename(p).rsplit("_c", 1)[0]].append(p)
    out = []                                     # per state: dict(true[k][24], imag[k][24, D], real[k][24, D], e0 [D])
    for g, paths in sorted(groups.items()):
        rec = dict(true={k: [] for k in HZ}, tmin={k: [] for k in HZ}, imag={k: [] for k in HZ}, real={k: [] for k in HZ})
        e0 = None
        for p in sorted(paths):
            c = load(p, REGISTRY["b1"], lazy_frames=True)
            d = np.load(p, allow_pickle=True)
            idx = [0] + [k * K for k in HZ]
            E = encode_clip(enc, np.asarray(c["frames"])[idx], 8).float().cuda()
            if e0 is None:
                e0 = E[0:1]
            e = E[0:1]
            for k in range(1, max(HZ) + 1):
                ch = action_chunk_at(c["actions"], (k - 1) * K + m.action_lag, K)
                z = m.proj(torch.as_tensor(ch[None], device="cuda"), "b1")
                e = m.ftm_step(e, z)
                if k in HZ:
                    j = HZ.index(k) + 1
                    rec["imag"][k].append(e.float().mean(1).cpu().numpy()[0])
                    rec["real"][k].append(E[j].float().mean(0).cpu().numpy())
                    rec["true"][k].append(float(d["clearance"][k * K]))
                    rec["tmin"][k].append(float(d["clearance"][:k * K + 1].min()))
            rec["c0"] = float(d["clearance"][0])
        rec["e0"] = e0.float().mean(1).cpu().numpy()[0]
        rec["state"] = g
        out.append(rec)
    return out


def realfit_data(m, enc, K):
    """real frames of the FIT futures (every 5th frame) -> (X, y) for R_real"""
    from wm.data.embodiment import REGISTRY, load
    from wm.evaluate import encode_clip
    X, y = [], []
    with torch.no_grad():
        for p in sorted(glob.glob(os.path.join(DATA, "fit", "*.npz"))):
            c = load(p, REGISTRY["b1"], lazy_frames=True)
            d = np.load(p, allow_pickle=True)
            idx = list(range(0, len(d["clearance"]), K))
            E = encode_clip(enc, np.asarray(c["frames"])[idx], 8).float().mean(1).cpu().numpy()
            X.append(E); y.append(d["clearance"][idx])
    return np.concatenate(X), np.concatenate(y)


def main():
    from sklearn.linear_model import RidgeCV
    from rollout_state_action_anova import Models
    from vjepa2_encoder import VJEPA2FrameEncoder
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NAME=path (a merged checkpoint with the B1 projector)")
    ap.add_argument("--set", default="v1", choices=("v1", "v2"), help="v2 = mirrored state set (F334 fix)")
    a = ap.parse_args()
    global DATA, CHK
    if a.set == "v2":
        DATA = os.path.join(ROOT, "data/wall_predcheck_v2/b1"); CHK = os.path.join(ROOT, "results/check/wall_predcheck_v2")
    name, ck = a.ckpt.split("=", 1)
    m = Models(os.path.join(ROOT, ck), "b1", 12, "cuda")      # a grayscale checkpoint switches the encoder (from_checkpoint)
    K = m.stride
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    alphas = np.logspace(-1, 6, 15)
    Xr, yr = realfit_data(m, enc, K)
    R_real = RidgeCV(alphas=alphas).fit(Xr, yr)
    import pickle
    cache = os.path.join(CHK, f"predcheck_{name}_embeddings.pkl")   # imagined / real pooled embeddings: re-analysis
    if os.path.exists(cache):                                         # without re-encoding
        fit, test = pickle.load(open(cache, "rb"))
    else:
        fit, test = collect(m, enc, "fit", K), collect(m, enc, "test", K)
        os.makedirs(CHK, exist_ok=True)
        pickle.dump((fit, test), open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    lines = [f"{name} ({ck}); B1 wall prediction check, constant command held, {len(fit)} fit / {len(test)} test start "
             f"states x 24 commands; horizons in FTM steps (0.25 s each); 8 = beyond every trained chain",
             "row: readout | within-state Spearman (mean over states) | MAE m | unsafe recall / precision (unsafe = true "
             f"min clearance over [0, t] < {UNSAFE} m)"]
    # command-only control: the fit set's mean change of TRUE clearance per command at horizon k, added to the start
    # distance read from e_0 (no prediction from the image beyond the current distance)
    dchg = {k: np.mean([np.asarray(s["true"][k]) - s["c0"] for s in fit], axis=0) for k in HZ}
    for k in HZ:
        R_imag = RidgeCV(alphas=alphas).fit(np.concatenate([np.asarray(s["imag"][k]) for s in fit]),
                                            np.concatenate([s["true"][k] for s in fit]))
        rows = {"real future (R_real on the true frame)": lambda s: R_real.predict(np.asarray(s["real"][k])),
                "imagined, R_real (pipeline as is)": lambda s: R_real.predict(np.asarray(s["imag"][k])),
                "imagined, R_imag (refit on imagined)": lambda s: R_imag.predict(np.asarray(s["imag"][k])),
                "copy e_0 (no prediction, distance only)": lambda s: np.repeat(R_real.predict(s["e0"][None]), 24),
                "command-only: e_0 distance + mean change": lambda s: R_real.predict(s["e0"][None])[0] + dchg[k],
                "command-only, TRUE start clearance + mean": lambda s: s["c0"] + dchg[k]}
        lines.append(f"--- horizon {k} step(s) = {k * 0.25:.2f} s" + ("  [beyond trained chain]" if k > 4 else ""))
        for rn, f in rows.items():
            sp, err, tp, fn, fp = [], [], 0, 0, 0
            for s in test:
                pr, tr, tm = f(s), np.asarray(s["true"][k]), np.asarray(s["tmin"][k])
                sp.append(spearman(pr, tr)); err.append(np.abs(pr - tr).mean())
                u, pu = tm < UNSAFE, pr < UNSAFE
                tp += int((u & pu).sum()); fn += int((u & ~pu).sum()); fp += int((~u & pu).sum())
            n_u = tp + fn
            rp = (f"{tp / n_u:.2f} / {tp / max(tp + fp, 1):.2f} (n unsafe {n_u})" if n_u else "n/a (no unsafe case)")
            sp_ = np.array(sp, float)
            spt = f"{np.nanmean(sp_):+.2f}" if np.isfinite(sp_).any() else "  n/a"
            lines.append(f"  {rn:42s} | {spt} | {np.mean(err):.2f} | {rp}")
    os.makedirs(CHK, exist_ok=True)
    open(os.path.join(CHK, f"predcheck_{name}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
