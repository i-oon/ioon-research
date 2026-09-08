"""Is the action-transition signal even IN the raw V-JEPA2 embedding, independent of any trained
module (ITM/FTM/state head) this project has built on top of it?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/embedding_transition_ceiling.py

**Why this is the one test left before blaming the encoder.** Test 1
(`ground_truth_action_flatness.py`) showed real signal in (action -> real Froude): within a fixed
behaviour, a clip's real action deviating from the mean predicts its real Froude outcome
deviating too (p=0.002). Every kill-gate since then (`sequence_context_killgate.py`,
`spatial_recurrent_killgate.py`) tested whether a TRAINED model -- pooled or spatial, recurrent or
not -- could recover that signal from `(e_t, action)`, and all failed. None of those tested
whether the signal is present in `(e_t, e_t+1)` ALONE, with no model in the loop at all, which is a
different and more basic question: **does the raw, frozen V-JEPA2 embedding pair even encode the
achieved physical change, independent of whether any of this project's trained heads can read it
out?**

This follows the exact methodology F190 already used successfully for the hexapod's fine
speed-magnitude signal (`embedding_speed_ceiling.py`: raw embedding delta, off-the-shelf probes,
held out by clip) -- applied here to B1, to the FULL 3-channel body-motion target, and to the
actual quantity this arc cares about: the CHANGE (`bm_next - bm_t`), not just the achieved value.

**Two feature sets, both stronger than anything any trained module in this pipeline uses:**
  - `delta = e_next - e_t` (pooled, 1408-D) -- what the state-head's own math is built on
  - `concat(e_t, e_next)` (pooled, 2816-D) -- strictly more information than the delta alone
    (delta is a linear projection of this), the strongest fair probe

**Two probes**: kNN (k=5, 15, curse-of-dimensionality-robust, no assumptions) and ridge regression
(linear, cheap, a lower bound). Held out by clip (20%), matching this session's own B1 convention.
Reports R2 and Spearman rho per channel (forward/lateral/yaw) plus the pooled 3-D R2.

**Reading the result:**
  - Any probe clears a real R2/rho on `concat(e_t, e_next)` -> the information IS in the raw
    embedding; every null this session measured (Test 2's three kill-gates) is a TRAINED-MODEL
    failure to extract/use it, not an encoder limit. Still a real, stubborn problem -- but the
    encoder is not where it lives.
  - All probes read null on BOTH feature sets -> V-JEPA2's frozen latent genuinely does not encode
    the action-transition signal at the resolution this task needs. The encoder is the wall, at
    the source, not downstream of it -- confirmed rather than inferred.

Diagnosis only; trains nothing but the offline probes, touches no checkpoint besides the frozen
encoder and `teacher_state.pt`'s body-motion standardisation.
"""
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from teacher_student_insect import load_teacher  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

EMBODIMENT = "b1"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CKPT = os.path.join(ROOT, "wm/runs/beh12_state/teacher_state.pt")   # only for body_stats/channels
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt only for body_stats/channels (standardisation) -- no ITM/FTM/state "
     "head touched below; every feature is the raw frozen encoder output...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
channels = [int(c) for c in cfg.body_channels]
mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

E_t_tr, E_next_tr, DBM_tr = [], [], []
E_t_te, E_next_te, DBM_te = [], [], []
for p in paths:
    clip = load(p, REGISTRY[EMBODIMENT])
    e = encode_clip(encoder, clip["frames"], 2).float()
    pooled = e.mean(1).cpu().numpy()                                    # [T, 1408], raw, frozen
    bm = (np.asarray(clip["body_motion"])[:, channels] - mean_s) / std_s  # [T, 3], standardised
    n = min(len(pooled), len(bm)) - 1
    et, enext = pooled[:n], pooled[1:n + 1]
    dbm = bm[1:n + 1] - bm[:n]                                          # real delta-Froude
    if p in held_paths:
        E_t_te.append(et); E_next_te.append(enext); DBM_te.append(dbm)
    else:
        E_t_tr.append(et); E_next_tr.append(enext); DBM_tr.append(dbm)

E_t_tr, E_next_tr, DBM_tr = map(np.concatenate, (E_t_tr, E_next_tr, DBM_tr))
E_t_te, E_next_te, DBM_te = map(np.concatenate, (E_t_te, E_next_te, DBM_te))
print(f"train: {len(E_t_tr)} transitions, test: {len(E_t_te)} transitions "
     f"(held out {n_held} of {len(paths)} clips)\n")

feature_sets = {
    "delta = e_next - e_t (1408-D)": (E_next_tr - E_t_tr, E_next_te - E_t_te),
    "concat(e_t, e_next) (2816-D)": (np.concatenate([E_t_tr, E_next_tr], 1),
                                     np.concatenate([E_t_te, E_next_te], 1)),
}

print(f"{'feature set':<32}{'probe':<16}{'R2 (pooled 3D)':>16}   per-channel R2 / rho")
best_r2 = -1e9
for fname, (Xtr, Xte) in feature_sets.items():
    scaler = StandardScaler().fit(Xtr)
    Xs_tr, Xs_te = scaler.transform(Xtr), scaler.transform(Xte)

    probes = {
        "ridge": Ridge(alpha=10.0).fit(Xs_tr, DBM_tr),
        "kNN k=5": KNeighborsRegressor(n_neighbors=5).fit(Xs_tr, DBM_tr),
        "kNN k=15": KNeighborsRegressor(n_neighbors=15).fit(Xs_tr, DBM_tr),
    }
    for pname, model in probes.items():
        pred = model.predict(Xs_te)
        ss_res = ((DBM_te - pred) ** 2).sum()
        ss_tot = ((DBM_te - DBM_tr.mean(0)) ** 2).sum()
        r2_pooled = 1 - ss_res / ss_tot
        best_r2 = max(best_r2, r2_pooled)
        per_ch = []
        for c, name in enumerate(CHANNEL_NAMES):
            ss_res_c = ((DBM_te[:, c] - pred[:, c]) ** 2).sum()
            ss_tot_c = ((DBM_te[:, c] - DBM_tr[:, c].mean()) ** 2).sum()
            r2_c = 1 - ss_res_c / ss_tot_c
            rho_c, _ = spearmanr(DBM_te[:, c], pred[:, c])
            per_ch.append(f"{name} R2={r2_c:+.3f}/rho={rho_c:+.3f}")
        print(f"{fname:<32}{pname:<16}{r2_pooled:>+16.3f}   " + "  ".join(per_ch))
    print()

print("=" * 70)
print("VERDICT")
print("=" * 70)
if best_r2 > 0.05:
    print(f"-> SIGNAL PRESENT (best pooled R2 = {best_r2:+.3f}): the raw V-JEPA2 embedding pair "
         "DOES encode the action-transition/delta-Froude signal above a null model. Every kill-gate "
         "null this session measured (pooled-GRU, spatial-attention-GRU, ConvGRU) is a TRAINED-"
         "MODEL failure to extract/use signal that IS there -- a real problem, but not an encoder "
         "limit. The encoder is not the wall.")
else:
    print(f"-> NO SIGNAL (best pooled R2 = {best_r2:+.3f}, at or below a null/mean-predicting "
         "model): even the strongest fair probe -- full (e_t, e_next), no model, no training "
         "beyond kNN/ridge -- cannot recover the action-transition signal. V-JEPA2's frozen latent "
         "does not encode this at the resolution the task needs. The encoder is the wall, "
         "confirmed at the source, not inferred from downstream failures.")
