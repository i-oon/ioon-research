"""Does proj(action) carry the same delta-Froude signal that ITM(e_t, e_next) does -- the
reconstruction-vs-planning gate, before any state-head retrain is scoped.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/proj_action_ceiling_check.py

**Why this has to run before retraining anything.** `embedding_representation_sweep.py` found
`z = ITM(e_t, e_next)` carries far more delta-Froude signal (median Spearman rho +0.535) than any
hand-built pooled feature, including the delta the state head is actually built on (rho +0.215).
But `ITM(e_t, e_next)` REQUIRES the real future frame -- it is a reconstruction-time quantity. At
control time, the only thing available is `z = proj(action)`, no `e_next`, no ITM. This project has
been burned by exactly this substitution before (F97: `a -> z` is one-to-many; F131: a head fitted
on ITM latents got WORSE when actually asked to score `proj(a)`, because the two latents occupy
different regions of the same space). **If `proj(a)` does not carry the same signal `ITM(e_t,e_next)`
does, retraining the state head to trust `z`'s content more only fixes offline scoring against a
real future -- not control, where `proj(a)` is all that exists.**

Runs the IDENTICAL probe as `embedding_representation_sweep.py`'s equation 5 (ridge + kNN k=5/15,
held out by clip, median Spearman rho across forward/lateral/yaw), on `z = proj(action, "b1")`
instead of `z = ITM(e_t, e_next)`. Same data, same split, same channels -- the only thing that
changes is which `z` is probed.

**Reading the result:**
  - rho close to 0.535 -> the signal survives the ITM-to-projector substitution. The fix works for
    CONTROL: retraining the state head to read `z`'s content is justified, not just for scoring
    with a real future in hand.
  - rho far below 0.535 -> the reconstruction-vs-planning gap again. `z`'s magnitude signal exists
    only when the real future is already known; retraining the state head on this basis would fix
    offline scoring/readout, not the controllable path. Scope a different fix, or scope this one
    with that limitation stated up front.

Diagnosis only; trains nothing but the offline probes, touches no checkpoint besides the frozen
encoder, `teacher_state.pt`'s body_stats, and the already-trained `proj` (action projector).
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
CKPT = os.path.join(ROOT, "wm/runs/beh12_state/teacher_state.pt")
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
ITM_Z_REFERENCE = 0.535   # embedding_representation_sweep.py's equation 5 result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for body_stats/channels + the ALREADY-TRAINED action projector "
     "`proj` -- no ITM, no FTM, no state head, nothing retrained below...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
proj.eval()
channels = [int(c) for c in cfg.body_channels]
mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

clips = []
for p in paths:
    clip = load(p, REGISTRY[EMBODIMENT])
    e = encode_clip(encoder, clip["frames"], 2).float()
    bm = (np.asarray(clip["body_motion"])[:, channels] - mean_s) / std_s
    actions = np.asarray(clip["actions"], dtype=np.float32)
    n = min(len(e), len(bm), len(actions)) - 1
    clips.append({"actions": torch.tensor(actions[:n], device=device),
                 "bm": torch.tensor(bm[:n + 1], dtype=torch.float32, device=device),
                 "n": n, "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]


def gather_proj_z(clip_list):
    Z, Dbm = [], []
    with torch.no_grad():
        for c in clip_list:
            for s in range(0, c["n"], 64):
                sl = slice(s, min(s + 64, c["n"]))
                a = c["actions"][sl]
                Z.append(proj(a, EMBODIMENT).cpu())
                Dbm.append((c["bm"][1:c["n"] + 1][sl] - c["bm"][sl]).cpu())
    return torch.cat(Z).numpy(), torch.cat(Dbm).numpy()


Z_tr, Dbm_tr = gather_proj_z(train_clips)
Z_te, Dbm_te = gather_proj_z(held_clips)
print(f"train: {len(Z_tr)} transitions, test: {len(Z_te)} transitions, z dim = {Z_tr.shape[1]}\n")

scaler = StandardScaler().fit(Z_tr)
Zs_tr, Zs_te = scaler.transform(Z_tr), scaler.transform(Z_te)
probes = {"ridge": Ridge(alpha=10.0).fit(Zs_tr, Dbm_tr),
         "kNN k=5": KNeighborsRegressor(n_neighbors=5).fit(Zs_tr, Dbm_tr),
         "kNN k=15": KNeighborsRegressor(n_neighbors=15).fit(Zs_tr, Dbm_tr)}

print(f"{'probe':<10}{'median rho':>12}   per-channel rho")
best_med, best_desc, best_rhos = -1e9, None, None
for pname, model in probes.items():
    pred = model.predict(Zs_te)
    rhos = [spearmanr(Dbm_te[:, c], pred[:, c])[0] for c in range(3)]
    med = float(np.median(rhos))
    print(f"{pname:<10}{med:>12.3f}   " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos)))
    if med > best_med:
        best_med, best_desc, best_rhos = med, pname, rhos

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print(f"proj(action) -> delta-Froude, best probe ({best_desc}): median rho = {best_med:+.3f}")
print(f"ITM(e_t,e_next) -> delta-Froude (reference, embedding_representation_sweep.py): "
     f"median rho = {ITM_Z_REFERENCE:+.3f}")
ratio = best_med / ITM_Z_REFERENCE if ITM_Z_REFERENCE else float("nan")
print(f"ratio: {ratio:.1%} of the ITM-z signal survives the reconstruction->control substitution")
if best_med > 0.7 * ITM_Z_REFERENCE:
    print("\n-> SURVIVES: proj(a) carries most of what ITM(e_t,e_next) carries. The fix works for "
         "CONTROL -- retraining the state head to read z's content as a primary signal is "
         "justified, not just for offline scoring against a real future.")
else:
    print("\n-> DOES NOT SURVIVE: this is the reconstruction-vs-planning gap again. z's "
         "magnitude signal is much weaker when built from the action alone, without the real "
         "future. Retraining the state head on this basis would fix offline scoring/readout "
         "(where e_next is available) but not the controllable path (where only proj(a) is). "
         "Scope a different fix, or scope this one with that limitation stated up front.")
