"""Which input representation of (e_t, e_next) best recovers real delta-Froude? Offline probing,
no retrain -- this picks what to retrain the state head on, it does not retrain it.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/embedding_representation_sweep.py

**Why this run exists.** `embedding_transition_ceiling.py` found the raw embedding pair DOES
encode the action-transition signal (Spearman rho +0.35 to +0.45 with `concat(e_t, e_next)`,
pooled) -- far more than the `delta = e_next - e_t` representation the state head is actually
built on (rho +0.02 to +0.22). That is an input-representation finding, not a loss finding: the
pipeline was fine, the READOUT was handed a lossy summary of what the encoder already has. But
`concat` was the first alternative tried, not necessarily the best one. This sweeps six
representations, ranks them by held-out Spearman rho per Froude channel, and the winner is what
gets retrained on next -- not a guess.

**Six equations, in increasing order of what they're allowed to see:**
  1. `delta = e_next - e_t`, pooled (1408-D) -- the baseline the state head already uses
  2. `concat = [e_t, e_next]`, pooled (2816-D) -- both frames, no interaction term
  3. `concat + delta = [e_t, e_next, e_next-e_t]`, pooled (4224-D) -- explicit delta alongside
     both frames (redundant information, but ridge/kNN may still use it differently)
  4. `bilinear = [e_t, e_next, e_t*e_next]`, pooled (4224-D) -- an explicit elementwise
     interaction term instead of a linear difference
  5. `z = ITM(e_t, e_next)` (64-D) -- the ALREADY-TRAINED cross-attention module this project's
     own pipeline computes from the full spatial token grids, zero extra training; tests whether
     the existing spatial cross-attention pathway already carries more signal than any pooled
     hand-built feature, and whether the failure is specifically downstream of `z` (state head's
     extra pool+offset+MLP processing) rather than in getting from frames to `z` at all
  6. a NEW, small cross-attention probe, fit briefly and only for this test: full 256-token grids
     for both frames (not pooled), one cross-attention layer (`e_t`'s tokens as queries into
     `e_next`'s tokens), pooled only at the very end, trained directly to regress delta-Froude
     (not to reconstruct anything) -- the "spatial, not pooled" arm this project's own record
     (F162/F163: attention-over-tokens beats pooling by ~0.08 R2 on egocentric command decoding,
     both bodies) says is worth checking before concluding concat is the ceiling

Probes 1-4: ridge + kNN (k=5, 15), held out by clip (20%). Probe 5: same, on the 64-D `z`. Probe 6
is its own small trained network (a probe, not a retrain of anything in the checkpoint) -- 1500
iterations, held-out evaluation only after training, same clip split throughout.

Ranks every representation by pooled median Spearman rho across the three Froude channels. That
ranking, not a guess, decides what `state_head` gets retrained on next.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
GRID = 16
TOK_DIM_IN = 1408
D = 64
PROBE6_ITERS = 1500
PROBE6_BATCH = 32
PROBE6_LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for body_stats/channels + the ALREADY-TRAINED ITM (equation 5) -- "
     "no FTM, no state head, nothing retrained below...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
itm.eval()
channels = [int(c) for c in cfg.body_channels]
mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

# ---- cache full token grids (GPU-resident, half precision -- the memory-safe pattern this
# session settled on after an earlier CPU-allocator OOM) plus pooled features and real delta-Froude
clips = []
for p in paths:
    clip = load(p, REGISTRY[EMBODIMENT])
    e = encode_clip(encoder, clip["frames"], 2).float()                    # [T, 256, 1408]
    bm = (np.asarray(clip["body_motion"])[:, channels] - mean_s) / std_s   # [T, 3]
    n = min(len(e), len(bm)) - 1
    clips.append({"e_full": e[:n + 1].half().to(device),                   # keep for the sweep
                 "bm": torch.tensor(bm[:n + 1], dtype=torch.float32, device=device),
                 "n": n, "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]

# ---- pooled e_t / e_next / real delta-bm, train and test, for equations 1-5 -----------------
def gather_pooled(clip_list):
    Et, Enext, Dbm = [], [], []
    for c in clip_list:
        pooled = c["e_full"].float().mean(1)              # [n+1, 1408]
        Et.append(pooled[:c["n"]]); Enext.append(pooled[1:c["n"] + 1])
        Dbm.append(c["bm"][1:c["n"] + 1] - c["bm"][:c["n"]])
    return (torch.cat(Et).cpu().numpy(), torch.cat(Enext).cpu().numpy(), torch.cat(Dbm).cpu().numpy())

Et_tr, Enext_tr, Dbm_tr = gather_pooled(train_clips)
Et_te, Enext_te, Dbm_te = gather_pooled(held_clips)
print(f"train: {len(Et_tr)} transitions, test: {len(Et_te)} transitions\n")

# equation 5: z = ITM(e_t, e_next), on the FULL token grids (real cross-attention, already trained)
def gather_z(clip_list):
    Z = []
    with torch.no_grad():
        for c in clip_list:
            e_full = c["e_full"]
            for s in range(0, c["n"], 32):
                sl = slice(s, min(s + 32, c["n"]))
                et = e_full[sl].float()
                enext = e_full[1:c["n"] + 1][sl].float()
                Z.append(itm(et, enext).cpu())
    return torch.cat(Z).numpy()

Z_tr, Z_te = gather_z(train_clips), gather_z(held_clips)

equations = {
    "1. delta = e_next-e_t (1408-D)": (Enext_tr - Et_tr, Enext_te - Et_te),
    "2. concat [e_t,e_next] (2816-D)": (np.concatenate([Et_tr, Enext_tr], 1),
                                        np.concatenate([Et_te, Enext_te], 1)),
    "3. concat+delta (4224-D)": (np.concatenate([Et_tr, Enext_tr, Enext_tr - Et_tr], 1),
                                 np.concatenate([Et_te, Enext_te, Enext_te - Et_te], 1)),
    "4. bilinear [e_t,e_next,e_t*e_next] (4224-D)": (
        np.concatenate([Et_tr, Enext_tr, Et_tr * Enext_tr], 1),
        np.concatenate([Et_te, Enext_te, Et_te * Enext_te], 1)),
    "5. z = ITM(e_t,e_next) (64-D, already trained)": (Z_tr, Z_te),
}

results = {}   # name -> (median rho, per-channel rho list, best probe description)
print(f"{'equation':<48}{'probe':<10}{'median rho':>12}   per-channel rho")
for name, (Xtr, Xte) in equations.items():
    scaler = StandardScaler().fit(Xtr)
    Xs_tr, Xs_te = scaler.transform(Xtr), scaler.transform(Xte)
    probes = {"ridge": Ridge(alpha=10.0).fit(Xs_tr, Dbm_tr),
             "kNN k=5": KNeighborsRegressor(n_neighbors=5).fit(Xs_tr, Dbm_tr),
             "kNN k=15": KNeighborsRegressor(n_neighbors=15).fit(Xs_tr, Dbm_tr)}
    best_med, best_desc, best_rhos = -1e9, None, None
    for pname, model in probes.items():
        pred = model.predict(Xs_te)
        rhos = [spearmanr(Dbm_te[:, c], pred[:, c])[0] for c in range(3)]
        med = float(np.median(rhos))
        if med > best_med:
            best_med, best_desc, best_rhos = med, pname, rhos
    print(f"{name:<48}{best_desc:<10}{best_med:>12.3f}   " +
         "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, best_rhos)))
    results[name] = (best_med, best_rhos, best_desc)

# ---- equation 6: a NEW, small cross-attention probe over full spatial tokens, trained briefly
class CrossAttnProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_t = nn.Linear(TOK_DIM_IN, D)
        self.proj_next = nn.Linear(TOK_DIM_IN, D)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        self.decoder = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, tok_t, tok_next):
        """tok_t, tok_next: [B,256,1408] -> predicted delta-Froude [B,3]"""
        q = self.proj_t(tok_t)                                  # [B,256,D], e_t as queries
        kv = self.proj_next(tok_next)                           # [B,256,D], e_next as keys/values
        out, _ = self.attn(q, kv, kv)                           # [B,256,D], token-wise interaction
        return self.decoder(out.mean(1))                        # pooled ONLY at the very end


def sample_transition_batch(clip_list, n):
    Tt, Tn, Db = [], [], []
    for _ in range(n):
        c = clip_list[rng.integers(0, len(clip_list))]
        t = rng.integers(0, c["n"])
        Tt.append(c["e_full"][t].float())
        Tn.append(c["e_full"][t + 1].float())
        Db.append(c["bm"][t + 1] - c["bm"][t])
    return torch.stack(Tt), torch.stack(Tn), torch.stack(Db)


probe6 = CrossAttnProbe().to(device)
opt = torch.optim.Adam(probe6.parameters(), lr=PROBE6_LR)
print(f"\ntraining probe 6 (cross-attention e_t<->e_next, spatial, not pooled), "
     f"{PROBE6_ITERS} iterations, {sum(p.numel() for p in probe6.parameters())} params...")
for it in range(PROBE6_ITERS):
    tok_t, tok_next, dbm = sample_transition_batch(train_clips, PROBE6_BATCH)
    pred = probe6(tok_t, tok_next)
    loss = F.mse_loss(pred, dbm)
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 300 == 0 or it == 0:
        print(f"  iter {it + 1:5d}  loss {loss.item():.4f}", flush=True)

probe6.eval()
with torch.no_grad():
    pred_list, true_list = [], []
    for c in held_clips:
        for s in range(0, c["n"], 32):
            sl = slice(s, min(s + 32, c["n"]))
            tok_t = c["e_full"][sl].float()
            tok_next = c["e_full"][1:c["n"] + 1][sl].float()
            pred_list.append(probe6(tok_t, tok_next).cpu())
            true_list.append((c["bm"][1:c["n"] + 1][sl] - c["bm"][sl]).cpu())
    pred6, true6 = torch.cat(pred_list).numpy(), torch.cat(true_list).numpy()
    rhos6 = [spearmanr(true6[:, c], pred6[:, c])[0] for c in range(3)]
    med6 = float(np.median(rhos6))
print(f"{'6. cross-attention probe, spatial (not pooled)':<48}{'trained':<10}{med6:>12.3f}   " +
     "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos6)))
results["6. cross-attention probe, spatial (not pooled)"] = (med6, rhos6, "trained")

print("\n" + "=" * 70)
print("RANKING (by median Spearman rho across the 3 Froude channels)")
print("=" * 70)
for name, (med, rhos, desc) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"  {med:+.3f}  {name}  (best probe: {desc})")

winner = max(results, key=lambda k: results[k][0])
print(f"\n-> WINNER: {winner} (median rho {results[winner][0]:+.3f})")
print("This is what the state head should be retrained on next -- not a guess.")
