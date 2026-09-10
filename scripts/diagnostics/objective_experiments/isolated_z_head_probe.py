"""Isolated frozen-z head: does a TRAINED head reach the offline probe's rho once every
confound this session has flagged is removed -- competing losses (L_recon/L_motion pulling z
around) and the loss shape (MSE optimizing magnitude, possibly wrecking the direction a
Spearman/cosine metric actually measures)?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/isolated_z_head_probe.py

**Why this, not another architecture change.** `embedding_representation_sweep.py` found
`z = ITM(e_t,e_next)` recovers real delta-Froude at median rho +0.535, offline, via ridge/kNN.
`zonly_action_lever_check.py` found the ALREADY-TRAINED z-only state head (F177,
`state_use_delta=false`, trained jointly with `L_recon + L_motion + L_body + L_state`) fails the
action-lever (+0.045 / +0.099, bar 0.110) and is actively worse than a mean guess on forward/yaw.
The gap between those two numbers is the open question, not "try another architecture": an offline
kNN/ridge probe and a network trained jointly with three other loss terms are not the same
experiment, and the state-head/delta branch this session has hammered on throughout is a dead
branch regardless (F177: `z+delta` R2 0.625 < `z` alone 0.781 -- delta actively hurts).

**This isolates the two remaining candidate causes cheaply, one small head, two loss variants:**
  - FREEZE the encoder, ITM, FTM, and action projector -- `z` is computed once and never updated,
    matching exactly what the offline probe saw (a fixed, unmoving feature).
  - Train ONLY a small, freshly-initialised head (`LayerNorm(z_dim) -> Linear -> GELU -> Linear`,
    the same shape as `MotionDecoder.body_head`, NOT the state-head/delta branch) -- on ONLY a
    Froude loss, no `L_recon`/`L_motion` in the loop to pull `z` toward appearance.
  - Two loss shapes: MSE (standard) and a cosine/direction loss (`1 - cosine_similarity`) --
    the offline probes (kNN, ridge scored by Spearman rho) are direction-sensitive and
    magnitude-tolerant in a way plain MSE is not; this session has repeatedly seen R2-negative,
    rho-positive results (F175's MLP, this session's ridge-vs-kNN gaps), consistent with MSE
    optimizing magnitude at the expense of the direction rho actually measures.

Two z sources tested (matching every prior check in this arc): `z = ITM(e_t,e_next)` (the +0.535
reference) and `z = proj(action)` (the control-relevant path, +0.429 offline). Target is
delta-Froude (`bm_next - bm_t`), matching this arc's own action-lever metric -- NOT `L_body`'s
absolute-Froude target; that is a related but different question, noted so the two are not
conflated.

**Pre-registered readings:**
  - Isolated head (either loss) reaches rho ~0.5, matching the offline probe -> the wall was JOINT
    TRAINING: competing losses pull `z` away from the Froude signal during end-to-end training.
    Fix: protect the Froude readout from `L_recon`/`L_motion` (isolated training, stop-gradient,
    or a much higher relative weight) -- cheap.
  - MSE fails but COSINE reaches rho ~0.5 -> the fix is the loss shape, not joint training: swap
    the state/body Froude loss from MSE to a direction-aware loss.
  - NEITHER loss recovers it on frozen z -> deeper than loss or competition: a trained network
    genuinely cannot fit what a static kNN/ridge probe fits on the same frozen features, which
    would be the more surprising and more serious finding.

Per-channel reported via SIGN-AGREEMENT GAP (real z's sign-agreement rate with the true change,
minus mean z's), not the raw-magnitude win-rate `zonly_action_lever_check.py` used -- that metric
assumes a comparable output scale, which the cosine loss does not guarantee (it is
scale-invariant by construction). Sign agreement is well-defined regardless of loss shape.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
CKPT = os.path.join(ROOT, "wm/runs/beh12_state/teacher_state.pt")   # the ORIGINAL z+delta checkpoint
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
HIDDEN = 128           # matches MotionDecoder.body_head's cfg.body_hidden
ITERS = 1500
BATCH = 64
LR = 1e-3
BAR = 2 * 0.055
REFERENCE_RHO = {"itm": 0.535, "proj": 0.429}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for the frozen encoder/ITM/proj/body_stats -- FTM untouched, "
     "nothing here is retrained beyond one small new head...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
channels = [int(c) for c in cfg.body_channels]
mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)], dtype=torch.float32, device=device)
std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)], dtype=torch.float32, device=device)
Z_DIM = cfg.z_dim

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

E_t, E_next, Bm_t, Bm_next, Actions, Held = [], [], [], [], [], []
for p in paths:
    clip = load(p, REGISTRY[EMBODIMENT])
    e = encode_clip(encoder, clip["frames"], 2).float()
    bm = (torch.tensor(np.asarray(clip["body_motion"])[:, channels], dtype=torch.float32, device=device) -
         mean_s) / std_s
    actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32, device=device)
    n = min(len(e), len(bm), len(actions)) - 1
    for t in range(1, n, 4):
        E_t.append(e[t]); E_next.append(e[t + 1])
        Bm_t.append(bm[t]); Bm_next.append(bm[t + 1])
        Actions.append(actions[t]); Held.append(p in held_paths)

e_t_cpu = torch.stack(E_t).cpu()
e_next_cpu = torch.stack(E_next).cpu()
bm_t = torch.stack(Bm_t)
bm_next = torch.stack(Bm_next)
actions_all = torch.stack(Actions)
held_mask = torch.tensor(Held)
true_change = bm_next - bm_t
print(f"{len(e_t_cpu)} transitions, {held_mask.sum().item()} held out\n")
torch.cuda.empty_cache()

# ---- precompute BOTH frozen z sources once, for every transition ----------------------------
SBATCH = 16
with torch.no_grad():
    z_itm_list = []
    for s in range(0, len(e_t_cpu), SBATCH):
        sl = slice(s, s + SBATCH)
        z_itm_list.append(itm(e_t_cpu[sl].float().to(device), e_next_cpu[sl].float().to(device)))
    z_itm = torch.cat(z_itm_list)
    z_proj = proj(actions_all, EMBODIMENT)
z_sources = {"itm": z_itm.detach(), "proj": z_proj.detach()}

train_idx = (~held_mask).nonzero(as_tuple=True)[0].to(device)
held_idx = held_mask.nonzero(as_tuple=True)[0].to(device)


class CleanHead(nn.Module):
    """Exactly MotionDecoder.body_head's shape -- z only, no delta, no frame."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(Z_DIM), nn.Linear(Z_DIM, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, len(channels)))

    def forward(self, z):
        return self.net(z)


def cosine_loss(pred, true):
    return (1 - F.cosine_similarity(pred, true, dim=1)).mean()


results = {}
for z_name, z_all in z_sources.items():
    for loss_name in ("mse", "cosine"):
        torch.manual_seed(SEED)
        head = CleanHead().to(device)
        opt = torch.optim.Adam(head.parameters(), lr=LR)
        for it in range(ITERS):
            idx = train_idx[torch.randint(0, len(train_idx), (BATCH,), device=device)]
            pred = head(z_all[idx])
            true = true_change[idx]
            loss = F.mse_loss(pred, true) if loss_name == "mse" else cosine_loss(pred, true)
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            pred_real = head(z_all[held_idx])
            true_h = true_change[held_idx]
            mean_z = z_all[train_idx].mean(0, keepdim=True).expand(len(held_idx), -1)
            pred_mean = head(mean_z)

            cos_real = F.cosine_similarity(pred_real, true_h, dim=1)
            cos_mean = F.cosine_similarity(pred_mean, true_h, dim=1)
            gap = cos_real.median().item() - cos_mean.median().item()

            from scipy.stats import spearmanr
            rhos = [spearmanr(true_h[:, c].cpu().numpy(), pred_real[:, c].cpu().numpy())[0]
                   for c in range(len(channels))]
            med_rho = float(np.median(rhos))

            sign_gaps = []
            for c in range(len(channels)):
                real_agree = (torch.sign(pred_real[:, c]) == torch.sign(true_h[:, c])).float().mean().item()
                mean_agree = (torch.sign(pred_mean[:, c]) == torch.sign(true_h[:, c])).float().mean().item()
                sign_gaps.append(real_agree - mean_agree)

        results[(z_name, loss_name)] = (gap, med_rho, rhos, sign_gaps)
        print(f"z={z_name:<6} loss={loss_name:<8} action-lever gap={gap:+.3f} (bar {BAR:.3f})  "
             f"median rho={med_rho:+.3f} (offline ref {REFERENCE_RHO[z_name]:+.3f})")
        print("  per-channel rho:      " +
             "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos)))
        print("  per-channel sign-gap: " +
             "  ".join(f"{n}={g:+.3f}" for n, g in zip(CHANNEL_NAMES, sign_gaps)))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for (z_name, loss_name), (gap, med_rho, rhos, sign_gaps) in results.items():
    ref = REFERENCE_RHO[z_name]
    close_to_offline = med_rho > 0.7 * ref
    passes_lever = gap > BAR
    print(f"z={z_name:<6} loss={loss_name:<8} rho={med_rho:+.3f} "
         f"({'MATCHES offline' if close_to_offline else 'does NOT match offline'} {ref:+.3f})  "
         f"lever={'PASS' if passes_lever else 'FAIL'} ({gap:+.3f})")

print("\n" + "=" * 70)
print("READING")
print("=" * 70)
any_matches = any(r[1] > 0.7 * REFERENCE_RHO[k[0]] for k, r in results.items())
mse_matches = any(r[1] > 0.7 * REFERENCE_RHO[k[0]] for k, r in results.items() if k[1] == "mse")
cos_matches = any(r[1] > 0.7 * REFERENCE_RHO[k[0]] for k, r in results.items() if k[1] == "cosine")
if any_matches and not mse_matches and cos_matches:
    print("-> Isolated head reaches offline rho with COSINE but not MSE: the fix is the LOSS "
         "SHAPE, not joint training. MSE optimizes magnitude at the expense of direction; a "
         "direction-aware loss on the Froude readout is the cheap fix.")
elif any_matches:
    print("-> Isolated head reaches offline rho (at least one condition): the wall was JOINT "
         "TRAINING -- competing losses (L_recon/L_motion) pull z away from the Froude signal. "
         "Fix: protect the Froude readout from those losses (isolated training, stop-gradient, "
         "or a much higher relative weight).")
else:
    print("-> NEITHER loss recovers the offline rho on frozen z. Deeper than loss shape or "
         "training competition: a trained network cannot fit what a static kNN/ridge probe fits "
         "on the same frozen features. This is the more serious finding -- report before any "
         "further pipeline change.")
