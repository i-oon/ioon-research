"""Gate check on the isolated-head PASS: does it transfer to L_body's REAL target, or was it
specific to the delta-Froude convention this session's diagnostic scripts used?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/isolated_z_head_probe_bodytarget.py

**Why this exists.** `isolated_z_head_probe.py` found a clean, frozen-z head recovers full
action-lever signal (gap +0.38 to +0.73, bar 0.110) when trained to predict `bm_next - bm_t`
("delta-Froude") -- a convention this session's scripts adopted from the embedding-space lever
(`ftm(e_t,z)-e_t` vs `e_next-e_t`). But reading `wm/train.py` just now: `L_body` and `L_state`
BOTH actually train against `state_target = body_target = batch["body_motion"]` -- the ACHIEVED,
ABSOLUTE value at frame `t` (`wm/data/dataset.py`: `sample["body_motion"] = clip["body_motion"][t]`),
not a difference between two readings at all. That is a real target mismatch between this
session's diagnostic convention and what the actual pipeline optimizes. Before scoping any
retrain, this checks whether the PASS survives swapping to the real target.

**Identical setup to `isolated_z_head_probe.py`** -- same frozen z sources (`ITM(e_t,e_next)`,
`proj(action)`), same clean `body_head`-shaped network, same two loss shapes (MSE, cosine) --
with ONLY the target changed: `bm[t]` (standardised absolute body_motion), not `bm[t+1]-bm[t]`.

**Reading:**
  - PASSES (gap > 0.110, rho comparable to the delta-target run) -> the fix is real and transfers.
    Build stop-gradient on `L_body`, consolidate away `L_state`/delta, retrain.
  - FAILS -> the recoverable signal this session found is specific to the delta/change framing,
    not the absolute value `L_body` actually needs. `L_body`'s real target may be a harder,
    different problem -- which would explain why `L_body` also underperforms jointly, and no
    stop-gradient fix to the CURRENT target would resolve it. Would need its own diagnosis.

Same two z sources as every prior check in this arc: `z = ITM(e_t,e_next)` and `z = proj(action)`.
Same two loss shapes (MSE, cosine). Per-channel reported via SIGN-AGREEMENT GAP (real z's
sign-agreement rate with the true value, minus mean z's) -- well-defined regardless of loss shape
or target convention, unlike a magnitude-based win-rate.
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
# the ACTUAL L_body/L_state target (wm/data/dataset.py: sample["body_motion"] = clip["body_motion"][t])
# -- the achieved, absolute value at frame t, NOT a difference between two readings.
true_change = bm_t
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
             f"median rho={med_rho:+.3f}  [target: absolute body_motion[t], L_body's real target]")
        print("  per-channel rho:      " +
             "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos)))
        print("  per-channel sign-gap: " +
             "  ".join(f"{n}={g:+.3f}" for n, g in zip(CHANNEL_NAMES, sign_gaps)))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for (z_name, loss_name), (gap, med_rho, rhos, sign_gaps) in results.items():
    passes_lever = gap > BAR
    print(f"z={z_name:<6} loss={loss_name:<8} rho={med_rho:+.3f}  "
         f"lever={'PASS' if passes_lever else 'FAIL'} ({gap:+.3f})")

print("\n" + "=" * 70)
print("READING")
print("=" * 70)
any_pass = any(gap > BAR for gap, *_ in results.values())
if any_pass:
    print("-> TRANSFERS: the isolated-head fix recovers signal on L_body's REAL (absolute) "
         "target too, not just the delta-Froude convention this session's scripts used. The fix "
         "is real. Build stop-gradient on L_body, drop L_state/delta (dead+redundant, same "
         "target), retrain.")
else:
    print("-> DOES NOT TRANSFER: the recoverable signal this session found is specific to the "
         "delta/change framing, not the absolute value L_body actually needs. L_body's real "
         "target may be a harder, different problem than what the delta-based checks measured -- "
         "which would explain why L_body also underperforms jointly. No stop-gradient fix to the "
         "current target would resolve this on its own; needs its own diagnosis before any "
         "pipeline change.")
