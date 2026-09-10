"""The untested cell: does FTM's OWN predicted next-embedding hold Froude-outcome signal, readable
by a clean, stop-gradient-protected head -- distinct from `direct` (no FTM), `f179` (multi-step
embedding rollout decoded via ITM), and the old `state`/-0.051 test (delta-based dead branch, no
stop-gradient, reused an unprotected head).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/ftm_froude_stopgrad_probe.py

**The cell, precisely.** `pred_next = FTM(e_t, z)` (frozen, no grad -- FTM's own training was
never touched by the stop-gradient fix on L_body). A NEW, freshly-initialised head
(`LayerNorm(pool_dim)->Linear->GELU->Linear`, the same shape as the winning `body_head`) reads
`pool(pred_next)` and predicts the ACTION'S OUTCOME -- `body_motion[t+1]`, the state one step
after taking the action, matching what a rollout is actually supposed to represent (not
`body_motion[t]`, which is what the OLD state_head's target oddly was). Trained on Froude loss
ALONE, `pred_next.detach()`, so this head's gradient never reaches FTM -- same stop-gradient
mechanism as the winning `body_head` fix, applied one step downstream.

**Why this differs from the -0.051 result.** That test reused the ALREADY-EXISTING `state_head`
(delta-based: `pool(ftm(e_t,z)-e_t) + z_proj(z)`, F177's confirmed-harmful architecture) on a
checkpoint where THAT head was trained under full joint competition -- conflating two already-
identified confounds (bad architecture, no stop-gradient) into one number. This test removes both:
no delta, no z_proj, no joint competition -- a clean test of whether FTM's raw predicted embedding
carries the signal at all.

**Pre-registered reading:**
  - Clears the action-lever (comparable to the z-fix's +0.38 to +1.2 wide margins) -> FTM's own
    output holds Froude signal in an extractable form; stop-gradient rescues it the same way it
    rescued `z`. Proceed to the live ranking test (condition_confusion.py) -- the real gate: does
    this beat `direct`'s 3.25 mean-rank, i.e. does the world model's OWN rollout earn a place a
    non-rollout scorer doesn't already have.
  - Fails/weak -> FTM's 1408-D, appearance-heavy, `L_recon`-optimised output does not concentrate
    Froude-relevant directions the way `z`'s 64-D transition-distilling bottleneck does, by
    construction. Real, specific information about WHY rollout-based Froude prediction fails here
    -- not another unexplained null.
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

EMBODIMENT = "hexapod"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat")   # matches
                     # condition_confusion.py's ranking-test embodiment -- the B1 run above stays
                     # valid on its own, but this checkpoint is what the ranking integration needs
CKPT = os.path.join(ROOT, "wm/runs/beh12_body_stopgrad/teacher_stopgrad.pt")
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
POOL_DIM = 1408
HIDDEN = 128
ITERS = 1500
BATCH = 64
LR = 1e-3
BAR = 2 * 0.055
SBATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading beh12_body_stopgrad/teacher_stopgrad.pt -- FTM frozen, untouched by the stop-"
     "gradient fix (that only protected L_body's read of z, not FTM's own training)...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
channels = [int(c) for c in cfg.body_channels]
mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)], dtype=torch.float32, device=device)
std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)], dtype=torch.float32, device=device)

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
bm_next = torch.stack(Bm_next)   # the OUTCOME target: state one step after the action
actions_all = torch.stack(Actions)
held_mask = torch.tensor(Held)
print(f"{len(e_t_cpu)} transitions, {held_mask.sum().item()} held out\n")
torch.cuda.empty_cache()

# ---- precompute FTM's predicted next-embedding (frozen, detached) for BOTH z sources -----------
with torch.no_grad():
    z_itm_list = []
    for s in range(0, len(e_t_cpu), SBATCH):
        sl = slice(s, s + SBATCH)
        z_itm_list.append(itm(e_t_cpu[sl].float().to(device), e_next_cpu[sl].float().to(device)))
    z_itm = torch.cat(z_itm_list)
    z_proj = proj(actions_all, EMBODIMENT)

    pred_next_pooled = {}
    for name, z_all in (("itm", z_itm), ("proj", z_proj)):
        pooled_list = []
        for s in range(0, len(e_t_cpu), SBATCH):
            sl = slice(s, s + SBATCH)
            e_t_b = e_t_cpu[sl].float().to(device)
            pred_next = ftm(e_t_b, z_all[sl], EMBODIMENT)          # FTM's OWN predicted embedding
            pooled_list.append(pred_next.mean(1))                  # pool over tokens
        pred_next_pooled[name] = torch.cat(pooled_list).detach()   # detach: never touches FTM

train_idx = (~held_mask).nonzero(as_tuple=True)[0].to(device)
held_idx = held_mask.nonzero(as_tuple=True)[0].to(device)


class CleanFroudeHead(nn.Module):
    """Same shape as the winning body_head -- reads FTM's pooled output, nothing else."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(POOL_DIM), nn.Linear(POOL_DIM, HIDDEN), nn.GELU(),
                                 nn.Linear(HIDDEN, len(channels)))

    def forward(self, x):
        return self.net(x)


results = {}
trained_heads = {}
for z_name, pooled_pred in pred_next_pooled.items():
    torch.manual_seed(SEED)
    head = CleanFroudeHead().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    for it in range(ITERS):
        idx = train_idx[torch.randint(0, len(train_idx), (BATCH,), device=device)]
        pred = head(pooled_pred[idx])
        loss = F.mse_loss(pred, bm_next[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        pred_real = head(pooled_pred[held_idx])
        true_h = bm_next[held_idx]
        mean_input = pooled_pred[train_idx].mean(0, keepdim=True).expand(len(held_idx), -1)
        pred_mean = head(mean_input)

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

    results[z_name] = (gap, med_rho, rhos, sign_gaps)
    trained_heads[z_name] = head.state_dict()
    print(f"z={z_name:<6} action-lever gap={gap:+.3f} (bar {BAR:.3f})  median rho={med_rho:+.3f}")
    print("  per-channel rho:      " +
         "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, rhos)))
    print("  per-channel sign-gap: " +
         "  ".join(f"{n}={g:+.3f}" for n, g in zip(CHANNEL_NAMES, sign_gaps)))

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
best_gap = max(gap for gap, *_ in results.values())
if best_gap > BAR:
    print(f"-> CLEARS (best gap {best_gap:+.3f} > {BAR:.3f}): FTM's own predicted embedding holds "
         "Froude-outcome signal, and stop-gradient extracts it. Proceed to the live ranking test "
         "(condition_confusion.py) -- does this beat direct's 3.25 mean-rank.")
else:
    print(f"-> DOES NOT CLEAR (best gap {best_gap:+.3f} <= {BAR:.3f}): FTM's predicted embedding "
         "does not concentrate Froude-relevant directions the way z's bottleneck does. Specific, "
         "real information about why rollout-based Froude prediction fails here -- not another "
         "unexplained null. Do not proceed to the ranking test on this basis.")

torch.save({"heads": trained_heads, "channels": channels}, os.path.join(ROOT,
          "wm/runs/ftm_froude_stopgrad_head_hexapod.pt"))
print("\nsaved -> wm/runs/ftm_froude_stopgrad_head_hexapod.pt (both z=itm and z=proj heads)")
