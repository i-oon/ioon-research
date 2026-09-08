"""Stage 1 RSSM kill-gate: does a REAL stochastic posterior/prior + KL latent (the field-standard
DreamerV3/RSSM ingredient no test in this arc has actually included) move rollout mean-rank toward
the oracle, where the deterministic ConvGRU (no KL, no stochastic latent) failed?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/rssm_stage1_gate.py

**What's actually new here, and what isn't.** R0 and the ConvGRU test were both deterministic,
teacher-forced-only recurrence -- real recurrent state, but no stochastic latent, no KL, no
imagined-vs-real consistency training. This is the one architectural ingredient every cited
working system (DreamerV3, DreamMimic, NE-Dreamer) has that nothing in this arc has tested:

    h_t = GRUCell([z_{t-1}, a_{t-1}], h_{t-1})           deterministic recurrent state
    posterior: z_t ~ q(z_t | h_t, e_t)                   sees the REAL frame
    prior:    zhat_t ~ p(zhat_t | h_t)                   does NOT see the real frame
    KL(q(z_t|h_t,e_t) || p(zhat_t|h_t))                  the actual "dynamics loss"
    Froude head: (h_t, z_t) -> body_motion[t]

**Deliberately minimal, one axis tested at a time.** Pooled input (mean over 256 tokens), not the
ConvGRU's spatial state -- if this gate shows nothing, combining it with spatial state is the next
increment, not bundled in now where a null result would be ambiguous about which ingredient failed.
No embedding reconstruction term (this is the field-standard RSSM's OTHER job -- predicting the
observation -- omitted here to isolate whether the stochastic-latent/KL mechanism alone helps
Froude-relevant control; if this passes, a real build restores it). Continuous Gaussian latents,
not DreamerV3's categorical -- faster to implement, same qualitative KL mechanism.

Trained on HEXAPOD only (`beh12_c10f10t10_ego_flat`), matching the embodiment
`condition_confusion.py`'s ranking test actually evaluates -- not B1, which this session's other
kill-gates used. Short: ~2000 iterations, one GPU, the same budget every other kill-gate in this
arc used.

Saves `wm/runs/rssm_stage1_gate.pt` for `condition_confusion.py`'s new `rssm` scorer to load.
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

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

EMBODIMENT = "hexapod"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat")
CKPT = os.path.join(ROOT, "wm/runs/beh12_body_stopgrad/best.pt")   # body_stats/channels only
POOL_DIM = 1408
ACTION_DIM = 18
H_DIM = 256
Z_DIM = 32
BODY_DIM = 3
KL_WEIGHT = 1.0
SEQ_LEN = 8
ITERS = 3000
KL_WARMUP = 500   # linear 0 -> KL_WEIGHT over this many iterations, to avoid posterior collapse
                  # (the first run's own log: kl fell to ~0.003 by iter 200-1200, held-out Froude
                  # MSE ratio 1.120 -- worse than the mean baseline, the standard failure mode for
                  # an un-warmed-up KL term)
BATCH = 32
LR = 3e-4
HELD_OUT_FRAC = 0.2
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading beh12_body_stopgrad/best.pt only for body_stats/channels -- the RSSM itself is "
     "new, trained from scratch, hexapod only...")
ck = torch.load(CKPT, map_location=device, weights_only=False)
cfg = from_checkpoint(ck["config"])
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
    pooled = e.mean(1).to(device)                                     # [T, 1408]
    bm = torch.tensor((np.asarray(clip["body_motion"])[:, channels] - mean_s) / std_s,
                      dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32, device=device)
    n = min(len(pooled), len(bm), len(actions))
    clips.append({"pooled": pooled[:n], "bm": bm[:n], "actions": actions[:n],
                 "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]


def sample_chunks(clip_list, n, seq_len):
    P, A, B = [], [], []
    tries = 0
    while len(P) < n and tries < n * 20:
        tries += 1
        c = clip_list[rng.integers(0, len(clip_list))]
        T = len(c["bm"])
        if T < seq_len + 1:
            continue
        t0 = rng.integers(0, T - seq_len)
        P.append(c["pooled"][t0:t0 + seq_len])
        A.append(c["actions"][t0:t0 + seq_len])
        B.append(c["bm"][t0:t0 + seq_len])
    return torch.stack(P), torch.stack(A), torch.stack(B)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(Z_DIM + ACTION_DIM, H_DIM)
        self.post = nn.Sequential(nn.Linear(H_DIM + POOL_DIM, 256), nn.GELU(),
                                  nn.Linear(256, 2 * Z_DIM))
        self.prior = nn.Sequential(nn.Linear(H_DIM, 128), nn.GELU(), nn.Linear(128, 2 * Z_DIM))
        self.froude = nn.Sequential(nn.Linear(H_DIM + Z_DIM, 128), nn.GELU(),
                                    nn.Linear(128, BODY_DIM))

    def posterior(self, h, e):
        mu, logvar = self.post(torch.cat([h, e], -1)).chunk(2, -1)
        return mu, logvar

    def prior_dist(self, h):
        mu, logvar = self.prior(h).chunk(2, -1)
        return mu, logvar

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def step_posterior(self, h, z_prev, a_prev, e_t):
        h = self.cell(torch.cat([z_prev, a_prev], -1), h)
        mu_q, lv_q = self.posterior(h, e_t)
        mu_p, lv_p = self.prior_dist(h)
        z = self.sample(mu_q, lv_q)
        pred = self.froude(torch.cat([h, z], -1))
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(-1)
        return h, z, pred, kl

    def step_prior(self, h, z_prev, a_prev):
        """Imagination step -- no real frame, matches how condition_confusion.py's rollout works."""
        h = self.cell(torch.cat([z_prev, a_prev], -1), h)
        mu_p, lv_p = self.prior_dist(h)
        z = self.sample(mu_p, lv_p)
        pred = self.froude(torch.cat([h, z], -1))
        return h, z, pred

    def init_state(self, b):
        return (torch.zeros(b, H_DIM, device=device), torch.zeros(b, Z_DIM, device=device))


model = RSSM().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"training {ITERS} iterations, seq_len={SEQ_LEN}, batch={BATCH}, "
     f"{sum(p.numel() for p in model.parameters())} params...\n")
for it in range(ITERS):
    pooled, actions, bm = sample_chunks(train_clips, BATCH, SEQ_LEN)
    h, z = model.init_state(BATCH)
    froude_losses, kl_losses = [], []
    for t in range(SEQ_LEN):
        a_prev = actions[:, t]
        h, z, pred, kl = model.step_posterior(h, z, a_prev, pooled[:, t])
        froude_losses.append(F.mse_loss(pred, bm[:, t]))
        kl_losses.append(kl.mean())
    kl_weight_now = KL_WEIGHT * min(1.0, it / KL_WARMUP)
    loss = torch.stack(froude_losses).mean() + kl_weight_now * torch.stack(kl_losses).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 200 == 0 or it == 0:
        print(f"  iter {it + 1:5d}  froude {torch.stack(froude_losses).mean().item():.4f}  "
             f"kl {torch.stack(kl_losses).mean().item():.4f}  kl_weight {kl_weight_now:.3f}",
             flush=True)

print("\n" + "=" * 70)
print("HELD-OUT SANITY: teacher-forced Froude MSE, real posterior, vs a mean-prediction baseline")
print("=" * 70)
model.eval()
with torch.no_grad():
    pooled, actions, bm = sample_chunks(held_clips, 500, SEQ_LEN)
    h, z = model.init_state(500)
    preds, trues = [], []
    for t in range(SEQ_LEN):
        h, z, pred, _ = model.step_posterior(h, z, actions[:, t], pooled[:, t])
        preds.append(pred); trues.append(bm[:, t])
    preds, trues = torch.cat(preds), torch.cat(trues)
    mse = F.mse_loss(preds, trues).item()
    mean_mse = F.mse_loss(trues.mean(0, keepdim=True).expand_as(trues), trues).item()
print(f"model MSE {mse:.4f} vs mean-prediction MSE {mean_mse:.4f} "
     f"(ratio {mse / mean_mse:.3f}, <1.0 = better than predicting the mean)")

torch.save({"model": model.state_dict(), "channels": channels, "mean_s": mean_s, "std_s": std_s},
          os.path.join(ROOT, "wm/runs/rssm_stage1_gate.pt"))
print("\nsaved -> wm/runs/rssm_stage1_gate.pt")
