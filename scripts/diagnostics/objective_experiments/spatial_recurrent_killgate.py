"""The strict kill-gate: spatial detail actually PRESERVED through a genuinely RECURRENT state,
at the same time -- the one cell of the 2x2 that `sequence_context_killgate.py`'s two attempts
both missed.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/spatial_recurrent_killgate.py

**Why the two prior kill-gates do not settle this.** Checked directly: R0's GRU pooled (naive
mean over 256 tokens -> 1 vector) before its recurrent state ever saw the frame -- gap +0.036. The
"corrected" attempt swapped the naive mean for a LEARNED, action-conditioned attention-pool (8
queries over 256 tokens) -- still a pooling operation, still collapsing the full token grid to one
vector before the GRU integrates anything temporally -- gap +0.058. A third, discarded run kept the
full token grid through self-attention across all frames (genuinely spatial) but was not
recurrent at all -- gap +0.048. **None of the three tests spatial-preservation and real recurrence
together.** Two are recurrent-but-pooled, one is spatial-but-non-recurrent.

**This script is the missing cell.** V-JEPA2's 256 tokens are a 16x16 spatial grid (confirmed from
the cached model config: `image_size=256, patch_size=16` -> 16x16=256; `tubelet_size=2` is why
`encode_clip` duplicates each frame across the tubelet). A ConvGRU keeps the recurrent hidden
state itself as a **16x16xC spatial map**, updated by 3x3 convolutions every step -- not
flattened to a vector until the very last read-out, after all temporal integration is done. This
is the only design that satisfies all four things at once: frame-sequence with spatial detail
actually flowing through the state, a real per-step command sequence, real recurrent state, and a
delta-Froude target -- unchanged from every prior test in this arc for comparability.

Pre-registered kill-gate, same methodology and bar as every prior lever in this arc:
  1. SANITY CHECK FIRST: the stateless-FTM lever must reproduce +0.042 on this exact data.
  2. THE GATE: build h_{T-1} from REAL preceding frames/actions (genuine spatial history), vary
     ONLY the final action (real vs. mean/generic), decode predicted delta-Froude by
     global-average-pooling h_T (pooling happens ONLY here, at the very end), score cosine
     similarity against the true change.
        gap > 2 x 0.055 (> 0.110)  -> PASS: spatial-preserving recurrence restores
                                      action-sensitivity where every pooled or non-recurrent
                                      variant could not. The full sequence-FTM rebuild is
                                      warranted.
        gap <= 0.110               -> FAIL: the strict hypothesis this whole arc has chased is
                                      now genuinely tested, not confounded by pooling -- and if it
                                      still fails, the bottleneck is not "the FTM doesn't see
                                      enough context" in any form this arc can still cheaply test.
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
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

EMBODIMENT = "b1"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CKPT = os.path.join(ROOT, "wm/runs/beh12_state/teacher_state.pt")
SEQ_LEN = 8
GRID = 16                # 256 tokens = 16x16, confirmed from the cached V-JEPA2 config
TOK_DIM_IN = 1408
C_IN = 32                 # per-token channels after down-projection, before the ConvGRU
C_H = 64                  # ConvGRU hidden channels (stays spatial: [B, C_H, 16, 16])
ACTION_DIM = 12
BODY_DIM = 3
ITERS = 2000
BATCH = 16
LR = 3e-4
HELD_OUT_FRAC = 0.2
EVAL_SEQS = 800
EVAL_BATCH = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for its frozen encoder/state-head/body_stats (sanity check + "
     "target standardisation) -- the ConvGRU itself is new, trained from scratch...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
channels = [int(c) for c in cfg.body_channels]
mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)], dtype=torch.float32)
std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)], dtype=torch.float32)
state_model = StateHead(cfg, len(channels), tuple(s.split("=", 1)[0] for s in cfg.sources),
                        use_delta=getattr(cfg, "state_use_delta", True)).to(device).eval()
state_model.load_state_dict(ck["state"])

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(0)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

clips = []
for p in paths:
    clip = load(p, REGISTRY[EMBODIMENT])
    e = encode_clip(encoder, clip["frames"], 2).float()          # [T, 256, 1408]
    bm = (torch.tensor(np.asarray(clip["body_motion"])[:, channels], dtype=torch.float32) -
         mean_s) / std_s
    actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32)
    n = min(len(e), len(bm), len(actions))
    # GPU-resident, half precision (whole dataset ~1.1 GB) -- CPU-resident storage fragmented the
    # host allocator badly enough to OOM-kill the process in an earlier version of this test.
    clips.append({"e": e[:n].half().to(device), "bm": bm[:n].to(device),
                 "actions": actions[:n].to(device), "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]
action_mean = torch.cat([c["actions"] for c in train_clips]).mean(0)


def sample_chunks(clip_list, n, seq_len):
    E, A, B = [], [], []
    tries = 0
    while len(E) < n and tries < n * 20:
        tries += 1
        c = clip_list[rng.integers(0, len(clip_list))]
        T = len(c["bm"])
        if T < seq_len + 2:
            continue
        t0 = rng.integers(1, T - seq_len - 1)
        E.append(c["e"][t0:t0 + seq_len + 1].float())
        A.append(c["actions"][t0:t0 + seq_len])
        B.append(c["bm"][t0:t0 + seq_len + 1])
    return torch.stack(E), torch.stack(A), torch.stack(B)


class ConvGRUCell(nn.Module):
    """Standard ConvGRU: every gate is a 3x3 conv over the spatial map, not a linear layer over
    a pooled vector. `h` stays [B, C_H, GRID, GRID] for the model's entire lifetime."""

    def __init__(self):
        super().__init__()
        self.conv_zr = nn.Conv2d(C_IN + C_H, 2 * C_H, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(C_IN + C_H, C_H, kernel_size=3, padding=1)

    def forward(self, x, h):
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = zr.chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class SpatialRecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(TOK_DIM_IN, C_IN)
        self.action_to_map = nn.Linear(ACTION_DIM, C_IN)   # broadcast onto every spatial location
        self.cell = ConvGRUCell()
        self.decoder = nn.Sequential(nn.LayerNorm(C_H), nn.Linear(C_H, 64), nn.GELU(),
                                     nn.Linear(64, BODY_DIM))

    def step(self, h, tokens, action):
        """tokens: [B,256,1408]  action: [B,12]  h: [B,C_H,16,16] -> new h, predicted delta-Froude"""
        b = tokens.shape[0]
        x = self.proj_in(tokens).transpose(1, 2).reshape(b, C_IN, GRID, GRID)   # [B,C_IN,16,16]
        x = x + self.action_to_map(action).view(b, C_IN, 1, 1)                  # per-step command
        h = self.cell(x, h)
        pooled = h.mean(dim=(2, 3))            # pooling happens ONLY here, at final read-out
        return h, self.decoder(pooled)

    def init_hidden(self, b):
        return torch.zeros(b, C_H, GRID, GRID, device=device)


model = SpatialRecurrentModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"training {ITERS} iterations, seq_len={SEQ_LEN}, batch={BATCH}, grid={GRID}x{GRID}, "
     f"{sum(p.numel() for p in model.parameters())} params...\n")
for it in range(ITERS):
    e_seq, actions, bm = sample_chunks(train_clips, BATCH, SEQ_LEN)
    h = model.init_hidden(BATCH)
    losses = []
    for t in range(SEQ_LEN):
        h, pred_change = model.step(h, e_seq[:, t], actions[:, t])
        true_change = bm[:, t + 1] - bm[:, t]
        losses.append(F.mse_loss(pred_change, true_change))
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 200 == 0 or it == 0:
        print(f"  iter {it + 1:5d}  loss {loss.item():.4f}", flush=True)

print("\n" + "=" * 70)
print("SANITY CHECK: stateless-FTM lever reproduces on this exact data (+0.042 reference)?")
print("=" * 70)
with torch.no_grad():
    E_t, E_next, Bm_t, Bm_next = [], [], [], []
    for c in clips:
        e_full = c["e"]
        n = len(e_full) - 1
        for t in range(1, n, 4):
            E_t.append(e_full[t]); E_next.append(e_full[t + 1])
            Bm_t.append(c["bm"][t]); Bm_next.append(c["bm"][t + 1])
    e_t_cpu, e_next_cpu = torch.stack(E_t), torch.stack(E_next)
    bm_t_cpu, bm_next_cpu = torch.stack(Bm_t), torch.stack(Bm_next)
    sanity_real, sanity_mean = [], []
    SBATCH = 32
    real_z_list = []
    for s in range(0, len(e_t_cpu), SBATCH):
        sl = slice(s, s + SBATCH)
        real_z_list.append(itm(e_t_cpu[sl].float().to(device), e_next_cpu[sl].float().to(device)).cpu())
    real_z_cpu = torch.cat(real_z_list)
    mean_z_cpu = real_z_cpu.mean(0, keepdim=True)
    for s in range(0, len(e_t_cpu), SBATCH):
        sl = slice(s, s + SBATCH)
        e_t = e_t_cpu[sl].float().to(device)
        true_change = (bm_next_cpu[sl] - bm_t_cpu[sl]).to(device)
        real_z = real_z_cpu[sl].to(device)
        mean_z = mean_z_cpu.expand(e_t.shape[0], -1).to(device)
        for tag, zz in (("real", real_z), ("mean", mean_z)):
            nxt = ftm(e_t, zz, EMBODIMENT)
            delta = nxt - e_t
            fc = F.cosine_similarity(state_model(delta, zz, EMBODIMENT), true_change, dim=1)
            (sanity_real if tag == "real" else sanity_mean).append(fc.cpu())
    sanity_gap = torch.cat(sanity_real).median().item() - torch.cat(sanity_mean).median().item()
print(f"stateless-FTM lever, ALL 48 clips: gap {sanity_gap:+.3f}  (reference: +0.042)")
SANITY_REPRODUCES = abs(sanity_gap - 0.042) < 0.03
if SANITY_REPRODUCES:
    print("-> REPRODUCES: methodology checks out. The gate below is trustworthy.")
else:
    print(f"-> DOES NOT REPRODUCE (off by {abs(sanity_gap - 0.042):.3f}) -- do NOT trust the gate "
         "below, pass or fail, until this is fixed.")

print("\n" + "=" * 70)
print("SPATIAL-RECURRENT (ConvGRU) KILL-GATE: real vs. mean FINAL action, real spatial history + "
     "real command sequence for the preceding steps, held-out B1 clips")
print("=" * 70)
torch.cuda.empty_cache()
model.eval()
cos_real_list, cos_mean_list = [], []
with torch.no_grad():
    for s in range(0, EVAL_SEQS, EVAL_BATCH):
        b = min(EVAL_BATCH, EVAL_SEQS - s)
        e_seq, actions, bm = sample_chunks(held_clips, b, SEQ_LEN)
        mean_action = action_mean.unsqueeze(0).expand(b, -1)

        h = model.init_hidden(b)
        for t in range(SEQ_LEN - 1):
            h, _ = model.step(h, e_seq[:, t], actions[:, t])
        true_change = bm[:, SEQ_LEN] - bm[:, SEQ_LEN - 1]

        _, pred_real = model.step(h, e_seq[:, SEQ_LEN - 1], actions[:, SEQ_LEN - 1])
        _, pred_mean = model.step(h, e_seq[:, SEQ_LEN - 1], mean_action)
        cos_real_list.append(F.cosine_similarity(pred_real, true_change, dim=1).cpu())
        cos_mean_list.append(F.cosine_similarity(pred_mean, true_change, dim=1).cpu())
    cos_real = torch.cat(cos_real_list).median().item()
    cos_mean = torch.cat(cos_mean_list).median().item()

gap = cos_real - cos_mean
print(f"real final action:  median cos {cos_real:.3f}")
print(f"mean final action:  median cos {cos_mean:.3f}")
print(f"gap: {gap:+.3f}   (kill-gate bar: > {2 * 0.055:.3f}, i.e. 2x the stateless-FTM reference "
     "+0.055, same threshold every prior gate in this arc used)")

if not SANITY_REPRODUCES:
    print("\n-> NO VERDICT: sanity check did not reproduce; fix before reading this gap either way.")
elif gap > 2 * 0.055:
    print("\n-> PASS: spatial-preserving recurrence restores action-sensitivity where every "
         "pooled or non-recurrent variant could not. The full sequence-FTM rebuild is warranted.")
else:
    print("\n-> FAIL: the strict hypothesis (spatial + sequence + real recurrence + delta target, "
         "all four at once) is now genuinely tested, not confounded by pooling -- and it still "
         "fails. Kill here.")

torch.save({"model": model.state_dict(), "gap": gap, "cos_real": cos_real, "cos_mean": cos_mean,
           "sanity_gap": sanity_gap},
          os.path.join(ROOT, "wm/runs/beh12_state/spatial_recurrent_killgate.pt"))
print("\nsaved -> wm/runs/beh12_state/spatial_recurrent_killgate.pt")
