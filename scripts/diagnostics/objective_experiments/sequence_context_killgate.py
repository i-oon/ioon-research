"""TEST 2 kill-gate: does conditioning on frame-SEQUENCE (spatial tokens, not pooled) +
command-SEQUENCE raise the action-signal above the single-step stateless FTM's +0.042?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/sequence_context_killgate.py

Gated on `ground_truth_action_flatness.py`: ground truth showed real signal (perm p=0.002,
ridge-LOO R^2=0.204) for real-action-deviation -> real-Froude-deviation within a fixed behaviour,
and Froude drifts within a clip rather than sitting at a plateau (favours delta-Froude as target).
This cleared the "worth building" bar. F198's addenda already tested TEMPORAL STATE in its
crudest form -- a GRU on POOLED features, 2-frame proxy, no command sequence -- and it FAILED
(+0.036, below even the +0.055 reference). This is a genuinely different, stronger hypothesis:
Yu/WMP-style conditioning keeps the SPATIAL token grid (not collapsed to one pooled vector per
frame) and a full per-step command SEQUENCE (not one summary action), reasoning that gait-phase
context needed to disambiguate one-action-many-outcomes lives in spatial detail across a short
history, not in a pooled recurrent summary.

Architecture (T0 -- a kill-gate, not the full multi-day rebuild):
  - K=4 consecutive real frames, FULL token grid each (256 tokens x 1408-d, from the frozen
    V-JEPA2 encoder already used everywhere else in this project).
  - Linear down-projection 1408 -> 256 per token (keeps attention compute bounded: K*256=1024
    tokens at d=256, not d=1408).
  - Per-frame temporal position embedding (learned, one of K slots) added to every token of that
    frame -- required because `wm/models/blocks.py`'s attention has NO positional embedding, so
    without this the model cannot tell frame order apart (F198's addenda relied on exactly this
    permutation-equivariance for the cheap 2-frame proxy; here it would be a liability that must
    be corrected, not exploited).
  - Per-frame ACTION embedding (linear(action_t) -> 256) added to every token of that same frame
    -- this is the command-SEQUENCE: each of the K frames carries its OWN associated action, not
    one action for the whole window.
  - 3 standard transformer self-attention layers over the concatenated 1024 tokens.
  - Decode: mean-pool the LAST frame's (K-1) contextualized tokens -> small MLP -> predicted
    delta-Froude for the step from frame K-1 to frame K (real target: bm[K]-bm[K-1]).
  - Trained via MSE, teacher-forced on real recorded frames/actions/body-motion (no rollout,
    matching the R0 GRU's discipline: this tests "does history+sequence predict Froude", not
    imagination).

Pre-registered kill-gate (same threshold discipline as the R0 GRU kill-gate, for direct
comparability):
  1. SANITY CHECK FIRST: the stateless-FTM lever must reproduce on this exact data (+0.042
     reference) before trusting anything below. If it doesn't reproduce within 0.03, STOP --
     something differs in this pipeline, do not read the gate either way.
  2. THE GATE: hold frames/actions at slots 0..K-2 REAL and fixed (genuine preceding history +
     its genuine command sequence), vary ONLY the FINAL action (slot K-1): real vs. mean/generic.
     Decode predicted delta-Froude, score cosine similarity against the true change (bm[K]-bm[K-1]
     from the REAL final action's trajectory; a fixed target across both conditions, exactly as
     every prior lever in this arc has done).
        gap = median_cos(real final action) - median_cos(mean final action)
        gap > 2 x 0.055 (> 0.110)  -> PASS: history+sequence (kept spatial) restores
                                      action-sensitivity beyond the stateless baseline and beyond
                                      the pooled-GRU's attempt. Worth scoping the full sequence
                                      FTM rebuild (train end-to-end, multi-body, replace the FTM).
        gap <= 0.110               -> FAIL: kill here. If this also fails, the action-insensitivity
                                      is not fixable by any form of added context this arc can
                                      still cheaply test, and the real signal
                                      `ground_truth_action_flatness.py` found in the ground truth
                                      is being lost specifically at model TRAINING (capacity/
                                      optimization), not at the input representation.
  A PASS is necessary, not sufficient -- MC-check, planning, and cross-embodiment remain untested
  even if this gate clears. This script only answers: does spatial history + command sequence
  move the single-step action-lever.
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
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")   # same 48-clip set as the R0 GRU
CKPT = os.path.join(ROOT, "wm/runs/beh12_state/teacher_state.pt")
K = 4                    # frame-history window (>=2; K-1 real history steps + 1 final step tested)
TOK_DIM_IN = 1408
D = 256                  # working transformer dim after down-projection
N_LAYERS = 3
N_HEADS = 4
ACTION_DIM = 12
BODY_DIM = 3
ITERS = 2000             # same short-train budget as the R0 GRU, for direct comparability
BATCH = 16               # smaller than R0's 32: K*256=1024 tokens/sample is heavier per-sample
LR = 3e-4
HELD_OUT_FRAC = 0.2
EVAL_SEQS = 500           # smaller than R0's 2000: full spatial tokens cost much more memory/sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for its frozen encoder/state-head/body_stats (sanity check + "
     "target standardisation) -- the sequence transformer itself is new, trained from scratch...")
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
    e = encode_clip(encoder, clip["frames"], 2).float()          # [T, 256, 1408], full token grid
    bm = (torch.tensor(np.asarray(clip["body_motion"])[:, channels], dtype=torch.float32) -
         mean_s) / std_s
    actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32)
    n = min(len(e), len(bm), len(actions))
    clips.append({"e": e[:n].half().cpu(), "bm": bm[:n], "actions": actions[:n],
                 "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]
action_mean = torch.cat([c["actions"] for c in train_clips]).mean(0)


def sample_windows(clip_list, n, k):
    """n random windows: k frames of tokens + k actions (frame i's action drives i -> i+1),
    plus bm at frame k-1 and frame k (the step being predicted)."""
    E, A, BM0, BM1 = [], [], [], []
    tries = 0
    while len(E) < n and tries < n * 20:
        tries += 1
        c = clip_list[rng.integers(0, len(clip_list))]
        T = len(c["bm"])
        if T < k + 2:
            continue
        t0 = rng.integers(1, T - k - 1)
        E.append(c["e"][t0:t0 + k].float())              # [k, 256, 1408]
        A.append(c["actions"][t0:t0 + k])                # [k, 12] -- command SEQUENCE
        BM0.append(c["bm"][t0 + k - 1])
        BM1.append(c["bm"][t0 + k])
    return torch.stack(E), torch.stack(A), torch.stack(BM0), torch.stack(BM1)


class SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(TOK_DIM_IN, D)
        self.time_embed = nn.Parameter(torch.randn(K, 1, D) * 0.02)
        self.action_embed = nn.Linear(ACTION_DIM, D)
        layer = nn.TransformerEncoderLayer(d_model=D, nhead=N_HEADS, dim_feedforward=4 * D,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.decoder = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 128), nn.GELU(),
                                     nn.Linear(128, BODY_DIM))

    def forward(self, e_seq, a_seq):
        """e_seq: [B,K,256,1408]  a_seq: [B,K,12]  ->  predicted delta-Froude [B,3]"""
        b, k, n_tok, _ = e_seq.shape
        x = self.proj_in(e_seq)                                  # [B,K,256,D]
        x = x + self.time_embed[:k].unsqueeze(0)                 # frame-order signal
        x = x + self.action_embed(a_seq).unsqueeze(2)            # per-frame command SEQUENCE
        x = x.reshape(b, k * n_tok, D)
        x = self.encoder(x)
        last_frame = x[:, (k - 1) * n_tok:k * n_tok, :].mean(1)  # pool the FINAL frame's tokens
        return self.decoder(last_frame)


model = SeqTransformer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"training {ITERS} iterations, K={K} frames, batch={BATCH}, "
     f"{sum(p.numel() for p in model.parameters())} params...\n")
for it in range(ITERS):
    e_seq, a_seq, bm0, bm1 = sample_windows(train_clips, BATCH, K)
    e_seq, a_seq = e_seq.to(device), a_seq.to(device)
    true_change = (bm1 - bm0).to(device)
    pred = model(e_seq, a_seq)
    loss = F.mse_loss(pred, true_change)
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
print(f"T0 KILL-GATE: real vs. mean FINAL action, real frame-history + real command sequence for "
     f"the preceding {K - 1} steps, held-out B1 clips")
print("=" * 70)
model.eval()
with torch.no_grad():
    e_seq, a_seq, bm0, bm1 = sample_windows(held_clips, EVAL_SEQS, K)
    e_seq, a_seq = e_seq.to(device), a_seq.to(device)
    true_change = (bm1 - bm0).to(device)
    mean_action = action_mean.to(device).unsqueeze(0).expand(EVAL_SEQS, -1)

    a_seq_real = a_seq.clone()
    a_seq_mean_final = a_seq.clone()
    a_seq_mean_final[:, -1] = mean_action

    pred_real = model(e_seq, a_seq_real)
    pred_mean = model(e_seq, a_seq_mean_final)
    cos_real = F.cosine_similarity(pred_real, true_change, dim=1).median().item()
    cos_mean = F.cosine_similarity(pred_mean, true_change, dim=1).median().item()

gap = cos_real - cos_mean
print(f"real final action:  median cos {cos_real:.3f}")
print(f"mean final action:  median cos {cos_mean:.3f}")
print(f"gap: {gap:+.3f}   (kill-gate bar: > {2 * 0.055:.3f}, i.e. 2x the stateless-FTM reference "
     "+0.055, same threshold the R0 GRU used)")

if not SANITY_REPRODUCES:
    print("\n-> NO VERDICT: sanity check did not reproduce; fix before reading this gap either way.")
elif gap > 2 * 0.055:
    print("\n-> PASS: spatial history + command sequence clears the kill-gate, and does so where "
         "the R0 pooled-GRU attempt (+0.036) failed. Scoping the full sequence-FTM rebuild "
         "(end-to-end, multi-body, replacing the single-step FTM) is now warranted.")
else:
    print("\n-> FAIL: gap does not clear the kill-gate. Spatial + sequence architecture also fails "
         "to move the action-lever, despite ground truth confirming real signal exists -- the loss "
         "is happening at model TRAINING/capacity, not at the input representation. Kill here; do "
         "not scope the full sequence-FTM rebuild on this basis.")

torch.save({"model": model.state_dict(), "gap": gap, "cos_real": cos_real, "cos_mean": cos_mean,
           "sanity_gap": sanity_gap},
          os.path.join(ROOT, "wm/runs/beh12_state/seq_t0_killgate.pt"))
print("\nsaved -> wm/runs/beh12_state/seq_t0_killgate.pt")
