"""TEST 2 kill-gate: does a per-frame SPATIAL (not naive-pooled) readout, fed through the SAME
recurrent architecture that already carries a command sequence and predicts delta-Froude, raise
the action-signal above the single-step stateless FTM's +0.042?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/sequence_context_killgate.py

Gated on `ground_truth_action_flatness.py`: ground truth showed real signal (perm p=0.002,
ridge-LOO R^2=0.204) for real-action-deviation -> real-Froude-deviation within a fixed behaviour.
This cleared the "worth building" bar for a context-based fix.

**This must differ from the R0 GRU kill-gate (F180 addenda) on the ONE axis that matters, and
change nothing else, or a null result is uninterpretable.** Checking R0 against the four
properties this line of attack needs (frame-sequence not pooled, command-sequence, real recurrent
state, delta-Froude target): R0 already had three of the four -- `h_t =
GRUCell([pool(e_t-1), action_t-1], h_t-1)` is genuine recurrent state, trained with a REAL
per-step action sequence (teacher-forced over 8 real steps), predicting Froude CHANGE
(`bm[t+1]-bm[t]`, not absolute Froude). The ONLY axis it did not have was spatial: `pool(e_t-1)`
is a naive `mean(dim=1)` over the 256 tokens before the GRU ever sees them. R0 FAILED (+0.036,
below even the +0.055 reference).

A first attempt at this script replaced R0's GRU with a non-recurrent windowed self-attention
Transformer over concatenated frames -- that also drops "real recurrent state", so a null result
from it would be uninterpretable (confounded between "spatial didn't help" and "non-recurrent
architecture doesn't work here"). Discarded before being trusted as a result.

**Corrected design: keep R0's GRU, real per-step action sequence, and delta-Froude target
UNCHANGED. Change only the per-step input.** Instead of `pool(e_t-1) = mean(dim=1)`, use a small
learned, ACTION-CONDITIONED attention-pool: 8 learned query vectors, each additively conditioned
on that step's real action, cross-attend into the frame's full 256-token grid (1408-d, down-
projected to D=64 per token) to produce the GRU's per-step input. This is the minimal change that
stops discarding spatial detail before the recurrence sees it, without changing the recurrent
architecture, the action-sequence mechanism, or the target -- so a null result here isolates the
spatial-pooling axis specifically, the same way R0 isolated recurrence.

Pre-registered kill-gate (identical methodology and threshold to R0, for direct comparability):
  1. SANITY CHECK FIRST: the stateless-FTM lever must reproduce on this exact data (+0.042
     reference) before trusting anything below. If it doesn't reproduce within 0.03, STOP.
  2. THE GATE: build h_{T-1} from REAL preceding frames/actions (genuine history), then vary ONLY
     the FINAL action -- real vs. mean/generic -- decode predicted delta-Froude, score cosine
     similarity against the true change (bm[T]-bm[T-1], fixed across both conditions).
        gap = median_cos(real final action) - median_cos(mean final action)
        gap > 2 x 0.055 (> 0.110)  -> PASS: spatial detail, isolated from every other axis,
                                      restores action-sensitivity where R0's pooled version could
                                      not. Worth scoping the full sequence-FTM rebuild.
        gap <= 0.110               -> FAIL: kill here. Combined with R0, this rules out BOTH
                                      pooling choices under an otherwise-identical recurrent +
                                      sequence + delta-target design -- a materially stronger
                                      negative than R0 alone, since the one axis anyone would
                                      still suspect (spatial detail) is now cleanly isolated.
  A PASS is necessary, not sufficient -- MC-check, planning, and cross-embodiment remain untested
  even if this gate clears.
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
SEQ_LEN = 8              # same chunk length as R0
TOK_DIM_IN = 1408
D = 64                   # per-token working dim for the attention-pool (kept small -- deliberately
                         # cheap; this is a kill-gate, not the full rebuild)
N_QUERIES = 8            # learned, action-conditioned readout queries per frame
HIDDEN = 256             # GRU hidden size, matches R0
POOL_DIM_OUT = N_QUERIES * D   # flattened attention-pool output fed to the GRU, replacing R0's 1408-d mean-pool
ACTION_DIM = 12
BODY_DIM = 3
ITERS = 2000             # same short-train budget as R0
BATCH = 16               # smaller than R0's 32: per-step attention over 256 tokens is heavier than a mean
LR = 3e-4
HELD_OUT_FRAC = 0.2
EVAL_SEQS = 800

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print("loading teacher_state.pt for its frozen encoder/state-head/body_stats (sanity check + "
     "target standardisation) -- the recurrent model itself is new, trained from scratch...")
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
    # kept GPU-resident, half precision (whole dataset is ~1.1 GB, trivial for an 11 GB card) --
    # repeatedly allocating/freeing ~200 MB CPU float32 tensors every training iteration (the
    # original CPU-resident version) fragmented the host allocator badly enough to OOM-kill the
    # process at ~25 GB RSS well before any real leak; GPU's caching allocator does not have
    # that problem, and this also removes a PCIe transfer every iteration.
    clips.append({"e": e[:n].half().to(device), "bm": bm[:n].to(device),
                 "actions": actions[:n].to(device), "held_out": p in held_paths})

train_clips = [c for c in clips if not c["held_out"]]
held_clips = [c for c in clips if c["held_out"]]
action_mean = torch.cat([c["actions"] for c in train_clips]).mean(0)


def sample_chunks(clip_list, n, seq_len):
    """n random (tokens, actions, bm) chunks of length seq_len+1 transitions -- same convention
    as R0's sample_chunks, except tokens keep the full [seq_len, 256, 1408] grid instead of a
    pre-pooled [seq_len, 1408] vector."""
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


class SpatialReader(nn.Module):
    """Action-conditioned learned attention-pool over one frame's full token grid. Replaces
    R0's `mean(dim=1)` -- the ONE thing this test changes relative to R0."""

    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(TOK_DIM_IN, D)
        self.queries = nn.Parameter(torch.randn(N_QUERIES, D) * 0.02)
        self.action_to_query = nn.Linear(ACTION_DIM, D)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)

    def forward(self, tokens, action):
        """tokens: [B,256,1408]  action: [B,12]  ->  [B, N_QUERIES*D]"""
        b = tokens.shape[0]
        x = self.proj_in(tokens)                                        # [B,256,D]
        q = self.queries.unsqueeze(0).expand(b, -1, -1) + \
            self.action_to_query(action).unsqueeze(1)                   # [B,N_QUERIES,D], action-conditioned
        readout, _ = self.attn(q, x, x)                                 # [B,N_QUERIES,D]
        return readout.reshape(b, -1)                                   # [B, N_QUERIES*D]


class GRUWorldModelSpatial(nn.Module):
    """Identical to R0's GRUWorldModel except the per-step input is `SpatialReader(tokens,
    action)` instead of a pre-pooled mean vector. GRU cell, decoder, and training loop unchanged."""

    def __init__(self):
        super().__init__()
        self.reader = SpatialReader()
        self.cell = nn.GRUCell(POOL_DIM_OUT + ACTION_DIM, HIDDEN)
        self.decoder = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 128), nn.GELU(),
                                     nn.Linear(128, BODY_DIM))

    def step(self, h, tokens_prev, action_prev):
        r = self.reader(tokens_prev, action_prev)
        h = self.cell(torch.cat([r, action_prev], -1), h)
        return h, self.decoder(h)


model = GRUWorldModelSpatial().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"training {ITERS} iterations, seq_len={SEQ_LEN}, batch={BATCH}, "
     f"{sum(p.numel() for p in model.parameters())} params...\n")
for it in range(ITERS):
    e_seq, actions, bm = sample_chunks(train_clips, BATCH, SEQ_LEN)
    e_seq, actions, bm = e_seq.to(device), actions.to(device), bm.to(device)
    h = torch.zeros(BATCH, HIDDEN, device=device)
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
print("SPATIAL-GRU KILL-GATE: real vs. mean FINAL action, real frame-history + real command "
     "sequence for the preceding steps, held-out B1 clips")
print("=" * 70)
torch.cuda.empty_cache()
EVAL_BATCH = 32   # the whole EVAL_SEQS set at once (was ~10 GB for [800,9,256,1408] float32) OOM'd
                  # the 2080 Ti after training left it fragmented; batch it like the sanity check
model.eval()
cos_real_list, cos_mean_list = [], []
with torch.no_grad():
    for s in range(0, EVAL_SEQS, EVAL_BATCH):
        b = min(EVAL_BATCH, EVAL_SEQS - s)
        e_seq, actions, bm = sample_chunks(held_clips, b, SEQ_LEN)
        mean_action = action_mean.unsqueeze(0).expand(b, -1)

        # build REAL context h_{T-1} using the real preceding steps -- same as R0
        h = torch.zeros(b, HIDDEN, device=device)
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
     "+0.055, same threshold R0 used)")

if not SANITY_REPRODUCES:
    print("\n-> NO VERDICT: sanity check did not reproduce; fix before reading this gap either way.")
elif gap > 2 * 0.055:
    print("\n-> PASS: spatial detail, isolated from every other axis (recurrence, command "
         "sequence, delta target all unchanged from R0), restores action-sensitivity where R0's "
         "pooled version could not. Scoping the full sequence-FTM rebuild is now warranted.")
else:
    print("\n-> FAIL: gap does not clear the kill-gate. Combined with R0, this rules out BOTH "
         "pooling choices under an otherwise-identical recurrent + sequence + delta-target "
         "design -- the loss is happening at model TRAINING/capacity or somewhere this arc has "
         "not yet isolated, not at pooled-vs-spatial input representation. Kill here; do not "
         "scope the full sequence-FTM rebuild on this basis.")

torch.save({"model": model.state_dict(), "gap": gap, "cos_real": cos_real, "cos_mean": cos_mean,
           "sanity_gap": sanity_gap},
          os.path.join(ROOT, "wm/runs/beh12_state/seq_spatial_gru_killgate.pt"))
print("\nsaved -> wm/runs/beh12_state/seq_spatial_gru_killgate.pt")
