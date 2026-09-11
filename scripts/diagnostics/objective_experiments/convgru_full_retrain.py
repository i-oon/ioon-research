"""The full-budget retrain of the ConvGRU kill-gate (F180), gated to decide whether the recurrent-
architecture line is worth pursuing further -- not a rebuild of the world model's own FTM.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/convgru_full_retrain.py \\
        --iters 20000 --out wm/runs/convgru_full/convgru_full.pt

**Why this exists.** The 2000-iteration probe (`spatial_recurrent_killgate.py`) is the only cell in
the recurrent-architecture 2x2 (pooled/spatial x non-recurrent/recurrent) with real positive
evidence: gap +0.069, 1.6x the stateless-FTM reference, best of four variants -- still short of the
pre-registered 0.110 bar, but on a probe far short of a real training budget. Every other
architecture and objective fix tried in this arc (R0, the two non-spatial attempts, the pooled
full-stochastic RSSM, the ActSWM hinge rebuild, counterfactual targets) has already been run at
real/exact strength and failed. This is the one lead left with headroom to test at real strength
before concluding the whole recurrent-architecture line is closed.

**Deliberately NOT the "real build" F180 scoped** (a data-pipeline change, decoders off `(h_t,z_t)`
for every existing target, wired into the actual world model's FTM). That commitment is only
warranted if THIS gate clears the bar -- the same two-stage discipline every other gate in this
arc used. This script trains the exact `SpatialRecurrentModel` the probe validated
(`wm/models/convgru_ftm.py`, extracted unchanged from the probe so architecture cannot drift),
extended to full clip sets and a real iteration budget, then re-runs the identical real-vs-mean-
action gate.

**Trains on hexapod only by default, matching every other probe in this arc.** R0 and the original
ConvGRU kill-gate trained on B1 only; the full RSSM trained on hexapod only -- none of them
jointly trained both bodies, deliberately, so "does recurrence help" is never conflated with "does
joint cross-embodiment training help," a different and untested question. B1 is left out entirely
here; if hexapod clears the gate, testing transfer to B1 is a separate follow-up (adapting the
hexapod-trained ConvGRU the way `wm.adapt` adapts the stateless FTM today), not something to fold
into this run. Pass `--embodiments hexapod b1` only if the joint-training question is explicitly
what is being tested.

**Fully additive and reversible.** New model file, new script, new checkpoint directory
(`wm/runs/convgru_full/`) -- nothing here edits `wm/train.py`, `wm/models/ftm.py`,
`wm/policy/planner.py`, or any live checkpoint. If the gate fails at full budget, delete
`wm/models/convgru_ftm.py`, this script, and `wm/runs/convgru_full/`; nothing else in the pipeline
is touched or needs to be.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.convgru_ftm import SpatialRecurrentModel  # noqa: E402

GRID = 16                 # 256 V-JEPA2 tokens = 16x16 (confirmed in the probe this extends)
BODY_CHANNELS = [0, 1, 2]  # forward/lateral/yaw Froude -- this project's standing convention
BODY_DIM = 3
STATELESS_REFERENCE_GAP = 0.055   # the same reference every gate in this arc is read against
KILL_GATE_BAR = 2 * STATELESS_REFERENCE_GAP  # 0.110

ALL_DATA_DIRS = {
    "hexapod": os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat"),
    "b1": os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat"),
}


def load_clips(name, data_dir, encoder, device, held_out_frac, seed):
    """One record per clip, GPU-resident half precision -- CPU-resident storage fragmented the
    host allocator badly enough to OOM-kill the process in the probe this extends (same fix kept)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"no clips found for {name} in {data_dir}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_held = max(1, int(len(paths) * held_out_frac))
    held_paths = {paths[i] for i in order[:n_held]}

    reg = REGISTRY[name]
    channels = BODY_CHANNELS
    clips = []
    for p in paths:
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float()
        bm_full = np.asarray(clip["body_motion"])
        bm = torch.tensor(bm_full[:, channels], dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32)
        n = min(len(e), len(bm), len(actions))
        clips.append({"e": e[:n].half().to(device), "bm": bm[:n].to(device),
                      "actions": actions[:n].to(device), "held_out": p in held_paths})
    print(f"  {name}: {len(paths)} clips, {n_held} held out, action_dim={reg.action_dim}")
    return clips


def standardize(clips_by_embodiment):
    """Per-embodiment body-motion standardisation, fit on TRAIN clips only, applied everywhere."""
    stats = {}
    for name, clips in clips_by_embodiment.items():
        train = torch.cat([c["bm"] for c in clips if not c["held_out"]])
        mean, std = train.mean(0), train.std(0).clamp_min(1e-6)
        stats[name] = (mean, std)
        for c in clips:
            c["bm"] = (c["bm"] - mean) / std
    return stats


def sample_chunks(clips, rng, n, seq_len):
    E, A, Bm = [], [], []
    tries = 0
    while len(E) < n and tries < n * 20:
        tries += 1
        c = clips[rng.integers(0, len(clips))]
        T = len(c["bm"])
        if T < seq_len + 2:
            continue
        t0 = rng.integers(1, T - seq_len - 1)
        E.append(c["e"][t0:t0 + seq_len + 1].float())
        A.append(c["actions"][t0:t0 + seq_len])
        Bm.append(c["bm"][t0:t0 + seq_len + 1])
    return torch.stack(E), torch.stack(A), torch.stack(Bm)


def run_gate(model, clips, embodiment, device, rng, seq_len, eval_seqs, eval_batch, action_mean):
    """Real-vs-mean FINAL action, real spatial+recurrent history for the preceding steps -- the
    exact methodology every gate in this arc uses. Read per body, never pooled (F141's locked rule)."""
    model.eval()
    cos_real_list, cos_mean_list = [], []
    with torch.no_grad():
        for s in range(0, eval_seqs, eval_batch):
            b = min(eval_batch, eval_seqs - s)
            e_seq, actions, bm = sample_chunks(clips, rng, b, seq_len)
            mean_action = action_mean.unsqueeze(0).expand(b, -1)
            h = model.init_hidden(b, device)
            for t in range(seq_len - 1):
                h, _ = model.step(h, e_seq[:, t], actions[:, t], embodiment)
            true_change = bm[:, seq_len] - bm[:, seq_len - 1]
            _, pred_real = model.step(h, e_seq[:, seq_len - 1], actions[:, seq_len - 1], embodiment)
            _, pred_mean = model.step(h, e_seq[:, seq_len - 1], mean_action, embodiment)
            cos_real_list.append(F.cosine_similarity(pred_real, true_change, dim=1).cpu())
            cos_mean_list.append(F.cosine_similarity(pred_mean, true_change, dim=1).cpu())
    cos_real = torch.cat(cos_real_list).median().item()
    cos_mean = torch.cat(cos_mean_list).median().item()
    return cos_real, cos_mean, cos_real - cos_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000,
                    help="full-budget training iterations -- 10x the probe's 2000")
    ap.add_argument("--embodiments", nargs="+", default=["hexapod"], choices=list(ALL_DATA_DIRS),
                    help="**default: hexapod only.** Every recurrent-architecture probe in this "
                         "arc (R0, the ConvGRU kill-gate, the full RSSM) trained on ONE body, "
                         "deliberately not conflating 'does recurrence help' with cross-embodiment "
                         "joint training -- a different, untested question. Pass e.g. "
                         "`--embodiments hexapod b1` only if that joint-training question is "
                         "explicitly what is being tested.")
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--held_out_frac", type=float, default=0.2)
    ap.add_argument("--eval_seqs", type=int, default=800)
    ap.add_argument("--eval_batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="wm/runs/convgru_full/convgru_full.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"loading V-JEPA2 encoder and clips for {args.embodiments}...")
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips_by_embodiment = {
        name: load_clips(name, ALL_DATA_DIRS[name], encoder, device, args.held_out_frac, args.seed)
        for name in args.embodiments
    }
    del encoder
    torch.cuda.empty_cache()

    standardize(clips_by_embodiment)
    action_dims = {name: REGISTRY[name].action_dim for name in args.embodiments}
    action_means = {
        name: torch.cat([c["actions"] for c in clips if not c["held_out"]]).mean(0)
        for name, clips in clips_by_embodiment.items()
    }

    model = SpatialRecurrentModel(action_dims, grid=GRID, body_dim=BODY_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntraining {args.iters} iterations, seq_len={args.seq_len}, batch={args.batch}, "
         f"grid={GRID}x{GRID}, {n_params} params, embodiments={args.embodiments}\n")

    names = args.embodiments
    for it in range(args.iters):
        name = names[it % len(names)]   # alternate embodiments every step, never pooled in one batch
        train_clips = [c for c in clips_by_embodiment[name] if not c["held_out"]]
        e_seq, actions, bm = sample_chunks(train_clips, rng, args.batch, args.seq_len)
        h = model.init_hidden(args.batch, device)
        losses = []
        for t in range(args.seq_len):
            h, pred_change = model.step(h, e_seq[:, t], actions[:, t], name)
            true_change = bm[:, t + 1] - bm[:, t]
            losses.append(F.mse_loss(pred_change, true_change))
        loss = torch.stack(losses).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 500 == 0 or it == 0:
            print(f"  iter {it + 1:6d}  [{name:8s}]  loss {loss.item():.4f}", flush=True)

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "action_dims": action_dims,
               "args": vars(args), "embodiments": args.embodiments}, out_path)
    print(f"\n-> {args.out}")

    print("\n" + "=" * 78)
    print(f"KILL-GATE, full budget: read per body, never pooled. Bar: gap > {KILL_GATE_BAR:.3f} "
         f"(2x the stateless-FTM reference +{STATELESS_REFERENCE_GAP:.3f})")
    print("=" * 78)
    any_pass = False
    for name, clips in clips_by_embodiment.items():
        held_clips = [c for c in clips if c["held_out"]]
        cos_real, cos_mean, gap = run_gate(model, held_clips, name, device, rng, args.seq_len,
                                           args.eval_seqs, args.eval_batch, action_means[name])
        verdict = "PASS" if gap > KILL_GATE_BAR else "FAIL"
        any_pass = any_pass or verdict == "PASS"
        print(f"  {name:8s}  real {cos_real:.3f}  mean {cos_mean:.3f}  gap {gap:+.3f}  {verdict}")

    print("\nRead per body -- a pooled average across embodiments is not a valid summary here "
         "(F141's locked rule: insect and B1 have shown different sensitivity curves throughout "
         "this arc). If FAIL on both: the full-sequence-FTM rebuild F180 scoped is not warranted, "
         "matching every other objective/architecture fix already tried and failed in this arc. "
         "If PASS on either: real signal that a from-scratch full FTM rebuild around this "
         "architecture might be worth the multi-day commitment -- re-derive the decision from the "
         "per-body numbers above, not from this line.")


if __name__ == "__main__":
    main()
