"""Does the ALREADY-TRAINED z-only state head (F177, `state_use_delta=false`,
`wm/runs/beh12_state_zonly/teacher_zonly.pt`) clear the action-lever bar this session's whole F180
arc has used -- a measurement F177's own real-retrain evaluation (`condition_confusion.py` ranking)
never ran? No training here; this is a free measurement on an existing checkpoint.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/zonly_action_lever_check.py

Two things measured, both real-z vs mean-z, Froude-space cosine, same methodology as every prior
lever in this arc (+0.042 reference on `teacher_state.pt`, bar 0.110 = 2x the +0.055 reference):

  1. z = ITM(e_t, e_next)   -- matches the established reference exactly, apples-to-apples
  2. z = proj(action)       -- the actual control-relevant path (proj_action_ceiling_check.py
                               showed 80% of the ITM-z's delta-Froude signal survives this
                               substitution on the offline probe; this checks the trained head)

Plus a per-channel breakdown using a DIFFERENT, self-contained metric -- cosine similarity is not
naturally decomposable per channel, so forcing a per-channel number onto the 0.110 bar (calibrated
for the 3-D cosine metric) would be invalid. Per channel: the fraction of held-out samples where
the REAL z's prediction is closer to the true value than the MEAN z's prediction is (a paired
win-rate against a 50% chance baseline) -- this is what actually tests the pre-registered
per-channel expectation (lateral/yaw should win more than forward).
"""
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

from teacher_student_insect import load_teacher  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402

EMBODIMENT = "b1"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CKPT = os.path.join(ROOT, "wm/runs/beh12_state_zonly/teacher_zonly.pt")
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
REFERENCE_GAP = 0.042
BAR = 2 * 0.055

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print(f"loading {CKPT} -- the ALREADY-TRAINED z-only (state_use_delta=false) state head, jointly "
     "trained on hexapod+B1 (F177)...")
ck, cfg, itm, ftm, md, proj = load_teacher(CKPT, device)
print(f"cfg.state_use_delta = {getattr(cfg, 'state_use_delta', 'MISSING')}")
channels = [int(c) for c in cfg.body_channels]
mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)], dtype=torch.float32, device=device)
std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)], dtype=torch.float32, device=device)
state_model = StateHead(cfg, len(channels), tuple(s.split("=", 1)[0] for s in cfg.sources),
                        use_delta=getattr(cfg, "state_use_delta", True)).to(device).eval()
state_model.load_state_dict(ck["state"])

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
print(f"{len(paths)} B1 clips (full set, matching every prior sanity check in this arc)\n")

E_t, E_next, Bm_t, Bm_next, Actions = [], [], [], [], []
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
        Actions.append(actions[t])

e_t_cpu = torch.stack(E_t).cpu()
e_next_cpu = torch.stack(E_next).cpu()
bm_t = torch.stack(Bm_t)
bm_next = torch.stack(Bm_next)
actions_all = torch.stack(Actions)
true_change = bm_next - bm_t
print(f"{len(e_t_cpu)} transitions\n")
torch.cuda.empty_cache()

SBATCH = 16


def batched_itm(a, b):
    out = []
    with torch.no_grad():
        for s in range(0, len(a), SBATCH):
            sl = slice(s, s + SBATCH)
            out.append(itm(a[sl].float().to(device), b[sl].float().to(device)))
    return torch.cat(out)


def predict(e_t_all, zz):
    out = []
    with torch.no_grad():
        for s in range(0, len(e_t_all), SBATCH):
            sl = slice(s, s + SBATCH)
            e_t_b = e_t_all[sl].float().to(device)
            z_b = zz[sl]
            nxt = ftm(e_t_b, z_b, EMBODIMENT)
            delta = nxt - e_t_b
            out.append(state_model(delta, z_b, EMBODIMENT))
    return torch.cat(out)


def report(tag, real_z, mean_z_single):
    mean_z = mean_z_single.unsqueeze(0).expand(len(real_z), -1)
    pred_real = predict(e_t_cpu, real_z)
    pred_mean = predict(e_t_cpu, mean_z)
    cos_real = F.cosine_similarity(pred_real, true_change, dim=1)
    cos_mean = F.cosine_similarity(pred_mean, true_change, dim=1)
    gap = cos_real.median().item() - cos_mean.median().item()
    print("=" * 70)
    print(f"{tag}: real z median cos {cos_real.median().item():.3f}, "
         f"mean z median cos {cos_mean.median().item():.3f}, gap {gap:+.3f} "
         f"(reference: +{REFERENCE_GAP}, bar: >{BAR:.3f})")
    print("=" * 70)
    print(f"{'channel':<10}{'win-rate (real closer than mean)':>36}   chance: 50%")
    for c, name in enumerate(CHANNEL_NAMES):
        err_real = (pred_real[:, c] - true_change[:, c]).abs()
        err_mean = (pred_mean[:, c] - true_change[:, c]).abs()
        win_rate = (err_real < err_mean).float().mean().item()
        print(f"{name:<10}{win_rate:>35.1%}")
    print()
    return gap


real_z_itm = batched_itm(e_t_cpu, e_next_cpu)
gap_itm = report("Z = ITM(e_t, e_next)  [matches the established +0.042 reference]",
                 real_z_itm, real_z_itm.mean(0))
del real_z_itm
torch.cuda.empty_cache()

with torch.no_grad():
    real_z_proj = proj(actions_all, EMBODIMENT)
gap_proj = report("Z = proj(action)  [the actual control-relevant path]",
                  real_z_proj, real_z_proj.mean(0))

print("=" * 70)
print("VERDICT")
print("=" * 70)
for tag, gap in (("ITM-z (reference-matching)", gap_itm), ("proj(a)-z (control-relevant)", gap_proj)):
    verdict = "PASS" if gap > BAR else "FAIL"
    print(f"{tag}: gap {gap:+.3f} vs bar {BAR:.3f} -> {verdict}")
