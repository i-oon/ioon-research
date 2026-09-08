"""Does z trained WITHOUT any L_body gradient (recon+motion only, from scratch) still develop the
Froude signal? The gate check before the 9-hour beh12_body_stopgrad retrain on BIAS-2.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/proxy_z_signal_check.py [--ckpt PATH]

**Why this exists.** Every prior passing test (`isolated_z_head_probe*.py`) probed a `z` from
`teacher_state.pt`, which was trained WITH L_body's gradient shaping it (`lambda_body=0.5`, full
joint recipe). The planned fix (`z.detach()` before L_body) trains a NEW `z` under recon+motion
ONLY -- an untested regime. No existing checkpoint (audited: every non-Stage-1 run either has
`lambda_body>0` or is allocentric/pre-cross-embodiment data) could answer this cheaply, so a short
(8-epoch, from-scratch) proxy was trained on BIAS-2 with `lambda_body=0.0` -- pure recon+motion,
same data/sources as the real retrain. This probes that proxy's `z`.

Loads `itm`/`ftm` directly from `best.pt` -- no `load_teacher()`, since this proxy has no action
projector (never assembled, and not needed: only `z = ITM(e_t,e_next)` is tested here, not
`proj(action)`; the control-relevant path is a separate, later question once the real fix exists).

Two targets probed, matching both prior isolated-head runs exactly:
  - delta-Froude (`bm_next - bm_t`), reference: teacher_state.pt's z gave rho +0.535
  - absolute body_motion[t], L_body's REAL target, reference: teacher_state.pt's z (via the
    isolated head, not this same probe, so treat as directional) supported rho 0.69-0.94 per channel

Same ridge + kNN(k=5,15) probe, held out by clip, as every other offline check in this arc.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="wm/runs/beh12_proxy_reconmotion/best.pt")
args = ap.parse_args()

EMBODIMENT = "b1"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
# teacher_state.pt's z (trained WITH L_body gradient), per-channel, from the isolated-head runs --
# the honest comparison is per-channel, not a single pooled number.
PER_CHANNEL_REFERENCE = {
    "delta-Froude (bm_next - bm_t)": {"forward": 0.529, "lateral": 0.598, "yaw": 0.535},
    "absolute body_motion[t] (L_body's real target)": {"forward": 0.660, "lateral": 0.755, "yaw": 0.938},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print(f"loading {args.ckpt} -- recon+motion-ONLY proxy, no L_body ever trained on this z...")
ck = torch.load(os.path.join(ROOT, args.ckpt), map_location=device, weights_only=False)
cfg = from_checkpoint(ck["config"])
print(f"cfg.lambda_body = {getattr(cfg, 'lambda_body', 'MISSING')}, "
     f"epoch = {ck.get('epoch', '?')}")
itm = InverseTransitionModel(cfg).to(device).eval()
itm.load_state_dict(ck["itm"])
for p in itm.parameters():
    p.requires_grad_(False)

# NOTE: this proxy was trained with cfg.body_channels=[0] (forward only -- --body_channels 0 1 2
# was omitted from the launch command). But lambda_body=0.0 here, so body_channels never touched
# how z itself was trained (no body loss ran at all) -- probe all 3 physical channels regardless,
# standardising from THIS gathered data rather than the checkpoint's 1-channel-sized body_stats.
channels = [0, 1, 2]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

Z_tr, Z_te, Bmt_tr, Bmnext_tr, Bmt_te, Bmnext_te = [], [], [], [], [], []
with torch.no_grad():
    for p in paths:
        clip = load(p, REGISTRY[EMBODIMENT])
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)   # [T, 256, 1408], full grid
        # raw, unstandardised -- Spearman rho is invariant to any monotonic (incl. affine) rescaling
        bm = np.asarray(clip["body_motion"])[:, channels]
        n = min(len(e), len(bm)) - 1
        z_list = []
        for s in range(0, n, 32):
            sl = slice(s, min(s + 32, n))
            z_list.append(itm(e[sl], e[1:n + 1][sl]).cpu().numpy())   # full tokens in, z out
        z = np.concatenate(z_list)
        bmt, bmnext = bm[:n], bm[1:n + 1]
        is_held = p in held_paths
        (Z_te if is_held else Z_tr).append(z)
        (Bmt_te if is_held else Bmt_tr).append(bmt)
        (Bmnext_te if is_held else Bmnext_tr).append(bmnext)

Z_tr, Z_te = np.concatenate(Z_tr), np.concatenate(Z_te)
Bmt_tr, Bmnext_tr = np.concatenate(Bmt_tr), np.concatenate(Bmnext_tr)
Bmt_te, Bmnext_te = np.concatenate(Bmt_te), np.concatenate(Bmnext_te)
print(f"train: {len(Z_tr)} transitions, test: {len(Z_te)} transitions\n")

targets = {
    "delta-Froude (bm_next - bm_t)": (Bmnext_tr - Bmt_tr, Bmnext_te - Bmt_te),
    "absolute body_motion[t] (L_body's real target)": (Bmt_tr, Bmt_te),
}

print(f"{'target':<48}{'probe':<10}{'median rho':>12}   per-channel rho")
summary = {}   # tname -> {channel: (this_rho, reference_rho, retention_pct)}
for tname, (Ytr, Yte) in targets.items():
    scaler = StandardScaler().fit(Z_tr)
    Zs_tr, Zs_te = scaler.transform(Z_tr), scaler.transform(Z_te)
    best_med, best_desc, best_rhos = -1e9, None, None
    for pname, model in (("ridge", Ridge(alpha=10.0)), ("kNN k=5", KNeighborsRegressor(5)),
                        ("kNN k=15", KNeighborsRegressor(15))):
        model.fit(Zs_tr, Ytr)
        pred = np.atleast_2d(model.predict(Zs_te).reshape(len(Zs_te), -1))
        Yte2 = np.atleast_2d(Yte.reshape(len(Yte), -1))
        rhos = [spearmanr(Yte2[:, c], pred[:, c])[0] for c in range(len(channels))]
        med = float(np.median(rhos))
        if med > best_med:
            best_med, best_desc, best_rhos = med, pname, rhos
    print(f"{tname:<48}{best_desc:<10}{best_med:>+12.3f}")
    print("  per-channel: " + "  ".join(f"{n}={r:+.3f}" for n, r in zip(CHANNEL_NAMES, best_rhos)))
    refs = PER_CHANNEL_REFERENCE[tname]
    per_ch = {}
    for n, r in zip(CHANNEL_NAMES, best_rhos):
        ref = refs[n]
        retention = (r / ref * 100) if ref > 0 else float("nan")
        per_ch[n] = (r, ref, retention)
    print("  vs teacher_state.pt (WITH L_body gradient): " +
         "  ".join(f"{n}: {r:+.3f}/{ref:+.3f} ({ret:.0f}%)" for n, (r, ref, ret) in per_ch.items()))
    summary[tname] = per_ch

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
all_retentions = [ret for per_ch in summary.values() for (_, _, ret) in per_ch.values()
                  if not np.isnan(ret)]
weak_channels = [(t, n) for t, per_ch in summary.items() for n, (r, ref, ret) in per_ch.items()
                 if ref > 0 and ret < 60]
if not weak_channels:
    print("-> SIGNAL FULLY PRESENT under recon+motion ONLY, retention >=60% of the WITH-L_body "
         "reference on every channel. Stop-grad is safe: the new z WILL develop the Froude "
         "signal on its own. Proceed with the 9-hour beh12_body_stopgrad retrain on BIAS-2.")
elif any(ret > 15 for per_ch in summary.values() for (_, _, ret) in per_ch.values()):
    print(f"-> PARTIAL: signal present but WEAKER than the WITH-L_body reference on "
         f"{len(weak_channels)} channel(s): {weak_channels}. L_body's gradient was doing some "
         "real work shaping z on top of what recon+motion develops alone -- full z.detach() will "
         "likely underperform the WITH-L_body z on those specific channels, though not "
         "necessarily worse than the CURRENT jointly-trained-but-unread failure (+0.045/+0.099 "
         "action-lever). Proceed with the retrain, but do not expect full parity on the weak "
         "channel(s); re-measure the action-lever per-channel once trained, do not assume success.")
else:
    print("-> SIGNAL WEAK OR ABSENT under recon+motion alone on most channels. L_body's gradient "
         "was doing real work shaping z's Froude content, not just competing for it. Full "
         "z.detach() risks losing the signal -- reconsider PARTIAL stop-gradient or the staged "
         "approach (train WITH L_body, then freeze and refit) instead of a clean detach.")
