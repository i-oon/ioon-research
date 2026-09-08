"""The decisive check: does the ACTUAL trained `body_head`, from the real `beh12_body_stopgrad`
retrain (z.detach() on L_body, L_state retired), clear the action-lever bar per channel?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/stopgrad_action_lever_check.py [--ckpt PATH]

**What this settles.** Every prior test in this thread was either an offline probe (ridge/kNN,
no training) or an ISOLATED freshly-trained head reading a frozen, pre-existing `z`
(`isolated_z_head_probe*.py`) -- both proxies for the real question. This is the real question:
after a full joint retrain with the stop-gradient fix actually applied, does the network's own
`body_head` -- reading whatever `z` the retrain actually produced, trained end-to-end alongside
`L_recon`/`L_motion` -- predict real actions better than generic/mean ones?

**Method**: `body_head(e_t, z)` needs only `itm` (for `z = ITM(e_t,e_next)`) and `md` (for
`.body(e_t, z)`) -- no FTM, no projector (this checkpoint has neither assembled; L_body's own
forward pass never used FTM at all, see `wm/train.py:201`). Loaded directly from `best.pt`, same
pattern as `proxy_z_signal_check.py`. Real-vs-mean-z action-lever, same methodology as every
other lever in this arc: hold the frame fixed, swap `z` (real vs the training-set mean), predict
`body_head`'s output, score cosine similarity against the TRUE absolute `body_motion[t]` (L_body's
actual target -- not delta-Froude).

**Pre-registered reading (from the proxy check's retention numbers, forward weakest at 32-49%,
lateral/yaw at 61-76%):**
  - ALL channels clear 0.110               -> full stop-gradient wins outright, done.
  - forward fails, lateral/yaw clear       -> matches the proxy's prediction; full detach costs
                                              forward specifically; scope PARTIAL/scaled
                                              stop-gradient as the follow-up, not a different fix.
  - ALL channels fail                      -> deeper than this diagnosis; re-open.
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

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="wm/runs/beh12_body_stopgrad/best.pt")
args = ap.parse_args()

EMBODIMENT = "b1"
DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
CHANNEL_NAMES = ["forward", "lateral", "yaw"]
HELD_OUT_FRAC = 0.2
SEED = 0
BAR = 2 * 0.055
SBATCH = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = VJEPA2FrameEncoder(dtype=torch.float32)

print(f"loading {args.ckpt} -- the REAL beh12_body_stopgrad retrain (z.detach() on L_body, "
     "L_state retired)...")
ck = torch.load(os.path.join(ROOT, args.ckpt), map_location=device, weights_only=False)
cfg = from_checkpoint(ck["config"])
print(f"cfg.lambda_body={getattr(cfg, 'lambda_body', '?')}, "
     f"cfg.body_channels={cfg.body_channels}, cfg.body_dim={cfg.body_dim}, epoch={ck.get('epoch', '?')}")
assert cfg.body_dim == len(cfg.body_channels), \
    f"body_dim {cfg.body_dim} != len(body_channels) {len(cfg.body_channels)} -- the exact bug " \
    "already caught once this session; do not trust this checkpoint's body loss if this fails."

itm = InverseTransitionModel(cfg).to(device).eval()
itm.load_state_dict(ck["itm"])
md = MotionDecoder(cfg, {"hexapod": 18}).to(device).eval()
md.load_state_dict(ck["md"], strict=False)
for m in (itm, md):
    for p in m.parameters():
        p.requires_grad_(False)
assert md.body_head is not None, "this checkpoint has no body_head -- wrong lambda_body?"

channels = [int(c) for c in cfg.body_channels]
mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(paths))
n_held = int(len(paths) * HELD_OUT_FRAC)
held_paths = {paths[i] for i in order[:n_held]}
print(f"{len(paths)} clips, {n_held} held out\n")

E_t_te = []
Z_tr, Z_te = [], []
Bmt_te = []
with torch.no_grad():
    for p in paths:
        is_held = p in held_paths
        clip = load(p, REGISTRY[EMBODIMENT])
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)   # [T, 256, 1408]
        bm = (np.asarray(clip["body_motion"])[:, channels] - mean_s) / std_s
        n = min(len(e), len(bm)) - 1
        z_list = []
        for s in range(0, n, SBATCH):
            sl = slice(s, min(s + SBATCH, n))
            z_list.append(itm(e[sl], e[1:n + 1][sl]))
        z = torch.cat(z_list)
        if is_held:
            # only held-out clips need frames kept (half precision -- the memory-safe pattern
            # this session settled on; storing all 48 clips' frames in float32 OOM'd here)
            E_t_te.append(e[:n].half())
            Bmt_te.append(bm[:n])
            Z_te.append(z)
        else:
            Z_tr.append(z)   # train clips: only z is needed, for mean_z below
        del e, z_list, z
        torch.cuda.empty_cache()

Z_tr, Z_te = torch.cat(Z_tr), torch.cat(Z_te)
E_t_te = torch.cat(E_t_te)
Bmt_te = torch.tensor(np.concatenate(Bmt_te), dtype=torch.float32, device=device)
print(f"train z: {len(Z_tr)} transitions, held-out: {len(Z_te)} transitions\n")

mean_z = Z_tr.mean(0, keepdim=True)

cos_real_list, cos_mean_list = [], []
with torch.no_grad():
    for s in range(0, len(E_t_te), SBATCH):
        sl = slice(s, s + SBATCH)
        e_t = E_t_te[sl].float()
        real_z = Z_te[sl]
        mz = mean_z.expand(e_t.shape[0], -1)
        pred_real = md.body(e_t, real_z)
        pred_mean = md.body(e_t, mz)
        true = Bmt_te[sl]
        cos_real_list.append(F.cosine_similarity(pred_real, true, dim=1).cpu())
        cos_mean_list.append(F.cosine_similarity(pred_mean, true, dim=1).cpu())
        if s == 0:
            pred_real0, pred_mean0, true0 = pred_real.cpu(), pred_mean.cpu(), true.cpu()
        else:
            pred_real0 = torch.cat([pred_real0, pred_real.cpu()])
            pred_mean0 = torch.cat([pred_mean0, pred_mean.cpu()])
            true0 = torch.cat([true0, true.cpu()])

cos_real = torch.cat(cos_real_list)
cos_mean = torch.cat(cos_mean_list)
gap = cos_real.median().item() - cos_mean.median().item()

print("=" * 70)
print(f"ACTION-LEVER: real z vs mean z, body_head's ACTUAL trained prediction, "
     f"absolute body_motion[t] target")
print("=" * 70)
print(f"real z:  median cos {cos_real.median().item():.3f}")
print(f"mean z:  median cos {cos_mean.median().item():.3f}")
print(f"gap: {gap:+.3f}   (bar: >{BAR:.3f})")

print(f"\n{'channel':<10}{'sign-agreement gap (real - mean)':>36}")
for c, name in enumerate(CHANNEL_NAMES[:len(channels)]):
    real_agree = (torch.sign(pred_real0[:, c]) == torch.sign(true0[:, c])).float().mean().item()
    mean_agree = (torch.sign(pred_mean0[:, c]) == torch.sign(true0[:, c])).float().mean().item()
    print(f"{name:<10}{real_agree - mean_agree:>+36.3f}   (real {real_agree:.1%} vs mean {mean_agree:.1%})")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if gap > BAR:
    print(f"-> PASS (gap {gap:+.3f} > {BAR:.3f}): full stop-gradient wins outright. Done.")
else:
    print(f"-> FAIL on pooled gap (gap {gap:+.3f} <= {BAR:.3f}). Check the per-channel "
         "sign-agreement above against the pre-registered expectation: forward weak, "
         "lateral/yaw carrying the signal -> matches the proxy's prediction, scope PARTIAL/"
         "scaled stop-gradient next. All channels near zero or negative -> deeper than this "
         "diagnosis, re-open.")
