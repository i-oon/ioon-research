"""Does B1's stage-1 adaptation (`wm.adapt`, plain one-step MSE, no hinge/readout/rollout terms)
erase the real-vs-null action separation the hexapod pretrain built (F199)?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/b1_adaptation_sep_check.py \\
        --pretrain wm/runs/beh12_hinge_multistep_anchor_v2/best.pt \\
        --adapted wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt/adapted_b1.pt

**Why this check.** `wm.adapt` (B1's stage 1) fine-tunes ITM+FTM with a single one-step
reconstruction loss (`scripts/diagnostics/cross_embodiment/finetune_ftm.py:adapt`, docstring:
"One-step prediction loss on the target clips") -- no `lambda_hinge`, `lambda_readout`, or
`lambda_rollout` term survives into this stage. If B1's own reward-quality-gate failure (F199
coda) is caused by stage 1 re-introducing exactly the MSE-dominance F142/F199 diagnosed and fixed
at the pretraining level, this metric should show it directly: the SAME `sep` quantity
`wm.train`'s own hinge term computes (real-action rollout vs null-action rollout, `1 -
cosine_similarity`, no action projector or embodiment token needed since `ftm_embodiment_channel`
is off by default) should be high on the pretrain checkpoint and collapse after stage 1's
adaptation, on B1's own held-out frames.

**What this does NOT need**: an action projector, an embodiment token, or any B1-specific action
plumbing at all -- `null = ITM(e_t, e_t)` ("nothing happened") and `real = ITM(e_t, e_t+1)` (the
true transition) are computed directly from frames, exactly as `wm/train.py`'s hinge term does.
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
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def load_itm_ftm(path, device):
    ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    return itm, ftm


def measure_sep(itm, ftm, clips_e, hinge_k, device):
    """`sep` at each of hinge_K steps, real-action rollout vs null-action rollout, same
    computation `wm/train.py`'s hinge term uses -- no embodiment/action plumbing needed."""
    seps_per_k = [[] for _ in range(hinge_k)]
    with torch.no_grad():
        for e in clips_e:
            n = len(e) - 1
            for t in range(0, n, 4):   # every 4th transition, matching this arc's sampling convention
                e_t = e[t:t + 1].to(device)
                e_next = e[t + 1:t + 2].to(device)
                z_real = itm(e_t, e_next)
                z_null = itm(e_t, e_t)
                real, null = e_t, e_t
                for k in range(hinge_k):
                    real = ftm(real, z_real)
                    null = ftm(null, z_null)
                    sep = 1 - F.cosine_similarity(real.flatten(1), null.flatten(1), dim=1).item()
                    seps_per_k[k].append(sep)
    return [float(np.mean(s)) for s in seps_per_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", required=True, help="the hexapod pretrain, before B1 adaptation")
    ap.add_argument("--adapted", required=True, help="wm.adapt's stage-1 output (adapted_b1.pt)")
    ap.add_argument("--data", default="data/egocentric/beh12_b1_ego_flat")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--hinge_k", type=int, default=2)
    ap.add_argument("--n_clips", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    reg = REGISTRY[args.embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))[:args.n_clips]
    clips_e = []
    for p in paths:
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float()
        clips_e.append(e)
    del encoder
    torch.cuda.empty_cache()
    print(f"{len(clips_e)} B1 clips loaded\n")

    print(f"loading pretrain: {args.pretrain}")
    itm_pre, ftm_pre = load_itm_ftm(args.pretrain, device)
    print(f"loading stage-1 adapted: {args.adapted}")
    itm_ad, ftm_ad = load_itm_ftm(args.adapted, device)

    sep_pre = measure_sep(itm_pre, ftm_pre, clips_e, args.hinge_k, device)
    sep_ad = measure_sep(itm_ad, ftm_ad, clips_e, args.hinge_k, device)

    print(f"\n{'step':>6}{'pretrain sep':>16}{'adapted sep':>16}")
    for k in range(args.hinge_k):
        print(f"{k + 1:>6}{sep_pre[k]:>16.4f}{sep_ad[k]:>16.4f}")

    print("\n`sep` = 1 - cosine_similarity(real-action rollout, null-action rollout) on B1's own "
         "frames -- higher means the model treats the real transition as more distinct from "
         "'nothing happened'. If `adapted` is much LOWER than `pretrain` at every step, stage 1's "
         "plain one-step-MSE fine-tune erased the separation the hinge+multistep-anchor pretrain "
         "built, on B1's own data -- the likely cause of F199's B1 propagation failure, and a fixable "
         "procedure gap (carry the hinge/rollout/readout terms into stage 1), not a fundamental wall.")


if __name__ == "__main__":
    main()
