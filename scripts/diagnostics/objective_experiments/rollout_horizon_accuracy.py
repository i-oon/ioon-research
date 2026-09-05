"""Does auto-regressive rollout supervision (lambda_rollout) actually keep e_t+k accurate longer?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/rollout_horizon_accuracy.py \\
        --ckpt_a wm/runs/beh12_ego/teacher_ego.pt --ckpt_b wm/runs/derisk_rollout5/best.pt

**What this is not.** `action_lever_vs_horizon.py` asks the FTM for a single-shot k-step jump from
a synthetic `z = ITM(e_t, e_t+k)` -- it never rolls the model on its own output. This asks the other
half of the same question: fed forward AUTO-REGRESSIVELY, one real single-step `z` at a time
(`z_i = ITM(e_i, e_i+1)` from real consecutive frames, matching exactly how `lambda_rollout`'s own
training term is built), does the FTM's own k-step-ahead prediction stay close to the true `e_t+k`,
and does that get better after training with the rollout term?

Two numbers per k, both against a fixed reference so they read as ratios rather than raw MSE:

    model rollout MSE     ||rollout_k(e_t) - e_t+k||^2, auto-regressive
    copy-forward MSE      ||e_t - e_t+k||^2, the do-nothing baseline (F87's shape)

A ratio near or above 1.0 means the model has stopped adding anything past that horizon -- it would
have done as well by predicting no change at all. Falling ratio with training is the direct test of
whether `lambda_rollout` earned its cost.

Diagnosis only; trains nothing.
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

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def load(ckpt, device):
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)
    return ck, cfg, itm, ftm


@torch.no_grad()
def rollout_mse(itm, ftm, clips, ks, stride, device, embodiment):
    out = {k: {"model": [], "copy": []} for k in ks}
    kmax = max(ks)
    for c in clips:
        e = c["e"].float()
        for t in range(1, len(e) - kmax - 1, stride):
            cur = e[t:t + 1].to(device)
            e0 = cur.clone()
            for step in range(1, kmax + 1):
                nxt = e[t + step:t + step + 1].to(device)
                z = itm(cur, nxt)                 # real single-step z, matching training exactly
                cur = ftm(cur, z, embodiment)      # auto-regressive: feed the model's own output back
                if step in ks:
                    truth = e[t + step:t + step + 1].to(device)
                    out[step]["model"].append(F.mse_loss(cur, truth).item())
                    out[step]["copy"].append(F.mse_loss(e0, truth).item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", default="wm/runs/beh12_ego/teacher_ego.pt",
                    help="baseline -- no rollout term")
    ap.add_argument("--ckpt_b", required=True, help="the lambda_rollout-trained checkpoint")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'k':>4}{'a: model':>12}{'a: copy':>10}{'a: ratio':>10}   "
          f"{'b: model':>12}{'b: copy':>10}{'b: ratio':>10}")
    for tag, ckpt in (("a", args.ckpt_a), ("b", args.ckpt_b)):
        ck, cfg, itm, ftm = load(ckpt, device)
        cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
        clips = gather(os.path.join(ROOT, args.data), args.embodiment, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        res = rollout_mse(itm, ftm, clips, args.ks, args.stride, device, args.embodiment)
        globals()[f"res_{tag}"] = res
        globals()[f"label_{tag}"] = ckpt

    for k in args.ks:
        row = f"{k:>4}"
        for tag in ("a", "b"):
            res = globals()[f"res_{tag}"]
            m = np.mean(res[k]["model"]); c = np.mean(res[k]["copy"])
            row += f"{m:>12.4f}{c:>10.4f}{m / max(c, 1e-9):>10.3f}   " if tag == "a" else \
                   f"{m:>12.4f}{c:>10.4f}{m / max(c, 1e-9):>10.3f}"
        print(row)
    print(f"\na = {globals()['label_a']}\nb = {globals()['label_b']}")
    print("\nratio < 1.0: the model beats copy-forward at this horizon. Falling ratio b vs a at the "
          "same k is the direct test of whether lambda_rollout improved auto-regressive accuracy.")


if __name__ == "__main__":
    main()
