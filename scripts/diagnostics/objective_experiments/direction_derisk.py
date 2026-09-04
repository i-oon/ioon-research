"""Can an objective that asks for direction move the FTM's cosine above 0.690?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/direction_derisk.py

`cosine_ceiling.py` ruled out a generalisation gap -- the FTM scores 0.699 on the body it trained on
against 0.687 held out -- and could not bound the ceiling from above, because every cheap oracle came
in near chance. What is left is whether 0.690 is what **MSE happens to yield** or what the model can
reach. MSE on a 360k-dimensional vector never asks for direction directly, so a loss that does is the
pointed test.

    control      MSE                                          today's objective
    direction    MSE + lambda * (1 - cos(pred - e_t, e_t+1 - e_t))

**The read.** Held-out cosine clearly above 0.687 under a term that asks for exactly that quantity
means headroom, and the fix is the objective. No movement, under a loss aimed straight at the
number, is strong evidence the ceiling is a property of the latent geometry rather than of training.

Same controls as `multistep_derisk.py`: ITM frozen so `z` is identical across arms, all arms from the
same weights with the same seed and batch order, fitted on the pretraining body and measured on the
held-out one. **Report the arm difference, not the treated arm alone.**

**Diagnosis only; writes no checkpoint into wm/runs.**
"""
import argparse
import copy
import os
import sys
import time

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


def pairs_of(clips):
    return [(ci, t) for ci, c in enumerate(clips) for t in range(1, len(c["e"]) - 2)]


@torch.no_grad()
def measure(ftm, itm, clips, device, stride):
    cos, ratio, mse = [], [], 0.0
    n = 0
    for c in clips:
        e = c["e"].float()
        for t in range(1, len(e) - 2, stride):
            e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
            p = ftm(e_t, itm(e_t, e1))
            dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
            cos.append(F.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)).item())
            ratio.append((dp.norm() / dt.norm().clamp_min(1e-9)).item())
            mse += F.mse_loss(p, e1).item()
            n += 1
    return float(np.median(cos)), float(np.median(ratio)), mse / max(n, 1)


def finetune(ftm, itm, clips, idx, lam, steps, batch, lr, seed, device):
    ftm.train()
    opt = torch.optim.AdamW(ftm.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        pick = [idx[i] for i in torch.randint(len(idx), (batch,), generator=g).tolist()]
        e_t = torch.stack([clips[c]["e"][t] for c, t in pick]).float().to(device)
        e1 = torch.stack([clips[c]["e"][t + 1] for c, t in pick]).float().to(device)
        with torch.no_grad():
            z = itm(e_t, e1)
        p = ftm(e_t, z)
        loss = F.mse_loss(p, e1)
        if lam > 0:
            cos = F.cosine_similarity((p - e_t).flatten(1), (e1 - e_t).flatten(1), dim=1)
            loss = loss + lam * (1 - cos).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ftm.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.5f}  "
                  f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
    ftm.eval()
    return ftm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 10.0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    lag = max(1, cfg.action_lag)
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    tr = gather(os.path.join(ROOT, args.train_data), args.embodiment, None, ck, cache, 2, lag,
                device)
    te = gather(os.path.join(ROOT, args.test_data), args.embodiment, None, ck, cache, 2, lag,
                device)
    idx = pairs_of(tr)

    base = ForwardTransitionModel(cfg).to(device).eval()
    base.load_state_dict(ck["ftm"])
    print(f"{args.ckpt}\ntrain {len(tr)} clips ({len(idx)} pairs), test {len(te)} clips\n")
    b_cos, b_ratio, b_mse = measure(base, itm, te, device, args.stride)
    print(f"  BEFORE   held-out cosine {b_cos:.3f}   ratio {b_ratio:.3f}   mse {b_mse:.4f}\n")

    rows = [("before", b_cos, b_ratio, b_mse, None)]
    for lam in args.lambdas:
        name = "control (MSE)" if lam == 0 else f"direction lambda={lam:g}"
        print(f"  ARM {name}")
        ftm = copy.deepcopy(base)
        for p in ftm.parameters():
            p.requires_grad_(True)
        finetune(ftm, itm, tr, idx, lam, args.steps, args.batch, args.lr, args.seed, device)
        c, r, m = measure(ftm, itm, te, device, args.stride)
        ctr, _r, _m = measure(ftm, itm, tr, device, args.stride)
        rows.append((name, c, r, m, ctr))
        print(f"    held-out cosine {c:.3f}   train cosine {ctr:.3f}   ratio {r:.3f}   "
              f"mse {m:.4f}\n")
        del ftm
        torch.cuda.empty_cache()

    print(f"  {'arm':>22}{'cosine':>9}{'vs before':>11}{'train cos':>11}{'ratio':>8}{'mse':>9}")
    for name, c, r, m, ctr in rows:
        print(f"  {name:>22}{c:>9.3f}{c - b_cos:>+11.3f}"
              f"{(f'{ctr:.3f}' if ctr is not None else '-'):>11}{r:>8.3f}{m:>9.4f}")
    ctrl = next((c for n, c, _r, _m, _t in rows if n.startswith("control")), None)
    if ctrl is not None:
        for name, c, _r, _m, _t in rows:
            if name.startswith("direction"):
                print(f"  {name} minus control: {c - ctrl:+.3f}")
        print("\n  **Read against the control, not against `before`.** A shared move is the "
              "fine-tune;\n  only the difference is the direction term.")


if __name__ == "__main__":
    main()
