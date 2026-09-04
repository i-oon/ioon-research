"""Does the FTM under-move, and is that the MSE optimum or a model failure?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/undermovement.py

`ftm_z_path.py` decomposed the action stamp into `e_t` inheritance plus a distributed `z` term, and
the residue is that the prediction sits near its input: 1-step MSE 3.325 against hold-still 6.466.
This measures the displacement directly and asks which of two causes it has.

    d_true = ||e_t+1 - e_t||        how far the world actually moved
    d_pred = ||pred  - e_t||        how far the model moved
    ratio  = d_pred / d_true        under 1 is under-movement
    cos    = angle between the two displacements

**The discriminator is the rescaling sweep, not the ratio.** Rescale the model's own displacement,
`e_t + alpha * (pred - e_t)`, and find the `alpha` that minimises MSE against the truth:

    alpha* ~ 1  while ratio < 1     the prediction is already the conditional mean. Shrinking is
                                    what MSE *wants* under an uncertain transition, so no loss that
                                    rewards movement can be added without making prediction worse.
                                    **Target-driven; the fix is a different target, not a penalty.**

    alpha* >> 1                     the model moves less than even MSE would prefer, so accuracy is
                                    being left on the table. **Model- or objective-driven, and a
                                    movement term is coherent.**

Binned by `d_true` because the two causes differ there: an averaging estimator shrinks *more* where
the outcome is more uncertain, so a ratio that falls as the true transition grows is the
uncertainty signature.

**Diagnosis only; trains nothing.**
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, None, ck, cache, 2,
                   max(1, cfg.action_lag), device)

    d_true, d_pred, cosines, fam = [], [], [], []
    num, den, cross = 0.0, 0.0, 0.0     # for the closed-form optimal alpha over the whole set
    err_true = 0.0
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            for t in range(1, len(e) - 2, args.stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                p = ftm(e_t, z)
                dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
                d_pred.append(dp.norm().item())
                d_true.append(dt.norm().item())
                cosines.append(torch.nn.functional.cosine_similarity(
                    dp.unsqueeze(0), dt.unsqueeze(0)).item())
                cross += torch.dot(dp, dt).item()
                den += torch.dot(dp, dp).item()
                err_true += (p - e1).pow(2).mean().item()
                num += 1
    d_true = np.array(d_true); d_pred = np.array(d_pred); cosines = np.array(cosines)
    ratio = d_pred / np.maximum(d_true, 1e-9)
    alpha_star = cross / max(den, 1e-12)     # argmin_a ||e_t + a*dp - e_t+1||^2

    print(f"{args.ckpt}\n{len(clips)} clips from {args.data}, {num} transitions\n")
    print(f"  median displacement ratio  d_pred/d_true : {np.median(ratio):.3f}")
    print(f"  mean                                     : {ratio.mean():.3f}"
          f"  (sd {ratio.std():.3f})")
    print(f"  fraction of transitions with ratio < 1   : {(ratio < 1).mean():.1%}")
    print(f"  median cosine(pred-e_t, true-e_t)        : {np.median(cosines):.3f}")
    print(f"\n  optimal rescaling of the model's own displacement")
    print(f"    alpha* = {alpha_star:.3f}   (1.0 = already calibrated; >1 = moves less than MSE wants)")

    print(f"\n  by size of the true transition, {args.bins} equal-count bins")
    print(f"  {'d_true range':>22}{'ratio':>9}{'cosine':>9}{'n':>7}")
    edges = np.quantile(d_true, np.linspace(0, 1, args.bins + 1))
    for i in range(args.bins):
        lo, hi = edges[i], edges[i + 1]
        m = (d_true >= lo) & (d_true <= hi if i == args.bins - 1 else d_true < hi)
        if m.sum():
            print(f"  {f'{lo:.1f}-{hi:.1f}':>22}{np.median(ratio[m]):>9.3f}"
                  f"{np.median(cosines[m]):>9.3f}{m.sum():>7}")

    trend = np.corrcoef(d_true, ratio)[0, 1]
    print(f"\n  correlation(d_true, ratio) = {trend:+.3f}   "
          + ("ratio falls as the transition grows -- the averaging signature"
             if trend < -0.1 else
             "no size dependence -- uniform shrinkage" if trend < 0.1 else
             "ratio rises with transition size"))


if __name__ == "__main__":
    main()
