"""Does the action's effect on the transition grow with the interval?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/action_lever_vs_horizon.py

At one step the action is worth **+0.055 cosine** to the FTM's predicted direction (0.689 with the
real `z`, 0.634 with the mean one). That tiny lever is everything a ranker has. This asks whether it
is a property of the 50 ms step rather than of the model, by widening the interval.

Two measurements per interval `k`, deliberately of different kinds:

    data      **model-free.** Take true displacement directions `e_t+k - e_t` from *different clips*
              and compare pairs sharing a behaviour condition against pairs that do not. The gap is
              how much the commanded behaviour determines where the body goes over `k` steps, with
              no model in the path at all.

    model     the FTM's real-`z` minus mean-`z` direction cosine, with `z = ITM(e_t, e_t+k)`.

**The model row is out of distribution for k > 1 and is reported second for that reason.** This FTM
was trained at `frame_stride` 1, so asking it for a `k`-step displacement asks for something it never
fitted; cosine is scale-free, which is why the row is worth reading at all, but a fall with `k` is
expected from the mismatch and must not be read as the lever shrinking. **The data row carries the
finding.**

Pairs are drawn across clips only: two transitions from one clip a few frames apart are the same
piece of gait and would inflate the within-condition number.

**Diagnosis only; trains nothing.**
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
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
    conds = [c["cond"] for c in clips]
    print(f"{args.ckpt}\n{len(clips)} clips, {len(set(conds))} conditions, "
          f"from {args.data}\n", flush=True)

    rng = np.random.default_rng(args.seed)
    print(f"  {'k':>4}{'steps (ms)':>12}"
          f"{'within-cond':>13}{'between-cond':>14}{'DATA gap':>11}"
          f"{'real z':>9}{'mean z':>9}{'MODEL gap':>11}")
    for k in args.ks:
        D, cid, cond_id = [], [], []
        real, meanz = [], []
        zs = []
        with torch.no_grad():
            for ci, c in enumerate(clips):
                e = c["e"].float()
                for t in range(1, len(e) - k - 1, args.stride):
                    e_t, ek = e[t:t + 1].to(device), e[t + k:t + k + 1].to(device)
                    D.append(F.normalize((ek - e_t).flatten(), dim=0).half().cpu())
                    cid.append(ci); cond_id.append(conds[ci])
                    zs.append(itm(e_t, ek))
            zbar = torch.cat(zs).mean(0, keepdim=True)
            i = 0
            for ci, c in enumerate(clips):
                e = c["e"].float()
                for t in range(1, len(e) - k - 1, args.stride):
                    e_t, ek = e[t:t + 1].to(device), e[t + k:t + k + 1].to(device)
                    dt = (ek - e_t).flatten()
                    for tag, zz in (("r", zs[i]), ("m", zbar)):
                        dp = (ftm(e_t, zz) - e_t).flatten()
                        (real if tag == "r" else meanz).append(
                            F.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)).item())
                    i += 1
        D = torch.stack(D)
        cid = np.array(cid); cond_id = np.array(cond_id)

        a = rng.integers(0, len(D), args.pairs)
        b = rng.integers(0, len(D), args.pairs)
        ok = cid[a] != cid[b]                      # different clips only
        a, b = a[ok], b[ok]
        cs = np.empty(len(a))
        step = 256                                  # 4096 rows of 360,448 floats is 5.9 GB on GPU
        for s in range(0, len(a), step):
            xa = D[a[s:s + step]].to(device).float()
            xb = D[b[s:s + step]].to(device).float()
            cs[s:s + step] = F.cosine_similarity(xa, xb, dim=1).cpu().numpy()
            del xa, xb
        torch.cuda.empty_cache()
        same = cond_id[a] == cond_id[b]
        w, bt = cs[same].mean(), cs[~same].mean()
        print(f"  {k:>4}{k * 50:>12}{w:>13.3f}{bt:>14.3f}{w - bt:>11.3f}"
              f"{np.median(real):>9.3f}{np.median(meanz):>9.3f}"
              f"{np.median(real) - np.median(meanz):>11.3f}", flush=True)

    print("\n  DATA gap = how much sharing a behaviour makes two transitions point the same way,")
    print("  with no model involved. If it grows with k, the action shapes longer intervals more")
    print("  and the training interval is the lever. If it stays flat, it does not.")
    print("  MODEL gap is out of distribution for k > 1 (this FTM was trained at stride 1).")


if __name__ == "__main__":
    main()
