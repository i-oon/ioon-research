"""Does single-frame command-recoverability error oscillate with the gait cycle, or just stay flat?

F26/F46 sampled 7 offsets (0,1,2,4,8,16,32) on `fwd_m3d`'s c10f10t10 clips and found error roughly
flat across them -- not enough to show periodicity, only flatness at those 7 points. This computes
RMSE continuously across h=0..max_h (covering >2 gait cycles at the measured period of 19 frames)
on the same dataset, to check for actual oscillation.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/periodicity_curve.py \\
        --ckpt wm/runs/beh12_hexonly_stopgrad/best.pt --data data/allocentric/fwd_m3d \\
        --cond c10f10t10 --max_h 45

Metric matches F26/F46: actions converted to degrees, ridge is multi-output (18 joints), RMSE is
sqrt(mean squared error) pooled over test samples and joints -- this recovers their reported
"command spread 11.33-11.34 deg" baseline (mean per-joint std in degrees) as a sanity check.
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
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cond", default="c10f10t10", help="restrict to this condition, as F26/F46 did")
    ap.add_argument("--max_h", type=int, default=45)
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--out", default="results/deck/periodicity_curve.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/periodicity_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    # `fwd_m3d` clips carry no `condition` field (only `behavior="walk"` for all of them) --
    # the body geometry (e.g. c10f10t10) lives in the filename, written by the collector, not in
    # any data field `gather()` reads. Filter on the filename instead.
    clips = [c for c in clips if args.cond in c["path"]]
    if not clips:
        raise SystemExit(f"no clips matched --cond {args.cond} in filename; "
                          f"paths look like: {[c['path'] for c in clips][:3]}")
    print(f"{len(clips)} clips matching {args.cond}")

    uniq_ids = list(range(len(clips)))
    test_clips = set(uniq_ids[1::2])

    # 2026-09-18 fix: the window `range(1, T-h-1)` shrinks and shifts earlier as h grows, so larger
    # h was scored on a smaller, earlier-biased subset of starting frames than smaller h -- not the
    # same measurement repeated at different offsets. Hold the starting-frame range FIXED across
    # every h (sized for the largest h requested) so the only thing that changes between rows is h
    # itself, not which frames get scored.
    rows = []
    for h in range(0, args.max_h + 1):
        E, A, clip_id = [], [], []
        for ci, c in enumerate(clips):
            e = c["e"].float()
            a = np.rad2deg(c["a"].numpy() if torch.is_tensor(c["a"]) else c["a"])
            T = len(e)
            if T <= args.max_h + 1:
                continue
            for t in range(1, T - args.max_h - 1):
                E.append(e[t].flatten().half())
                A.append(a[t + h])
                clip_id.append(ci)
        if len(E) < 20:
            continue
        E = torch.stack(E)
        A = np.stack(A)
        clip_id = np.array(clip_id)

        te = np.array([c in test_clips for c in clip_id])
        tr = ~te
        folds = np.array([int(c) % 4 for c in clip_id[tr]])

        K = gram(E, E, device).numpy()
        r2, pred, alpha = ridge_r2(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], A[tr], A[te], folds)
        rmse = float(np.sqrt(((pred - A[te]) ** 2).mean()))
        rows.append((h, rmse, r2, int(te.sum())))
        print(f"h={h:3d}  n_test={te.sum():4d}  RMSE={rmse:.3f} deg  R2={r2:.3f}")

    print("\nh\tRMSE_deg")
    for h, rmse, r2, n in rows:
        print(f"{h}\t{rmse:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hs = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        plt.figure(figsize=(9, 4))
        plt.plot(hs, vals, marker="o", ms=3)
        for k in range(0, args.max_h + 1, 19):
            plt.axvline(k, color="gray", ls="--", lw=0.6)
        plt.xlabel("offset h (frames)")
        plt.ylabel("RMSE, deg (command spread ~11.3)")
        plt.title(f"Single-frame command recoverability vs. offset -- {args.cond}\n"
                  f"dashed lines mark multiples of the claimed 19-frame period")
        out_path = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nsaved {args.out}")
    except Exception as e:
        print(f"plot failed: {e}")


if __name__ == "__main__":
    main()
