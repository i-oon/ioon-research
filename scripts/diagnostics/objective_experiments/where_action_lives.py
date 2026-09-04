"""Is the action stamped into the FTM's output, already in the frame, or carried by `z`?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/where_action_lives.py \\
        --ckpt wm/runs/beh12_ego/teacher_ego.pt \\
        --data data/egocentric/beh12_c08f09t09_ego_flat \\
        --embodiment hexapod --cache results/wm/cache/ego_hex.pt

**Separates three places the action could live**, which the fine-ranking failure (F179, 47%) cannot
distinguish on its own:

    e_t                     the frame alone            -- F159's question, 0.779 allocentric
    e_t+1 (ground truth)    the true next frame
    FTM(e_t, z)             the model's prediction     -- does the FTM stamp the action in?
    z = ITM(e_t, e_t+1)     the latent
    e_t -> z                is z single-frame readable -- F168's question, 0.856 allocentric

**The read.** Ground truth low and prediction high means the FTM writes the action into its output
rather than predicting a future that happens to contain it -- the shortcut sits in the objective.
Ground truth already high means the frame carries the action before any model touches it, and the
problem is the encoder/data, not the objective. Similar means the FTM passes through what is there.

**`[e_t, z]` is the control and it is not optional.** `FTM(e_t, z)` is a deterministic function of
its two inputs, so nothing decodable from its output can exceed what is decodable from the pair.
Without that row a high number on the prediction reads as "the FTM stamped it" when it may only be
"the FTM passed `z` through", and `z` carries the action heavily by construction -- the projector
fits `proj(a) ~ z` from one side and the body head fits `z -> motion` from the other.

Each feature block's kernel is normalised by its mean diagonal before any are summed, because `z` is
64 numbers against the embedding's 360,448 and an unnormalised sum is the embedding alone. Single
blocks are unaffected: scaling a kernel is a shift in the ridge penalty, which is cross-validated.

Same ridge, same clip-level split and same `gather` as `inverse_dynamics_r2.py`, so these numbers sit
beside F159's and F168's rather than needing their own protocol. **Diagnosis only; trains nothing.**
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
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_clips", type=int, default=0, help="0 uses every clip")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu", mmap=True)
    before = len(cache)
    # every clip of this set is already encoded, so no encoder is constructed; gather only calls one
    # for a cache miss, and a miss here would mean the wrong cache was named
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, None, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        raise SystemExit(f"{len(cache) - before} clips were missing from {args.cache}")
    if args.max_clips:
        clips = clips[:args.max_clips]

    cols = collections.defaultdict(list)
    A, Z, fam, clip_id = [], [], [], []
    with torch.no_grad():
        for ci, c in enumerate(clips):
            e = c["e"].float()
            if len(e) < 4:
                continue
            for t in range(1, len(e) - 2, args.stride):
                e_t = e[t:t + 1].to(device)
                e_next = e[t + 1:t + 2].to(device)
                z = itm(e_t, e_next)
                pred = ftm(e_t, z)
                cols["e_t"].append(e[t].flatten().half())
                cols["e_next_gt"].append(e[t + 1].flatten().half())
                cols["pred"].append(pred[0].flatten().half().cpu())
                cols["pred_delta"].append((pred[0] - e_t[0]).flatten().half().cpu())
                cols["z"].append(z[0].float().cpu())
                Z.append(z[0].float().cpu())
                A.append(c["a"][t].flatten().float())
                fam.append(FAMILY(c["cond"]))
                clip_id.append(ci)

    cols = {k: torch.stack(v) for k, v in cols.items()}
    A = torch.stack(A).numpy()
    Z = torch.stack(Z).numpy()
    fam = np.array(fam); clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id])
    tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip, "
          f"action width {A.shape[1]}, z width {Z.shape[1]}, "
          f"embedding dimension {cols['e_t'].shape[1]}\n")

    K = {}
    for name, v in cols.items():
        g = gram(v, v, device).numpy()
        K[name] = g / max(np.mean(np.diag(g)), 1e-12)   # unit mean diagonal, so blocks can be summed
    del cols

    def score(Kf, y):
        y = (y - y[tr].mean(0)) / (y[tr].std(0) + 1e-6)
        r2, _pred, alpha = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], y[tr], y[te], folds)
        return r2, alpha

    print("  MEASUREMENT 1 & 2 -- decoding the ACTION")
    print(f"  {'features':>34}{'action R2':>11}{'alpha':>9}")
    rows = [
        ("e_t  (frame alone)", K["e_t"]),
        ("e_t+1  GROUND TRUTH", K["e_next_gt"]),
        ("FTM(e_t, z)  PREDICTED", K["pred"]),
        ("FTM(e_t, z) - e_t  (delta)", K["pred_delta"]),
        ("z = ITM(e_t, e_t+1)", K["z"]),
        ("[e_t, z]  ceiling for the prediction", K["e_t"] + K["z"]),
    ]
    got = {}
    for name, Kf in rows:
        r2, alpha = score(Kf, A.copy())
        got[name] = r2
        print(f"  {name:>34}{r2:>11.3f}{alpha:>9.4g}")

    print("\n  MEASUREMENT 2b -- is z itself readable from one frame  (F168 asked this)")
    print(f"  {'features':>34}{'z R2':>11}{'alpha':>9}")
    for name, Kf in (("e_t  (frame alone)", K["e_t"]),
                     ("e_t+1  GROUND TRUTH", K["e_next_gt"])):
        r2, alpha = score(Kf, Z.copy())
        print(f"  {name:>34}{r2:>11.3f}{alpha:>9.4g}")

    print(f"\n  per family, action R2 on held-out clips")
    hdr = ("e_t", "e_t+1 gt", "FTM pred", "z")
    keys = ("e_t", "e_next_gt", "pred", "z")
    print(f"  {'family':>10}" + "".join(f"{h:>12}" for h in hdr) + f"{'n':>7}")
    for f in sorted(set(fam[te])):
        m = fam[te] == f
        row = ""
        for k in keys:
            y = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)
            r2, _p, _a = ridge_r2(K[k][np.ix_(tr, tr)], K[k][np.ix_(te, tr)][m], y[tr], y[te][m],
                                  folds)
            row += f"{r2:>12.3f}"
        print(f"  {f:>10}{row}{m.sum():>7}")

    gt, pr = got["e_t+1  GROUND TRUTH"], got["FTM(e_t, z)  PREDICTED"]
    ceil = got["[e_t, z]  ceiling for the prediction"]
    print(f"\n  prediction - ground truth: {pr - gt:+.3f}   "
          f"prediction - its own ceiling: {pr - ceil:+.3f}")


if __name__ == "__main__":
    main()
