"""How much does the *transition* add over a single frame, for reading the action?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/inverse_dynamics_r2.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/best.pt --data data/allocentric/beh12_c08f09t09_flat \\
        --embodiment hexapod

**Stated in the metric Yeom et al. (2606.07687) use.** They show V-JEPA carries
inverse-dynamics-recoverable action structure -- R2 0.40 frozen, 0.85 with an ID head -- and observe
that CALVIN's static tabletop lets per-frame appearance stand in for temporal context. **This asks
how far that goes in legged locomotion**, where a gait makes the pose itself an almost complete
statement of the command.

Three feature sets, one ridge, one split:

    e_t                 a single frame
    [e_t, e_t+1]        the frame pair -- their inverse-dynamics setup
    [e_t, e_t+3]        a wider pair, in case spacing is what separates them

**If the single frame matches the pair, the action is already in the pose** and the transition
carries almost nothing extra. That is a stronger statement than "the residual is noise" (F158),
because it locates the cause: **periodicity, not a weak model.**

**This measurement barely depends on the checkpoint.** The features are frozen encoder embeddings;
only the action normalisation and chunking come from the loaded config. A model-independent result
is the point -- it is a property of V-JEPA2 on this data, not of anything we trained.

Ridge is solved in the dual, so a concatenated pair costs one added Gram matrix and no projection.
Splits are **by clip**.
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
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--pair_lags", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--target", choices=("action", "z"), default="action",
                    help="**`action` is F159's question, `z` is the one never asked.** `z` is "
                         "`ITM(e_t, e_t+1)`, built from two frames by construction, and our "
                         "pipeline then pushes it toward the action from both sides -- the "
                         "projector fits `proj(a) ~ z` and the body head fits `z -> motion`. So a "
                         "redundant `z` could mean the transition itself is pose-determined, or "
                         "only that we forced `z` to be the action. Reading it from a single frame "
                         "separates those. **Unlike the action target this depends on the "
                         "checkpoint**, since `z` is that checkpoint's latent.")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = None
    if args.target == "z":
        itm = InverseTransitionModel(cfg).to(device).eval()
        itm.load_state_dict(ck["itm"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
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

    reach = max(args.pair_lags)
    cols = collections.defaultdict(list)
    A, fam, clip_id = [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < reach + 3:
            continue
        for t in range(1, len(e) - reach - 1, args.stride):
            cols[0].append(e[t].flatten().half())
            for k in args.pair_lags:
                cols[k].append(e[t + k].flatten().half())
            if itm is None:
                A.append(c["a"][t].flatten().float())
            else:
                with torch.no_grad():
                    A.append(itm(e[t:t + 1].to(device),
                                 e[t + 1:t + 2].to(device))[0].flatten().float().cpu())
            fam.append(FAMILY(c["cond"]))
            clip_id.append(ci)
    cols = {k: torch.stack(v) for k, v in cols.items()}
    A = torch.stack(A).numpy()
    fam = np.array(fam); clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id])
    tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    A = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip, "
          f"{args.target} width {A.shape[1]}, embedding dimension {cols[0].shape[1]}\n")

    # a linear kernel on a concatenation is the sum of the parts' kernels
    K = {k: gram(v, v, device).numpy() for k, v in cols.items()}
    feats = {"e_t  (single frame)": K[0]}
    for k in args.pair_lags:
        feats[f"[e_t, e_t+{k}]  (pair)"] = K[0] + K[k]

    preds = {}
    print(f"  {'features':>26}{args.target + ' R2':>11}{'vs single':>11}{'alpha':>9}")
    base = None
    for name, Kf in feats.items():
        r2, pred, alpha = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], A[tr], A[te], folds)
        preds[name] = pred
        if base is None:
            base = r2
        print(f"  {name:>26}{r2:>11.3f}{r2 - base:>+11.3f}{alpha:>9.4g}")

    print(f"\n  per family, action R2 on held-out clips")
    print(f"  {'family':>10}" + "".join(f"{n:>26}" for n in feats) + f"{'n':>7}")
    for f in sorted(set(fam[te])):
        m = fam[te] == f
        row = ""
        for name in feats:
            ss = ((preds[name][m] - A[te][m]) ** 2).sum()
            row += f"{1 - ss / max(((A[te][m] - A[tr].mean(0)) ** 2).sum(), 1e-9):>26.3f}"
        print(f"  {f:>10}{row}{m.sum():>7}")

    gains = [r2 for name, r2 in
             ((n, 1 - ((preds[n] - A[te]) ** 2).sum() /
               max(((A[te] - A[tr].mean(0)) ** 2).sum(), 1e-9)) for n in feats)][1:]
    best = max(gains) - base if gains else 0.0
    print(f"\n  the widest pair beats a single frame by {best:+.3f} R2.  "
          + ("**the action is already in the pose** -- the transition adds almost nothing, so "
             "inverse-recoverable does not imply forward-necessary" if best < 0.05 else
             "the transition carries action information a single frame does not"))


if __name__ == "__main__":
    main()
