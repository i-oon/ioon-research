"""Redo of frame_vs_delta.py's embodiment-leak check with a classifier, not a ridge regressor.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/frame_vs_delta_classify.py

`frame_vs_delta.py`'s one-hot ridge regression returned a NEGATIVE R2 for embodiment identity from
**`z` alone** -- worse than predicting the mean. That contradicts this project's own established
result (`identity_linearity.py`, session start): logistic regression on `z` reads embodiment at
**1.000** raw. A probe that cannot see a leak known to exist is not a probe that clears anything;
this repeats the check with the same tool that found it the first time.

**`e_t` and `delta` are reduced to their top principal components before fitting, via the dual.**
Two OOM kills so far: raw logistic regression at 360,448 features first (`cross_val_score(n_jobs=-1)`
copies the array into every fold's worker), then `np.linalg.svd` on the raw (1532, 360448) matrix
second -- the factorisation itself needs workspace on that order before a single fold is fit. The
fix used everywhere else in this session for exactly this shape: `n=1532 << d=360448`, so the
eigendecomposition of the `n x n` Gram matrix gives the same top components as the full SVD
(`G = X X^T = U S^2 U^T`) without ever forming a `d`-dimensional factorisation. `gram()` is the same
GPU-chunked routine `residual_structure.py` and its callers already use.
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import gram  # noqa: E402


def top_pcs(X, r, device):
    """Scores on the top `r` principal components, via the n x n Gram matrix (n << d here)."""
    mean = X.float().mean(0, keepdim=True)
    Xc = (X.float() - mean).half()
    G = gram(Xc, Xc, device).numpy()               # n x n, float64, on CPU
    vals, vecs = np.linalg.eigh(G)                 # ascending
    order = np.argsort(vals)[::-1][:r]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    return vecs * np.sqrt(vals)[None, :]           # == U[:, :r] * S[:r] from the SVD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--extra_cache", default="results/wm/cache/ego_b1.pt")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 32, 128])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    E, D, Z, EID = [], [], [], []
    cache1 = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    cache2 = torch.load(os.path.join(ROOT, args.extra_cache), map_location="cpu", mmap=True)
    for eid, spec in enumerate(args.sources):
        name, path = spec.split("=", 1)
        cache = cache1 if name == "hexapod" else cache2
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        with torch.no_grad():
            for c in clips:
                e = c["e"].float()
                for t in range(1, len(e) - 2, args.stride):
                    e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                    z = itm(e_t, e1)
                    pred = ftm(e_t, z)
                    E.append(e[t].flatten().half())
                    D.append((pred - e_t)[0].flatten().half().cpu())
                    Z.append(z[0].float().cpu())
                    EID.append(eid)
        print(f"  {name}: {len(clips)} clips", flush=True)

    E, D, Z = torch.stack(E), torch.stack(D), torch.stack(Z)
    EID = np.array(EID)
    print(f"\n{args.ckpt}\n{len(EID)} transitions, classes {np.bincount(EID)}\n", flush=True)

    def clf():
        return LogisticRegression(max_iter=500)

    print(f"  {'features':>34}{'embodiment accuracy':>22}")
    for r in args.ranks:
        for label, X in (("e_t", E), ("delta", D)):
            Xr = top_pcs(X, r, device)
            acc = cross_val_score(clf(), Xr, EID, cv=5, n_jobs=1).mean()
            print(f"  {f'{label}, top {r} PCs':>34}{acc:>22.3f}", flush=True)
    zacc = cross_val_score(clf(), Z.numpy(), EID, cv=5, n_jobs=1).mean()
    print(f"  {'z  (reference, known ~1.0)':>34}{zacc:>22.3f}", flush=True)

    print(f"\n  chance: {max(np.bincount(EID)) / len(EID):.3f}")
    print("\n  READ: z should read near 1.0, matching identity_linearity.py -- if it does not, this")
    print("  script's probe is also insensitive and the whole check needs a different method.")
    print("  If e_t reads high and delta reads near chance, the delta design closes the leak.")
    print("  If delta ALSO reads high, delta does not close it -- rethink before com7.")


if __name__ == "__main__":
    main()
