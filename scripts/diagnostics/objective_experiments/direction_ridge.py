"""How much transition direction is recoverable from `(e_t, z)` by a strong non-neural predictor?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/direction_ridge.py

The FTM predicts the direction of `e_t+1 - e_t` at cosine 0.689 and no objective moves it. This asks
whether that is the FTM or the information in its inputs, using kernel ridge on the **same inputs**
-- a different function class, fitted in closed form, no neural training.

    e_t alone         how much direction the frame determines on its own
    z alone           how much the 64-d latent determines
    [e_t, z]          the FTM's own inputs
    [e_t, z_r]        z truncated to its top r principal components, to see whether direction
                      scales with the width of the latent code

**The read.** `[e_t, z]` near 0.689 means the ceiling is the information in the inputs, so neither a
larger FTM nor a different encoder recovers it. Clearly above 0.689 means the information is there
and the FTM is not extracting it, which is a capacity result. And if the number climbs with `z`'s
width, the 64-d code is the throttle rather than either.

**Everything is computed through Gram matrices**, so a 360,448-dimensional prediction is never
materialised: with unit-norm direction targets `Y`, the ridge prediction is `pred_i = sum_j c_ij Y_j`
and both the numerator `<pred_i, d_i>` and the norm `||pred_i||` follow from the train-train and
test-train direction cosines alone.

Alpha is chosen on a split of the training body, never on the held-out one.
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

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def collect(clips, itm, ftm, device, stride):
    E, Z, D, cos, cid = [], [], [], [], []
    with torch.no_grad():
        for ci, c in enumerate(clips):
            e = c["e"].float()
            for t in range(1, len(e) - 2, stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                p = ftm(e_t, z)
                dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
                cos.append(F.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)).item())
                E.append(e[t].flatten().half())
                Z.append(z[0].float().cpu())
                D.append(F.normalize(dt, dim=0).half().cpu())
                cid.append(ci)
    return torch.stack(E), torch.stack(Z), torch.stack(D), np.array(cos), np.array(cid)


def gram(a, b, device, chunk=64, bchunk=256):
    out = torch.empty(len(a), len(b), dtype=torch.float64)
    for i in range(0, len(a), chunk):
        q = a[i:i + chunk].to(device).float()
        for j in range(0, len(b), bchunk):
            out[i:i + chunk, j:j + bchunk] = (q @ b[j:j + bchunk].to(device).float().T).double().cpu()
        del q
        torch.cuda.empty_cache()
    return out.numpy()


def unit(K):
    return K / max(np.mean(np.diag(K)), 1e-12)


def ridge_cos(K_tr, K_qu, G_tr, G_qu, alpha):
    """Median cosine of the kernel-ridge prediction, computed entirely in Gram space."""
    c = K_qu @ np.linalg.inv(K_tr + alpha * np.eye(len(K_tr)))
    num = np.einsum("ij,ij->i", c, G_qu)                  # <pred_i, d_i>
    nrm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", c, G_tr, c), 1e-12))
    return float(np.median(num / nrm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--train_stride", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    lag = max(1, cfg.action_lag)
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    tr = gather(os.path.join(ROOT, args.train_data), args.embodiment, None, ck, cache, 2, lag,
                device)
    te = gather(os.path.join(ROOT, args.test_data), args.embodiment, None, ck, cache, 2, lag,
                device)
    E_tr, Z_tr, D_tr, _c, cid_tr = collect(tr, itm, ftm, device, args.train_stride)
    E_te, Z_te, D_te, cos_ftm, _ = collect(te, itm, ftm, device, args.stride)
    print(f"{args.ckpt}\ntrain {len(D_tr)} transitions, test {len(D_te)}", flush=True)
    print(f"FTM held-out direction cosine: {np.median(cos_ftm):.3f}\n", flush=True)

    mean_e = E_tr.float().mean(0, keepdim=True)
    for i in range(0, len(E_tr), 256):
        E_tr[i:i + 256] = (E_tr[i:i + 256].float() - mean_e).half()
    for i in range(0, len(E_te), 256):
        E_te[i:i + 256] = (E_te[i:i + 256].float() - mean_e).half()

    print("  building Gram matrices", flush=True)
    G_tr = gram(D_tr, D_tr, device)                     # direction targets, unit norm
    G_te = gram(D_te, D_tr, device)
    Kee_tr, Kee_te = unit(gram(E_tr, E_tr, device)), None
    Kee_te = gram(E_te, E_tr, device) / max(np.mean(np.diag(gram(E_tr[:1], E_tr[:1], device))), 1e-12)
    # recompute consistently: normalise both by the train diagonal mean
    raw_tr = gram(E_tr, E_tr, device)
    scale = max(np.mean(np.diag(raw_tr)), 1e-12)
    Kee_tr = raw_tr / scale
    Kee_te = gram(E_te, E_tr, device) / scale

    Zt, Zq = Z_tr.numpy().astype(np.float64), Z_te.numpy().astype(np.float64)
    mu = Zt.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zt - mu, full_matrices=False)

    # held-out split of the TRAINING body, by clip, for choosing alpha
    clips_tr = sorted(set(cid_tr.tolist()))
    val_clips = set(clips_tr[1::3])
    va = np.array([c in val_clips for c in cid_tr]); fit = ~va

    def evaluate(name, Ktr_full, Kte_full):
        best_a, best_v = ALPHAS[0], -9
        for a in ALPHAS:
            v = ridge_cos(Ktr_full[np.ix_(fit, fit)], Ktr_full[np.ix_(va, fit)],
                          G_tr[np.ix_(fit, fit)], G_tr[np.ix_(va, fit)], a)
            if v > best_v:
                best_v, best_a = v, a
        t = ridge_cos(Ktr_full, Kte_full, G_tr, G_te, best_a)
        print(f"  {name:>34}{t:>10.3f}{best_v:>12.3f}{best_a:>10.4g}", flush=True)
        return t

    print(f"\n  {'ridge from':>34}{'held-out':>10}{'train-val':>12}{'alpha':>10}")
    evaluate("e_t alone", Kee_tr, Kee_te)
    for r in args.ranks:
        P = Vt[:r].T
        zt, zq = (Zt - mu) @ P, (Zq - mu) @ P
        Kzz_tr, Kzz_te = zt @ zt.T, zq @ zt.T
        s = max(np.mean(np.diag(Kzz_tr)), 1e-12)
        if r == max(args.ranks):
            evaluate(f"z alone  (top {r} PCs)", Kzz_tr / s, Kzz_te / s)
        evaluate(f"[e_t, z]  z at {r} PCs", Kee_tr + Kzz_tr / s, Kee_te + Kzz_te / s)

    print(f"\n  the FTM gets {np.median(cos_ftm):.3f} from the same inputs.")


if __name__ == "__main__":
    main()
