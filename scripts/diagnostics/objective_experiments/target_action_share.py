"""Does predicting body motion instead of the embedding recover the action signal?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/target_action_share.py

The action is worth **+0.055 cosine** to the FTM's predicted embedding direction. That was measured
under a reconstruction target, which is dominated by appearance. This asks whether the *target* is
what drowns the action, by holding the inputs `(e_t, z)` fixed and swapping what is predicted.

    embedding    e_t+1 - e_t          360,448 dims, what `L_recon` optimises
    delta-state  body motion at t     forward / lateral / yaw, the shared Froude coordinate

For each target, kernel ridge is fitted on the pretraining body and scored on the held-out one from
three feature sets. **Dropping `z` is exactly the mean-`z` arm** -- a constant feature contributes
nothing to a ridge -- so `e_t alone` is the no-action control and the difference is the action's
contribution.

    action share = ( R2[e_t, z] - R2[e_t] ) / R2[e_t, z]

is reported because raw R2 is not comparable across targets of different dimension; the share is the
fraction of what the model explains that the action is responsible for.

`z alone -> delta-state` is the third question: how much body motion the latent carries by itself,
which bounds the action signal in the inputs independently of what is predicted. The trained body
head is scored the same way beside it, since that head is exactly this mapping.

**Diagnosis only; trains nothing neural.**
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
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


def gram(a, b, device, chunk=64, bchunk=256):
    out = torch.empty(len(a), len(b), dtype=torch.float64)
    for i in range(0, len(a), chunk):
        q = a[i:i + chunk].to(device).float()
        for j in range(0, len(b), bchunk):
            out[i:i + chunk, j:j + bchunk] = (
                q @ b[j:j + bchunk].to(device).float().T).double().cpu()
        del q
        torch.cuda.empty_cache()
    return out.numpy()


def collect(clips, paths, itm, device, stride, channels):
    E, Z, D, B, cid = [], [], [], [], []
    with torch.no_grad():
        for ci, (c, p) in enumerate(zip(clips, paths)):
            bm = np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, channels]
            e = c["e"].float()
            for t in range(1, len(e) - 2, stride):
                if t >= len(bm):
                    continue
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                Z.append(itm(e_t, e1)[0].float().cpu())
                E.append(e[t].flatten().half())
                D.append((e1 - e_t).flatten().half().cpu())
                B.append(torch.tensor(bm[t], dtype=torch.float32))
                cid.append(ci)
    return (torch.stack(E), torch.stack(Z), torch.stack(D), torch.stack(B), np.array(cid))


def r2_small(K_tr, K_te, y_tr, y_te, alpha):
    w = np.linalg.solve(K_tr + alpha * np.eye(len(K_tr)), y_tr)
    pred = K_te @ w
    ss = ((pred - y_te) ** 2).sum()
    return float(1 - ss / max(((y_te - y_tr.mean(0)) ** 2).sum(), 1e-12))


def r2_gram(K_tr, K_te, G_tr, G_te, nrm_te, alpha):
    """R^2 for a very wide target, entirely through Gram matrices."""
    c = K_te @ np.linalg.inv(K_tr + alpha * np.eye(len(K_tr)))
    pp = np.einsum("ij,jk,ik->i", c, G_tr, c)
    pd = np.einsum("ij,ij->i", c, G_te)
    ss_res = pp - 2 * pd + nrm_te
    ybar_d = G_te.mean(1)
    ybar_n = G_tr.mean()
    ss_tot = nrm_te - 2 * ybar_d + ybar_n
    return float(1 - ss_res.sum() / max(ss_tot.sum(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--train_stride", type=int, default=2)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"hexapod": 18}).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    for m in (itm, ftm, md):
        for p in m.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    lag = max(1, cfg.action_lag)
    import glob
    tr_paths = sorted(glob.glob(os.path.join(ROOT, args.train_data, "*.npz")))
    te_paths = sorted(glob.glob(os.path.join(ROOT, args.test_data, "*.npz")))
    tr = gather(os.path.join(ROOT, args.train_data), "hexapod", None, ck, cache, 2, lag, device)
    te = gather(os.path.join(ROOT, args.test_data), "hexapod", None, ck, cache, 2, lag, device)
    E_tr, Z_tr, D_tr, B_tr, cid_tr = collect(tr, tr_paths, itm, device, args.train_stride, channels)
    E_te, Z_te, D_te, B_te, _ = collect(te, te_paths, itm, device, args.stride, channels)
    print(f"{args.ckpt}\ntrain {len(B_tr)} transitions, test {len(B_te)}, "
          f"body channels {channels}\n", flush=True)

    mean_e = E_tr.float().mean(0, keepdim=True)
    for i in range(0, len(E_tr), 256):
        E_tr[i:i + 256] = (E_tr[i:i + 256].float() - mean_e).half()
    for i in range(0, len(E_te), 256):
        E_te[i:i + 256] = (E_te[i:i + 256].float() - mean_e).half()

    print("  building Gram matrices", flush=True)
    raw = gram(E_tr, E_tr, device); s_e = max(np.mean(np.diag(raw)), 1e-12)
    Kee_tr, Kee_te = raw / s_e, gram(E_te, E_tr, device) / s_e
    zmu = Z_tr.numpy().astype(np.float64).mean(0, keepdims=True)   # centre both by the train mean
    zt = Z_tr.numpy().astype(np.float64) - zmu
    zq = Z_te.numpy().astype(np.float64) - zmu
    Kzz_tr = zt @ zt.T; s_z = max(np.mean(np.diag(Kzz_tr)), 1e-12)
    Kzz_tr, Kzz_te = Kzz_tr / s_z, (zq @ zt.T) / s_z

    G_tr = gram(D_tr, D_tr, device)
    G_te = gram(D_te, D_tr, device)
    nrm_te = np.array([float(D_te[i].float().pow(2).sum()) for i in range(len(D_te))])

    clips_tr = sorted(set(cid_tr.tolist()))
    val = set(clips_tr[1::3])
    va = np.array([c in val for c in cid_tr]); fit = ~va

    y_tr = B_tr.numpy().astype(np.float64)
    y_te = B_te.numpy().astype(np.float64)
    mu, sd = y_tr[fit].mean(0), y_tr[fit].std(0) + 1e-9
    ys_tr, ys_te = (y_tr - mu) / sd, (y_te - mu) / sd

    feats = {"e_t alone  (= mean-z control)": (Kee_tr, Kee_te),
             "z alone": (Kzz_tr, Kzz_te),
             "[e_t, z]": (Kee_tr + Kzz_tr, Kee_te + Kzz_te)}

    print(f"  {'features':>30}{'R2 embedding':>15}{'R2 delta-state':>17}")
    got = {}
    for name, (Ktr, Kte) in feats.items():
        best_e = max(ALPHAS, key=lambda a: r2_gram(
            Ktr[np.ix_(fit, fit)], Ktr[np.ix_(va, fit)],
            G_tr[np.ix_(fit, fit)], G_tr[np.ix_(va, fit)],
            np.array([G_tr[i, i] for i in np.where(va)[0]]), a))
        re = r2_gram(Ktr, Kte, G_tr, G_te, nrm_te, best_e)
        best_b = max(ALPHAS, key=lambda a: r2_small(
            Ktr[np.ix_(fit, fit)], Ktr[np.ix_(va, fit)], ys_tr[fit], ys_tr[va], a))
        rb = r2_small(Ktr, Kte, ys_tr, ys_te, best_b)
        got[name] = (re, rb)
        print(f"  {name:>30}{re:>15.3f}{rb:>17.3f}", flush=True)

    e_only = got["e_t alone  (= mean-z control)"]
    both = got["[e_t, z]"]
    print(f"\n  action contribution (with z minus without z)")
    print(f"    embedding target  : {both[0] - e_only[0]:+.3f}   "
          f"share {(both[0] - e_only[0]) / max(both[0], 1e-9):.1%}")
    print(f"    delta-state target: {both[1] - e_only[1]:+.3f}   "
          f"share {(both[1] - e_only[1]) / max(both[1], 1e-9):.1%}")

    # the trained models' own sensitivity, for reference
    with torch.no_grad():
        zb = Z_te.mean(0, keepdim=True).to(device)
        real = md.body(None, Z_te.to(device)).cpu().numpy()
        meanz = md.body(None, zb.expand(len(Z_te), -1)).cpu().numpy()
    bs = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)], \
        np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]
    tgt = (y_te - bs[0]) / bs[1]
    def r2(p):
        return float(1 - ((p - tgt) ** 2).sum() / max(((tgt - tgt.mean(0)) ** 2).sum(), 1e-12))
    print(f"\n  trained body head, z -> delta-state (held out)")
    print(f"    real z {r2(real):+.3f}   mean z {r2(meanz):+.3f}   "
          f"gap {r2(real) - r2(meanz):+.3f}")


if __name__ == "__main__":
    main()
