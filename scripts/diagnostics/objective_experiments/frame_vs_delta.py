"""Does feeding the state head the raw frame instead of the FTM's delta reopen F64's identity leak?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/frame_vs_delta.py

The state-head design reads `FTM(e_t,z) - e_t`, not `e_t`, specifically to avoid the failure
`body_sees_frame=True` hit once already: a frame-conditioned head learns one mapping per robot and
stops needing the latent, collapsing cross-embodiment transfer from +0.544/+0.435 to -10.5/-57.2
(F64). This checks the analogous risk for the new head before spending a com7 run on it.

Two ridge fits, same [z] block, only the other input swapped -- same protocol as
`target_action_share.py`, so the numbers sit beside its 0.358/0.359/0.404.

    [e_t, z]              raw frame + z            the configuration F64 found unsafe
    [e_t+1 - e_t, z]       predicted delta + z       the proposed state-head input

**What would make the delta design unsafe:** if `[e_t, z]` scores much higher than
`[delta, z]` for the WRONG reason -- because `e_t` lets the readout identify which embodiment it is
looking at, the same shortcut F64 took. That is checked directly with a third fit on **which body
this is**: if `e_t` predicts embodiment far better than `delta` does, the raw-frame route has the
leak available; if the two are close, the frame is not the thing supplying the advantage, if any.

`e_t+1` is the FTM's own prediction (`FTM(e_t, z)`), computed here rather than reusing the true next
frame, since that is what the state head sees during training.

**Diagnosis only; trains nothing.**
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

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


def unit(K):
    return K / max(np.mean(np.diag(K)), 1e-12)


def fit_eval(K_all, y, tr, va, te, alphas=ALPHAS):
    best_a, best_v = alphas[0], -1e18
    for a in alphas:
        w = np.linalg.solve(K_all[np.ix_(tr, tr)] + a * np.eye(tr.sum()), y[tr])
        pred = K_all[np.ix_(va, tr)] @ w
        ss = ((pred - y[va]) ** 2).sum()
        v = 1 - ss / max(((y[va] - y[tr].mean(0)) ** 2).sum(), 1e-12)
        if v > best_v:
            best_v, best_a = v, a
    w = np.linalg.solve(K_all[np.ix_(tr, tr)] + best_a * np.eye(tr.sum()), y[tr])
    pred = K_all[np.ix_(te, tr)] @ w
    ss = ((pred - y[te]) ** 2).sum()
    r2 = 1 - ss / max(((y[te] - y[tr].mean(0)) ** 2).sum(), 1e-12)
    return r2, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--extra_cache", default="results/wm/cache/ego_b1.pt")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    E, D, Z, Y, EID, cid = [], [], [], [], [], []
    cache1 = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    cache2 = torch.load(os.path.join(ROOT, args.extra_cache), map_location="cpu", mmap=True) \
        if os.path.exists(os.path.join(ROOT, args.extra_cache)) else {}
    offset = 0
    for eid, spec in enumerate(args.sources):
        name, path = spec.split("=", 1)
        cache = cache1 if name == "hexapod" else cache2
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        with torch.no_grad():
            for ci, (c, p) in enumerate(zip(clips, paths)):
                bm = np.asarray(load(p, REGISTRY[name])["body_motion"])[:, channels]
                e = c["e"].float()
                for t in range(1, min(len(e) - 2, len(bm)), args.stride):
                    e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                    z = itm(e_t, e1)
                    pred = ftm(e_t, z)
                    delta = (pred - e_t)[0].flatten().half().cpu()
                    E.append(e[t].flatten().half())
                    D.append(delta)
                    Z.append(z[0].float().cpu())
                    Y.append(torch.tensor(bm[t], dtype=torch.float64))
                    EID.append(eid)
                    cid.append(offset + ci)
        offset += len(clips)
        print(f"  {name}: {len(clips)} clips, {sum(1 for x in EID if x == eid)} transitions",
              flush=True)

    E, D, Z = torch.stack(E), torch.stack(D), torch.stack(Z)
    Y = torch.stack(Y).numpy()
    EID = np.array(EID); cid = np.array(cid)
    n = len(Y)
    print(f"\n{args.ckpt}\n{n} transitions across {len(args.sources)} embodiments\n")

    clips_all = sorted(set(cid.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(clips_all)
    n_va = max(1, len(clips_all) // 5)
    va_clips, te_clips = set(clips_all[:n_va]), set(clips_all[n_va:2 * n_va])
    va = np.array([c in va_clips for c in cid])
    te = np.array([c in te_clips for c in cid])
    tr = ~(va | te)

    mean_e = E[tr].float().mean(0, keepdim=True)
    for i in range(0, len(E), 256):
        E[i:i + 256] = (E[i:i + 256].float() - mean_e).half()
    mean_d = D[tr].float().mean(0, keepdim=True)
    for i in range(0, len(D), 256):
        D[i:i + 256] = (D[i:i + 256].float() - mean_d).half()
    zmu = Z[tr].numpy().astype(np.float64).mean(0, keepdims=True)
    Zc = Z.numpy().astype(np.float64) - zmu

    print("  building Gram matrices", flush=True)
    Kee = unit(gram(E, E, device))
    Kdd = unit(gram(D, D, device))
    Kzz = unit(Zc @ Zc.T)

    y = (Y - Y[tr].mean(0)) / (Y[tr].std(0) + 1e-9)
    eid_y = np.eye(len(args.sources))[EID].astype(np.float64)   # one-hot embodiment target

    print(f"  {'target':>26}{'features':>20}{'R2':>9}{'alpha':>9}")
    for tgt_name, tgt in (("body motion", y), ("embodiment id (one-hot)", eid_y)):
        for feat_name, K in (("[e_t, z]  (raw frame)", Kee + Kzz),
                             ("[delta, z]  (proposed)", Kdd + Kzz),
                             ("z alone  (reference)", Kzz)):
            r2, a = fit_eval(K, tgt, tr, va, te)
            print(f"  {tgt_name:>26}{feat_name:>20}{r2:>9.3f}{a:>9.4g}", flush=True)

    print("\n  READ: body-motion R2 should be close for [e_t,z] and [delta,z] (both carry state)."
          "\n  embodiment-id R2 should be LOW for [delta,z] and can be high for [e_t,z] -- that gap"
          "\n  is F64's leak. If [delta,z] also identifies embodiment well, the delta does not close"
          "\n  the leak and the design needs rethinking before com7.")


if __name__ == "__main__":
    main()
