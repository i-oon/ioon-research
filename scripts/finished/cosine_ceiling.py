"""Is the FTM's transition-direction cosine of 0.690 near the achievable ceiling, or is there room?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/cosine_ceiling.py --mode ceiling
    .venv/bin/python3 scripts/diagnostics/objective_experiments/cosine_ceiling.py --mode knn
    .venv/bin/python3 scripts/diagnostics/objective_experiments/cosine_ceiling.py --mode derisk
    .venv/bin/python3 scripts/diagnostics/objective_experiments/cosine_ceiling.py --mode ridge

**Merged 2026-09-11 from four scripts that were one investigation** (`cosine_ceiling.py`,
`direction_ceiling_knn.py`, `direction_derisk.py`, `direction_ridge.py`) into one file with a
`--mode` flag, since they shared setup and no single one was "the" finding -- unlike a script with a
self-declared successor, these four run side by side to bracket the same question from different
angles. Each mode's own computation is kept verbatim (only renamed/namespaced), not rewritten, to
avoid introducing a bug in working numerical diagnostic code for the sake of fewer files.

`undermovement.py` reduced the fine-ranking and long-horizon wall to one number: the FTM predicts the
direction of `e_t+1 - e_t` at cosine **0.690**, and its under-movement is the correct MSE response to
that (`alpha*` 0.971). Whether that number is fixable decides whether anything is worth rebuilding.

    --mode ceiling   (was cosine_ceiling.py) Four cheap references, none trained: the mean training
                     displacement (the floor that matters), k-NN on e_t, k-NN on [e_t, z], and the
                     FTM re-evaluated on its OWN training body (train ~ held-out means a capacity/
                     objective limit, not a generalisation gap).
    --mode knn       (was direction_ceiling_knn.py) Separates "capacity" (a smooth predictor beats
                     0.687) from "embedding limit" (nothing does): retrieval sanity, k-NN oracle,
                     best-match ceiling (the single closest training direction anywhere), and
                     neighbour spread (do near-identical states move consistently at all).
    --mode derisk    (was direction_derisk.py) Actually TRAINS: fine-tunes the FTM with an added
                     `lambda * (1 - cos(...))` term to see if asking for direction directly moves it.
                     The only mode that writes gradients; still writes no checkpoint into wm/runs.
    --mode ridge     (was direction_ridge.py) Kernel ridge (closed-form, no neural training) on the
                     FTM's own inputs `(e_t, z)`, computed entirely through Gram matrices so the
                     360,448-d prediction is never materialised. Tests whether `[e_t, z]` alone
                     bounds the ceiling, and whether it climbs with z's PCA width.

**The read, common to all four.** Anything clearly above ~0.690 means headroom and the FTM/objective
is the binding constraint. Everything plateauing near 0.690 -- including the untrained constant in
`ceiling` mode -- means the ceiling is a property of V-JEPA2's latent geometry on this data, and no
objective or architecture recovers it. `ridge`/`knn`/`ceiling` diagnose only and train nothing;
`derisk` is the one mode that actually fine-tunes, to test the objective-side hypothesis directly.
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


def load_model_and_data(args, device, need_ftm=True):
    """Shared across all four modes: load the checkpoint, freeze it, gather train/test clips."""
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    lag = max(1, cfg.action_lag)
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    ftm = None
    if need_ftm:
        ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
        for p in ftm.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    tr = gather(os.path.join(ROOT, args.train_data), args.embodiment, None, ck, cache, 2, lag, device)
    te = gather(os.path.join(ROOT, args.test_data), args.embodiment, None, ck, cache, 2, lag, device)
    return ck, cfg, itm, ftm, tr, te


# ===================================================================================== mode: ceiling
def _ceiling_collect(clips, itm, ftm, device, stride):
    E, Z, D, cos = [], [], [], []
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            for t in range(1, len(e) - 2, stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                p = ftm(e_t, z)
                dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
                cos.append(F.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)).item())
                E.append(e[t].flatten().half())
                Z.append(z[0].float().cpu())
                D.append(dt.half().cpu())
    return (torch.stack(E), torch.stack(Z), torch.stack(D), np.array(cos))


def _ceiling_cosine_rows(a, b, device, chunk=64):
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk].to(device).float()
        y = b[i:i + chunk].to(device).float()
        out[i:i + chunk] = F.cosine_similarity(x, y, dim=1).cpu().numpy()
    return out


def _ceiling_knn_predict(feat_te, feat_tr, disp_tr, k, device, chunk=32):
    tr_n = F.normalize(feat_tr.float(), dim=1)
    pred = torch.empty(len(feat_te), disp_tr.shape[1], dtype=torch.float16)
    for i in range(0, len(feat_te), chunk):
        q = F.normalize(feat_te[i:i + chunk].to(device).float(), dim=1)
        sims = torch.empty(len(q), len(tr_n))
        for j in range(0, len(tr_n), 512):
            sims[:, j:j + 512] = (q @ tr_n[j:j + 512].to(device).T).cpu()
        idx = sims.topk(k, dim=1).indices
        for r in range(len(q)):
            pred[i + r] = disp_tr[idx[r]].float().mean(0).half()
        del q
        torch.cuda.empty_cache()
    return pred


def mode_ceiling(args, device):
    if args.ks is None:
        args.ks = [1, 5, 20]
    if args.stride is None:
        args.stride = 4
    _ck, _cfg, itm, ftm, tr_clips, te_clips = load_model_and_data(args, device)
    E_tr, Z_tr, D_tr, cos_tr = _ceiling_collect(tr_clips, itm, ftm, device, args.stride)
    E_te, Z_te, D_te, cos_te = _ceiling_collect(te_clips, itm, ftm, device, args.stride)
    print(f"{args.ckpt}")
    print(f"train {len(tr_clips)} clips / {len(D_tr)} transitions from {args.train_data}")
    print(f"test  {len(te_clips)} clips / {len(D_te)} transitions from {args.test_data}\n")

    print(f"  {'predictor of the transition direction':>44}{'cosine':>9}")
    print(f"  {'FTM, held-out body  (the 0.690)':>44}{np.median(cos_te):>9.3f}")
    print(f"  {'FTM, TRAINING body':>44}{np.median(cos_tr):>9.3f}")

    const = D_tr.float().mean(0, keepdim=True).half().expand(len(D_te), -1)
    print(f"  {'constant: mean training displacement':>44}"
          f"{np.median(_ceiling_cosine_rows(const, D_te, device)):>9.3f}")

    for k in args.ks:
        p = _ceiling_knn_predict(E_te, E_tr, D_tr, k, device)
        print(f"  {f'k-NN on e_t, k={k}':>44}{np.median(_ceiling_cosine_rows(p, D_te, device)):>9.3f}")
    feat_tr = torch.cat([F.normalize(E_tr.float(), dim=1), F.normalize(Z_tr, dim=1)], dim=1).half()
    feat_te = torch.cat([F.normalize(E_te.float(), dim=1), F.normalize(Z_te, dim=1)], dim=1).half()
    for k in args.ks:
        p = _ceiling_knn_predict(feat_te, feat_tr, D_tr, k, device)
        print(f"  {f'k-NN on [e_t, z], k={k}':>44}{np.median(_ceiling_cosine_rows(p, D_te, device)):>9.3f}")

    print(f"\n  train - held-out gap for the FTM: {np.median(cos_tr) - np.median(cos_te):+.3f}")


# ======================================================================================== mode: knn
def _knn_collect(clips, itm, ftm, device, stride):
    E, D, cos = [], [], []
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            for t in range(1, len(e) - 2, stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                p = ftm(e_t, itm(e_t, e1))
                dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
                cos.append(F.cosine_similarity(dp.unsqueeze(0), dt.unsqueeze(0)).item())
                E.append(e[t].flatten().half())
                D.append(F.normalize(dt, dim=0).half().cpu())
    return torch.stack(E), torch.stack(D), np.array(cos)


def _knn_cos_matrix(a, b, device, chunk=64, bchunk=256):
    out = torch.empty(len(a), len(b))
    for i in range(0, len(a), chunk):
        q = F.normalize(a[i:i + chunk].to(device).float(), dim=1)
        for j in range(0, len(b), bchunk):
            kb = F.normalize(b[j:j + bchunk].to(device).float(), dim=1)
            out[i:i + chunk, j:j + bchunk] = (q @ kb.T).cpu()
            del kb
        del q
        torch.cuda.empty_cache()
    return out


def mode_knn(args, device):
    if args.ks is None:
        args.ks = [1, 5, 20, 50]
    if args.stride is None:
        args.stride = 4
    _ck, _cfg, itm, ftm, tr, te = load_model_and_data(args, device)
    E_tr, D_tr, _ = _knn_collect(tr, itm, ftm, device, args.train_stride)
    E_te, D_te, cos_ftm = _knn_collect(te, itm, ftm, device, args.stride)
    print(f"{args.ckpt}\ntrain {len(D_tr)} transitions, test {len(D_te)}", flush=True)
    print(f"FTM held-out direction cosine: {np.median(cos_ftm):.3f}\n", flush=True)

    mean_e = E_tr.float().mean(0, keepdim=True)
    raw = _knn_cos_matrix(E_te[:128], E_tr[:512], device)
    cen = _knn_cos_matrix((E_te[:128].float() - mean_e).half(),
                          (E_tr[:512].float() - mean_e).half(), device)
    print("  retrieval sanity -- mean cosine between unrelated e_t")
    print(f"    raw       {raw.mean():.4f}   (near 1 means every frame looks like every other, "
          f"so raw-cosine k-NN is arbitrary)")
    print(f"    centred   {cen.mean():.4f}\n")

    for i in range(0, len(E_tr), 256):
        E_tr[i:i + 256] = (E_tr[i:i + 256].float() - mean_e).half()
    for i in range(0, len(E_te), 256):
        E_te[i:i + 256] = (E_te[i:i + 256].float() - mean_e).half()
    Ec_tr, Ec_te = E_tr, E_te
    print("  centred; computing similarities", flush=True)
    sims = _knn_cos_matrix(Ec_te, Ec_tr, device)
    dcos = _knn_cos_matrix(D_te, D_tr, device)

    best = dcos.max(dim=1).values.numpy()
    print("  best-match ceiling -- the single closest direction anywhere in the training library")
    print(f"    median {np.median(best):.3f}   mean {best.mean():.3f}   "
          f"(the FTM gets {np.median(cos_ftm):.3f})\n")

    print(f"  {'k':>5}{'k-NN oracle cosine':>22}{'neighbour spread':>20}{'query vs nbrs':>16}")
    for k in args.ks:
        idx = sims.topk(k, dim=1).indices
        preds, spread, qn = [], [], []
        for i in range(len(idx)):
            nb = D_tr[idx[i]].float()
            preds.append(F.normalize(nb.mean(0), dim=0))
            qn.append(dcos[i, idx[i]].mean().item())
            if k > 1:
                g = F.normalize(nb, dim=1)
                m = g @ g.T
                spread.append(((m.sum() - k) / (k * (k - 1))).item())
        p = torch.stack(preds).half()
        c = torch.tensor([F.cosine_similarity(p[i].float().unsqueeze(0),
                                              D_te[i].float().unsqueeze(0)).item()
                          for i in range(len(p))])
        print(f"  {k:>5}{np.median(c.numpy()):>22.3f}"
              f"{(np.mean(spread) if spread else float('nan')):>20.3f}"
              f"{np.mean(qn):>16.3f}")

    print("\n  neighbour spread is the mean pairwise cosine among the k neighbours' own directions:")
    print("  high means near-identical states move the same way (signal present, predictor is the")
    print("  gap); low means they move differently (the latent does not determine the direction).")


# ===================================================================================== mode: derisk
def _derisk_pairs_of(clips):
    return [(ci, t) for ci, c in enumerate(clips) for t in range(1, len(c["e"]) - 2)]


@torch.no_grad()
def _derisk_measure(ftm, itm, clips, device, stride):
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


def _derisk_finetune(ftm, itm, clips, idx, lam, steps, batch, lr, seed, device):
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


def mode_derisk(args, device):
    if args.stride is None:
        args.stride = 4
    _ck, cfg, itm, _ftm_unused, tr, te = load_model_and_data(args, device, need_ftm=False)
    idx = _derisk_pairs_of(tr)

    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    base = ForwardTransitionModel(cfg).to(device).eval()
    base.load_state_dict(ck["ftm"])
    print(f"{args.ckpt}\ntrain {len(tr)} clips ({len(idx)} pairs), test {len(te)} clips\n")
    b_cos, b_ratio, b_mse = _derisk_measure(base, itm, te, device, args.stride)
    print(f"  BEFORE   held-out cosine {b_cos:.3f}   ratio {b_ratio:.3f}   mse {b_mse:.4f}\n")

    rows = [("before", b_cos, b_ratio, b_mse, None)]
    for lam in args.lambdas:
        name = "control (MSE)" if lam == 0 else f"direction lambda={lam:g}"
        print(f"  ARM {name}")
        ftm = copy.deepcopy(base)
        for p in ftm.parameters():
            p.requires_grad_(True)
        _derisk_finetune(ftm, itm, tr, idx, lam, args.steps, args.batch, args.lr, args.seed, device)
        c, r, m = _derisk_measure(ftm, itm, te, device, args.stride)
        ctr, _r, _m = _derisk_measure(ftm, itm, tr, device, args.stride)
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


# ======================================================================================= mode: ridge
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _ridge_collect(clips, itm, ftm, device, stride):
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


def _ridge_gram(a, b, device, chunk=64, bchunk=256):
    out = torch.empty(len(a), len(b), dtype=torch.float64)
    for i in range(0, len(a), chunk):
        q = a[i:i + chunk].to(device).float()
        for j in range(0, len(b), bchunk):
            out[i:i + chunk, j:j + bchunk] = (q @ b[j:j + bchunk].to(device).float().T).double().cpu()
        del q
        torch.cuda.empty_cache()
    return out.numpy()


def _ridge_unit(K):
    return K / max(np.mean(np.diag(K)), 1e-12)


def _ridge_cos(K_tr, K_qu, G_tr, G_qu, alpha):
    c = K_qu @ np.linalg.inv(K_tr + alpha * np.eye(len(K_tr)))
    num = np.einsum("ij,ij->i", c, G_qu)
    nrm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", c, G_tr, c), 1e-12))
    return float(np.median(num / nrm))


def mode_ridge(args, device):
    if args.stride is None:
        args.stride = 8
    _ck, _cfg, itm, ftm, tr, te = load_model_and_data(args, device)
    E_tr, Z_tr, D_tr, _c, cid_tr = _ridge_collect(tr, itm, ftm, device, args.train_stride)
    E_te, Z_te, D_te, cos_ftm, _ = _ridge_collect(te, itm, ftm, device, args.stride)
    print(f"{args.ckpt}\ntrain {len(D_tr)} transitions, test {len(D_te)}", flush=True)
    print(f"FTM held-out direction cosine: {np.median(cos_ftm):.3f}\n", flush=True)

    mean_e = E_tr.float().mean(0, keepdim=True)
    for i in range(0, len(E_tr), 256):
        E_tr[i:i + 256] = (E_tr[i:i + 256].float() - mean_e).half()
    for i in range(0, len(E_te), 256):
        E_te[i:i + 256] = (E_te[i:i + 256].float() - mean_e).half()

    print("  building Gram matrices", flush=True)
    G_tr = _ridge_gram(D_tr, D_tr, device)
    G_te = _ridge_gram(D_te, D_tr, device)
    raw_tr = _ridge_gram(E_tr, E_tr, device)
    scale = max(np.mean(np.diag(raw_tr)), 1e-12)
    Kee_tr = raw_tr / scale
    Kee_te = _ridge_gram(E_te, E_tr, device) / scale

    Zt, Zq = Z_tr.numpy().astype(np.float64), Z_te.numpy().astype(np.float64)
    mu = Zt.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Zt - mu, full_matrices=False)

    clips_tr = sorted(set(cid_tr.tolist()))
    val_clips = set(clips_tr[1::3])
    va = np.array([c in val_clips for c in cid_tr]); fit = ~va

    def evaluate(name, Ktr_full, Kte_full):
        best_a, best_v = ALPHAS[0], -9
        for a in ALPHAS:
            v = _ridge_cos(Ktr_full[np.ix_(fit, fit)], Ktr_full[np.ix_(va, fit)],
                           G_tr[np.ix_(fit, fit)], G_tr[np.ix_(va, fit)], a)
            if v > best_v:
                best_v, best_a = v, a
        t = _ridge_cos(Ktr_full, Kte_full, G_tr, G_te, best_a)
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


# =========================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ceiling", "knn", "derisk", "ridge"], required=True)
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--stride", type=int, default=None, help="default 4 for ceiling/knn/derisk, "
                    "8 for ridge (their original, different, defaults)")
    ap.add_argument("--train_stride", type=int, default=2, help="knn/ridge modes only: library-side "
                    "stride, kept denser than the query set (see those modes' own history)")
    ap.add_argument("--ks", type=int, nargs="+", default=None, help="ceiling/knn modes -- default "
                    "[1,5,20] for ceiling, [1,5,20,50] for knn (their original, different, defaults)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64], help="ridge mode only")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 10.0], help="derisk mode only")
    ap.add_argument("--steps", type=int, default=2000, help="derisk mode only")
    ap.add_argument("--batch", type=int, default=8, help="derisk mode only")
    ap.add_argument("--lr", type=float, default=1e-5, help="derisk mode only")
    ap.add_argument("--seed", type=int, default=0, help="derisk mode only")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"ceiling": mode_ceiling, "knn": mode_knn, "derisk": mode_derisk, "ridge": mode_ridge}[args.mode](
        args, device)


if __name__ == "__main__":
    main()
