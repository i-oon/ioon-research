"""Is what the null-action prediction misses **structured by the action**, or is it noise?

    .venv/bin/python3 scripts/diagnostics/residual_structure.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/best.pt --data data/allocentric/beh12_c08f09t09_flat \\
        --embodiment hexapod

**F157 closed the objective-level path**: no weighting and no frameskip makes the action matter more
than about three percent of one-step prediction error. This asks what those three percent *are*
before anything is rebuilt.

    r  =  e_t+k  -  FTM(e_t, ITM(e_t, e_t))

`r` is exactly what a phase-only, action-blind prediction leaves on the table. Three questions about
it, and the answer decides which representation-level direction is worth an advisor's time:

  1. **structured?**    does `r` separate the behaviour families above chance
  2. **recoverable?**   can a ridge read the real joint command out of `r`
  3. **or noise?**      how much of `r` survives averaging over states that share an action

**Question 2 is meaningless without its control.** `e_t` alone predicts the action well on this data
-- the frame shows the pose -- so a probe on `r` that merely matches `e_t` has found nothing. The
control is reported beside it every time, along with the raw difference `e_t+k - e_t`, which is what
`r` would collapse to if the forward model's null prediction were doing nothing at all.

Ridge is solved in the dual, so the full 360,448-dimensional embedding is used exactly rather than
projected down. Splits are **by clip**, never by frame.
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda cond: "side" if cond.startswith("side") else cond.split("_")[0]
ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


def gram(a, b, device, chunk=64):
    """`a @ b.T` for tall, very wide matrices, held on the CPU in half precision.

    **Both sides are chunked.** Moving `b` across whole was fine at 1,000 rows of 360,448 and is
    4.5 GB at 3,100 -- the egocentric sets are three times the size of the ones this was written
    for, and it died on the GPU rather than degrading.
    """
    out = torch.empty(len(a), len(b), dtype=torch.float64)
    for j in range(0, len(b), chunk * 8):
        bt = b[j:j + chunk * 8].to(device).float()
        for i in range(0, len(a), chunk):
            out[i:i + chunk, j:j + chunk * 8] = (
                a[i:i + chunk].to(device).float() @ bt.T).double().cpu()
        del bt
        torch.cuda.empty_cache()
    return out


def ridge_r2(K_tr, K_te, y_tr, y_te, folds):
    """Dual ridge with the penalty chosen by grouped cross-validation on the training half."""
    best, best_a = -1e9, ALPHAS[0]
    for alpha in ALPHAS:
        scores = []
        for f in sorted(set(folds)):
            m = folds != f
            A = K_tr[np.ix_(m, m)] + alpha * np.eye(m.sum())
            w = np.linalg.solve(A, y_tr[m])
            p = K_tr[np.ix_(~m, m)] @ w
            scores.append(1 - ((p - y_tr[~m]) ** 2).sum() / max(((y_tr[~m] - y_tr[m].mean(0)) ** 2).sum(), 1e-9))
        if np.mean(scores) > best:
            best, best_a = np.mean(scores), alpha
    A = K_tr + best_a * np.eye(len(K_tr))
    w = np.linalg.solve(A, y_tr)
    pred = K_te @ w
    ss = ((pred - y_te) ** 2).sum()
    return 1 - ss / max(((y_te - y_tr.mean(0)) ** 2).sum(), 1e-9), pred, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--lag", type=int, default=1, help="**set this to the checkpoint's own "
                    "`frame_stride`.** Measuring off-distribution is what made F156 wrong")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])

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

    lag = args.lag
    R, E, D, A, fam, clip_id = [], [], [], [], [], []
    with torch.no_grad():
        for ci, c in enumerate(clips):
            e = c["e"].float().to(device)
            if len(e) < lag + 3:
                continue
            for t in range(1, len(e) - lag - 1, args.stride):
                z_null = itm(e[t:t + 1], e[t:t + 1])
                null = ftm(e[t:t + 1], z_null)[0]
                R.append((e[t + lag] - null).flatten().half().cpu())
                E.append(e[t].flatten().half().cpu())
                D.append((e[t + lag] - e[t]).flatten().half().cpu())
                A.append(c["a"][t].flatten().float().cpu())
                fam.append(FAMILY(c["cond"]))
                clip_id.append(ci)
    R, E, D = torch.stack(R), torch.stack(E), torch.stack(D)
    A = torch.stack(A).numpy()
    fam = np.array(fam); clip_id = np.array(clip_id)

    # split by clip, alternating within each condition so both halves see every behaviour
    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id])
    tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])

    A = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)
    onehot = np.stack([(fam == f).astype(float) for f in sorted(set(fam))], 1)

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}, lag {lag}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip, "
          f"embedding dimension {R.shape[1]}\n")

    feats = {"r  (null residual)": R, "e_t (control)": E, "e_t+k - e_t (raw)": D}
    preds = {}
    print(f"  {'features':>22}{'action R2':>11}{'family acc':>12}{'chance':>9}{'alpha':>9}")
    for name, X in feats.items():
        Kf = gram(X, X, device).numpy()
        K_tr = Kf[np.ix_(tr, tr)]; K_te = Kf[np.ix_(te, tr)]
        r2, pred, alpha = ridge_r2(K_tr, K_te, A[tr], A[te], folds)
        preds[name] = pred
        _, ph, _ = ridge_r2(K_tr, K_te, onehot[tr], onehot[te], folds)
        acc = (ph.argmax(1) == onehot[te].argmax(1)).mean()
        chance = max(onehot[te].mean(0))
        print(f"  {name:>22}{r2:>11.3f}{acc:>12.3f}{chance:>9.3f}{alpha:>9.4g}")

    print(f"\n  per family, action R2 on held-out clips")
    print(f"  {'family':>10}" + "".join(f"{n:>22}" for n in feats) + f"{'n':>7}")
    for f in sorted(set(fam[te])):
        m = fam[te] == f
        row = ""
        for name in feats:
            ss = ((preds[name][m] - A[te][m]) ** 2).sum()
            row += f"{1 - ss / max(((A[te][m] - A[tr].mean(0)) ** 2).sum(), 1e-9):>22.3f}"
        print(f"  {f:>10}{row}{m.sum():>7}")

    # how much of r survives averaging over states that share an action
    Ate = A[te]
    d2 = ((Ate[:, None, :] - Ate[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(1)
    Rte = R[te]
    Kr = gram(Rte, Rte, device).numpy()
    diag = np.diag(Kr)
    matched = diag + diag[nn] - 2 * Kr[np.arange(len(nn)), nn]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(nn))
    random_pair = diag + diag[perm] - 2 * Kr[np.arange(len(nn)), perm]
    ratio = matched.mean() / max(random_pair.mean(), 1e-9)
    print(f"\n  matched-action pairs differ by {matched.mean():.4f}, random pairs by "
          f"{random_pair.mean():.4f}   ratio {ratio:.3f}")
    print("  " + ("**r is largely determined by the action** -- states sharing an action share a "
                  "residual" if ratio < 0.7 else
                  "**r barely depends on the action** -- two states with the same command leave "
                  "almost as different a residual as two unrelated ones"))


if __name__ == "__main__":
    main()
