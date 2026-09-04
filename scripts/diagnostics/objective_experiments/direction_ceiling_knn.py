"""Does this latent support better transition-direction prediction than the FTM's 0.687?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/direction_ceiling_knn.py

Separates the two branches left after the objective one died, without training anything:

    capacity          the mapping e_t -> direction is learnable and the FTM is too small or too
                      lightly trained. A smooth non-parametric predictor should then beat 0.687.
    embedding limit   the latent does not determine the direction, so no predictor does better.

Four measurements, each answering a different half of that:

    retrieval sanity   mean cosine between unrelated `e_t`. **A previous k-NN pass scored 0.04 and
                       this line is why it is repeated here**: these embeddings carry a large shared
                       component, so raw cosine ranks near-arbitrary frames as neighbours. Retrieval
                       is therefore done on centred embeddings, and the raw number is printed beside
                       it so the earlier result can be read for what it was.
    k-NN oracle        predict the direction as the mean of the k nearest training frames' own
                       directions. Roughly the best a smooth predictor can do at this data density.
    best-match ceiling max over the whole training library of cosine with the true direction. **The
                       strongest cheap bound**: if even the single best-matching recorded transition
                       cannot exceed the FTM, the library does not contain the needed directions and
                       no retrieval or interpolation of them will.
    neighbour spread   among the k neighbours of a query, how consistent are their own directions
                       with each other. Inconsistent means near-identical states are followed by
                       different motions, which caps any predictor from `e_t` alone.

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


def collect(clips, itm, ftm, device, stride):
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


def cos_matrix(a, b, device, chunk=64, bchunk=256):
    """Cosine between every row of `a` and every row of `b`, both very wide.

    **Both sides are normalised in GPU chunks.** Materialising a float32 copy of the training side
    is 4.3 GB at 3,000 rows of 360,448 and killed this script silently the first time.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--stride", type=int, default=4, help="test-side stride (query count)")
    ap.add_argument("--train_stride", type=int, default=2,
                    help="library-side stride. **Density is the whole point of the k-NN bound**, so "
                         "the library is kept denser than the query set; both sides at full width "
                         "is 360,448 floats a row and exhausts RAM.")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20, 50])
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
    E_tr, D_tr, _ = collect(tr, itm, ftm, device, args.train_stride)
    E_te, D_te, cos_ftm = collect(te, itm, ftm, device, args.stride)
    print(f"{args.ckpt}\ntrain {len(D_tr)} transitions, test {len(D_te)}", flush=True)
    print(f"FTM held-out direction cosine: {np.median(cos_ftm):.3f}\n", flush=True)

    # --- retrieval sanity -----------------------------------------------------------------------
    mean_e = E_tr.float().mean(0, keepdim=True)
    raw = cos_matrix(E_te[:128], E_tr[:512], device)
    cen = cos_matrix((E_te[:128].float() - mean_e).half(),
                     (E_tr[:512].float() - mean_e).half(), device)
    print(f"  retrieval sanity -- mean cosine between unrelated e_t")
    print(f"    raw       {raw.mean():.4f}   (near 1 means every frame looks like every other, "
          f"so raw-cosine k-NN is arbitrary)")
    print(f"    centred   {cen.mean():.4f}\n")

    for i in range(0, len(E_tr), 256):               # centre in place, then drop the raw copies
        E_tr[i:i + 256] = (E_tr[i:i + 256].float() - mean_e).half()
    for i in range(0, len(E_te), 256):
        E_te[i:i + 256] = (E_te[i:i + 256].float() - mean_e).half()
    Ec_tr, Ec_te = E_tr, E_te
    print("  centred; computing similarities", flush=True)
    sims = cos_matrix(Ec_te, Ec_tr, device)          # test x train, centred embedding similarity
    dcos = cos_matrix(D_te, D_tr, device)            # test x train, direction agreement

    # --- best-match ceiling ---------------------------------------------------------------------
    best = dcos.max(dim=1).values.numpy()
    print(f"  best-match ceiling -- the single closest direction anywhere in the training library")
    print(f"    median {np.median(best):.3f}   mean {best.mean():.3f}   "
          f"(the FTM gets {np.median(cos_ftm):.3f})\n")

    # --- k-NN oracle and neighbour spread -------------------------------------------------------
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


if __name__ == "__main__":
    main()
