"""Is the FTM's transition-direction cosine of 0.690 near the achievable ceiling, or is there room?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/cosine_ceiling.py

`undermovement.py` reduced the fine-ranking and long-horizon wall to one number: the FTM predicts the
direction of `e_t+1 - e_t` at cosine **0.690**, and its under-movement is the correct MSE response to
that (`alpha*` 0.971). Whether that number is fixable decides whether anything is worth rebuilding.

Four references, none of which needs training:

    constant           the mean training displacement, ignoring both `e_t` and the action. **The
                       floor that matters** -- if the gait's average step already scores near 0.690,
                       the FTM is barely beating "always step the same way".
    k-NN on e_t        copy the displacement of the most similar training frames. A non-parametric
                       predictor with no action information.
    k-NN on [e_t, z]   the same with the action, so the pair brackets what a strong memoriser gets
                       from the inputs the FTM is given.
    FTM on TRAIN       the same model on transitions it was fitted on. **Train ~ held-out means the
                       model cannot fit direction even where it has seen the answer**, which is a
                       capacity or objective limit rather than a generalisation gap.

**The read.** A k-NN or the training set clearly above 0.690 means headroom and the FTM is the
binding constraint. Everything plateauing near 0.690 -- including the constant -- means the ceiling
is a property of V-JEPA2's latent geometry on this data, and no objective or architecture recovers
it.

Cosines are computed on the same held-out body the 0.690 was measured on, except the `FTM on TRAIN`
row, which is the point of that row. **Diagnosis only; trains nothing.**
"""
import argparse
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


def collect(clips, itm, ftm, device, stride):
    """e_t, z and the true displacement per transition, plus the FTM's own cosine."""
    E, Z, D, cos = [], [], [], []
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            for t in range(1, len(e) - 2, stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                p = ftm(e_t, z)
                dp, dt = (p - e_t).flatten(), (e1 - e_t).flatten()
                cos.append(torch.nn.functional.cosine_similarity(
                    dp.unsqueeze(0), dt.unsqueeze(0)).item())
                E.append(e[t].flatten().half())
                Z.append(z[0].float().cpu())
                D.append(dt.half().cpu())
    return (torch.stack(E), torch.stack(Z), torch.stack(D), np.array(cos))


def cosine_rows(a, b, device, chunk=64):
    """Row-wise cosine between two equally shaped, very wide half-precision matrices."""
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        x = a[i:i + chunk].to(device).float()
        y = b[i:i + chunk].to(device).float()
        out[i:i + chunk] = torch.nn.functional.cosine_similarity(x, y, dim=1).cpu().numpy()
    return out


def knn_predict(feat_te, feat_tr, disp_tr, k, device, chunk=32):
    """Mean displacement of the k most similar training rows, by cosine on `feat`."""
    tr_n = torch.nn.functional.normalize(feat_tr.float(), dim=1)
    pred = torch.empty(len(feat_te), disp_tr.shape[1], dtype=torch.float16)
    for i in range(0, len(feat_te), chunk):
        q = torch.nn.functional.normalize(feat_te[i:i + chunk].to(device).float(), dim=1)
        sims = torch.empty(len(q), len(tr_n))
        for j in range(0, len(tr_n), 512):
            sims[:, j:j + 512] = (q @ tr_n[j:j + 512].to(device).T).cpu()
        idx = sims.topk(k, dim=1).indices
        for r in range(len(q)):
            pred[i + r] = disp_tr[idx[r]].float().mean(0).half()
        del q
        torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    lag = max(1, cfg.action_lag)
    tr_clips = gather(os.path.join(ROOT, args.train_data), args.embodiment, None, ck, cache, 2,
                      lag, device)
    te_clips = gather(os.path.join(ROOT, args.test_data), args.embodiment, None, ck, cache, 2,
                      lag, device)

    E_tr, Z_tr, D_tr, cos_tr = collect(tr_clips, itm, ftm, device, args.stride)
    E_te, Z_te, D_te, cos_te = collect(te_clips, itm, ftm, device, args.stride)
    print(f"{args.ckpt}")
    print(f"train {len(tr_clips)} clips / {len(D_tr)} transitions from {args.train_data}")
    print(f"test  {len(te_clips)} clips / {len(D_te)} transitions from {args.test_data}\n")

    print(f"  {'predictor of the transition direction':>44}{'cosine':>9}")
    print(f"  {'FTM, held-out body  (the 0.690)':>44}{np.median(cos_te):>9.3f}")
    print(f"  {'FTM, TRAINING body':>44}{np.median(cos_tr):>9.3f}")

    const = D_tr.float().mean(0, keepdim=True).half().expand(len(D_te), -1)
    print(f"  {'constant: mean training displacement':>44}"
          f"{np.median(cosine_rows(const, D_te, device)):>9.3f}")

    for k in args.ks:
        p = knn_predict(E_te, E_tr, D_tr, k, device)
        print(f"  {f'k-NN on e_t, k={k}':>44}{np.median(cosine_rows(p, D_te, device)):>9.3f}")
    feat_tr = torch.cat([torch.nn.functional.normalize(E_tr.float(), dim=1),
                         torch.nn.functional.normalize(Z_tr, dim=1)], dim=1).half()
    feat_te = torch.cat([torch.nn.functional.normalize(E_te.float(), dim=1),
                         torch.nn.functional.normalize(Z_te, dim=1)], dim=1).half()
    for k in args.ks:
        p = knn_predict(feat_te, feat_tr, D_tr, k, device)
        print(f"  {f'k-NN on [e_t, z], k={k}':>44}{np.median(cosine_rows(p, D_te, device)):>9.3f}")

    print(f"\n  train - held-out gap for the FTM: {np.median(cos_tr) - np.median(cos_te):+.3f}")


if __name__ == "__main__":
    main()
