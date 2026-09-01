"""Is the signal a broken rhythm exposes **intent**, or only *which behaviour this clip is*?

    .venv/bin/python3 scripts/diagnostics/intent_recoverability.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/best.pt --data data/allocentric/beh12_c10f10t10_intent_flat \\
        --embodiment hexapod

**This is the control F164's random noise fails.** Jitter opens the single-frame-to-pair gap because
a random component can only ever be seen in the transition -- which is true and useless. The
question that decides Direction B is whether the recovered thing is a **command change** a
controller could act on, or a **behaviour label** we already have and already rejected.

Three targets under one protocol, ridge in the dual on the full embedding, split by clip:

    a_t          the instantaneous command -- F159's quantity, for reference
    da = a_t+1 - a_t   **the intent**: what the command is about to do differently
    family       speed / turn / side, as a classification **control**

**The result is the `da` row and only the `da` row.** A pair that beats a single frame on `family`
has recovered a label; a pair that beats a single frame on `da` has recovered a change. **If the
gap opens on `family` but not on `da`, the data is no better than what we have.**

**Separability improving is not a result either** -- meaningful turns can make behaviours easier to
tell apart, which looks like a win and is not the one being tested.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--pair_lag", type=int, default=1)
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    from_checkpoint(ck["config"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, 1, device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    k = args.pair_lag
    E0, E1, A, D, fam, cid = [], [], [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        a = c["a"]
        if len(e) < k + 3:
            continue
        for t in range(1, min(len(e) - k - 1, len(a) - 2), args.stride):
            E0.append(e[t].flatten().half())
            E1.append(e[t + k].flatten().half())
            A.append(a[t].flatten().float())
            D.append((a[t + 1] - a[t]).flatten().float())
            fam.append(FAMILY(c["cond"])); cid.append(ci)
    E0, E1 = torch.stack(E0), torch.stack(E1)
    A, D = torch.stack(A).numpy(), torch.stack(D).numpy()
    fam = np.array(fam); cid = np.array(cid)

    order = collections.defaultdict(list)
    for ci in sorted(set(cid.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test for c in cid]); tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in cid[tr]])

    K0 = gram(E0, E0, device).numpy()
    K1 = gram(E1, E1, device).numpy()
    feats = {"e_t (single frame)": K0, f"[e_t, e_t+{k}] (pair)": K0 + K1}

    fams = sorted(set(fam))
    onehot = np.stack([(fam == f).astype(float) for f in fams], 1)
    targets = {
        "a_t   instantaneous command": (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6),
        "da    command change (INTENT)": (D - D[tr].mean(0)) / (D[tr].std(0) + 1e-6),
        "family  label (CONTROL)": onehot,
    }

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test, split by clip, "
          f"embedding dimension {E0.shape[1]}\n")
    print(f"  {'target':>30}{'single frame':>15}{'pair':>10}{'gap':>9}")
    for name, y in targets.items():
        row, base = {}, None
        for fname, Kf in feats.items():
            r2, pred, _ = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], y[tr], y[te], folds)
            row[fname] = (r2, pred)
        (s_r2, _), (p_r2, p_pred) = row[list(feats)[0]], row[list(feats)[1]]
        extra = ""
        if name.startswith("family"):
            _, s_pred = row[list(feats)[0]][1], row[list(feats)[0]][1]
            acc_s = (row[list(feats)[0]][1].argmax(1) == onehot[te].argmax(1)).mean()
            acc_p = (p_pred.argmax(1) == onehot[te].argmax(1)).mean()
            extra = f"   accuracy {acc_s:.3f} -> {acc_p:.3f}, chance {onehot[te].mean(0).max():.3f}"
        print(f"  {name:>30}{s_r2:>15.3f}{p_r2:>10.3f}{p_r2 - s_r2:>+9.3f}{extra}")

    print(f"\n  per family, the INTENT row only")
    y = targets["da    command change (INTENT)"]
    preds = {}
    for fname, Kf in feats.items():
        _, preds[fname], _ = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], y[tr], y[te], folds)
    print(f"  {'family':>10}{'single frame':>15}{'pair':>10}{'gap':>9}{'n':>7}")
    for f in fams:
        m = fam[te] == f
        if not m.any():
            continue
        denom = max(((y[te][m] - y[tr].mean(0)) ** 2).sum(), 1e-9)
        r = [1 - ((preds[fn][m] - y[te][m]) ** 2).sum() / denom for fn in feats]
        print(f"  {f:>10}{r[0]:>15.3f}{r[1]:>10.3f}{r[1] - r[0]:>+9.3f}{m.sum():>7}")

    print("\n  **the `da` row is the result.** A gap on `family` and not on `da` means a label was "
          "recovered,\n  which we already have and already rejected. Separability improving is not "
          "a result either.")


if __name__ == "__main__":
    main()
