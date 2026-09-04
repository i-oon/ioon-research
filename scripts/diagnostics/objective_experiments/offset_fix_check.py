"""Does a learned per-embodiment offset remove the pooled-delta leak without killing Delta-state?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/confirm_counterfactual.py

`frame_vs_delta_classify.py` found the state head's actual input -- `delta.mean(1)`, the FTM's
predicted change pooled over patch tokens -- reads embodiment identity at 0.977, and that an ORACLE
per-embodiment mean removal (using the true label on the whole set) drops it to 0.114, pointing at
F41's mechanism: an additive per-embodiment offset, not something distributed and adversarial-shaped.

**This is the non-oracle version.** The offset is fit on TRAIN clips only, applied by known
embodiment id (available at train and inference alike -- this project already routes every
per-embodiment head this way), and both numbers are read on TEST clips the offset never saw:

    leak       embodiment accuracy on offset-corrected pooled delta, held-out clips
    signal     ridge R2, body motion from [pooled delta, z], held out, WITH vs WITHOUT the offset

**The read.** Leak toward chance (0.501) and signal preserved -> the offset fix is targeted and
cheap, proceed to a short fine-tune confirm. Leak drops but signal drops with it -> identity and
body motion are entangled in the pooled delta and this does not separate them. Leak does not move
-> the earlier oracle result does not generalise past the exact means it was computed on, and the
per-embodiment-offset diagnosis was wrong.

**Diagnosis only; the offset here is fit in closed form for the test, not trained into the model.**
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--extra_cache", default="results/wm/cache/ego_b1.pt")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
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

    Dpool, Z, Y, EID, cid = [], [], [], [], []
    offset = 0
    cache1 = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    cache2 = torch.load(os.path.join(ROOT, args.extra_cache), map_location="cpu", mmap=True)
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
                    Dpool.append((pred - e_t).mean(1)[0].float().cpu().numpy())
                    Z.append(z[0].float().cpu().numpy())
                    Y.append(bm[t])
                    EID.append(eid)
                    cid.append(offset + ci)
        offset += len(clips)
        print(f"  {name}: {len(clips)} clips", flush=True)

    Dpool, Z = np.stack(Dpool), np.stack(Z)
    Y = np.stack(Y).astype(np.float64)
    EID, cid = np.array(EID), np.array(cid)
    n = len(EID)
    print(f"\n{args.ckpt}\n{n} transitions, chance {max(np.bincount(EID)) / n:.3f}\n")

    rng = np.random.default_rng(args.seed)
    clips_all = sorted(set(cid.tolist()))
    rng.shuffle(clips_all)
    n_te = max(1, int(len(clips_all) * args.test_frac))
    te_clips = set(clips_all[:n_te])
    te = np.array([c in te_clips for c in cid]); tr = ~te

    # per-embodiment offset, fit on TRAIN clips only
    off = np.zeros((2, Dpool.shape[1]))
    for eid in np.unique(EID):
        m = tr & (EID == eid)
        off[eid] = Dpool[m].mean(0)
    Dcorr = Dpool - off[EID]

    def clf_acc(X):
        Xs = (X[tr] - X[tr].mean(0, keepdims=True)) / (X[tr].std(0, keepdims=True) + 1e-6)
        Xte = (X[te] - X[tr].mean(0, keepdims=True)) / (X[tr].std(0, keepdims=True) + 1e-6)
        clf = LogisticRegression(max_iter=500).fit(Xs, EID[tr])
        return clf.score(Xte, EID[te])

    print("  LEAK -- embodiment accuracy, held-out clips, offset fit on train only")
    print(f"    raw pooled delta          : {clf_acc(Dpool):.3f}")
    print(f"    offset-corrected          : {clf_acc(Dcorr):.3f}")
    print(f"    z (reference)             : {clf_acc(Z):.3f}\n")

    def ridge_r2(Xfeat):
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        Yt = (Y - mu) / sd
        best_r2, best_a = -1e9, None
        for a in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            r = Ridge(alpha=a).fit(Xfeat[tr], Yt[tr])
            pred = r.predict(Xfeat[te])
            ss = ((pred - Yt[te]) ** 2).sum()
            r2 = 1 - ss / max(((Yt[te] - Yt[tr].mean(0)) ** 2).sum(), 1e-12)
            if r2 > best_r2:
                best_r2, best_a = r2, a
        return best_r2, best_a

    print("  SIGNAL -- Delta-state ridge R2, [pooled delta, z], held-out clips")
    for name, D in (("raw", Dpool), ("offset-corrected", Dcorr)):
        X = np.concatenate([D, Z], axis=1)
        r2, a = ridge_r2(X)
        print(f"    {name:>20}: R2 {r2:.3f}  (alpha {a:g})")

    print("\n  READ: leak should fall toward chance (0.501) for 'offset-corrected'; SIGNAL should")
    print("  stay close between 'raw' and 'offset-corrected'. Both holding -> the fix is targeted.")


if __name__ == "__main__":
    main()
