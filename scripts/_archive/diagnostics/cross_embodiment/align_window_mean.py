"""Alignment on window-mean z (mean over P consecutive stride-k pairs, never across a command switch): does it keep the
motion, and does a head-only alignment on it generalise to new clips? (F318 follow-up, 2026-10-07)

Frozen checkpoint; train = rr_{c10,b1}_clips_train, test = rr_{c10,b1}_clips_heldout. Per window: mean z, mean label
(the 1 s CoM Froude), standardised over both bodies.
  1. motion kept: ridge window-mean z -> window-mean Froude, per body, fit on train, R2 on test (vs the same with single z)
  2. head-only soft InfoNCE (c10 <-> B1) on window-means, train vs test loss, with floor and chance

    .venv/bin/python3 scripts/diagnostics/cross_embodiment/align_window_mean.py wm/runs/round2_align_s0_rr/best.pt
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from wm.align import AlignHead, soft_infonce, soft_targets  # noqa: E402
from wm.data.emb_cache import file_cached  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

CW = "data/counterfactual_walks"; CACHE = os.path.join(ROOT, "results/wm/cache/fitproj_files")
P = int(os.environ.get("P", 11))


@torch.no_grad()
def windows(m, enc_get, d, emb, stride_win=2):
    Zs, Zw, Fs, Fw = [], [], [], []
    k = m.stride
    for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
        c = load(p, REGISTRY[emb], lazy_frames=True)
        e = file_cached(p, CACHE, lambda: encode_clip(enc_get(), np.asarray(c["frames"]), 2)).float()
        if m.offset is not None:
            e = e - m.offset.float().reshape(e.shape[1:])
        n = len(e) - k
        z = torch.cat([m.itm(e[i:min(i + 16, n)].to(m.device), e[i + k:min(i + 16, n) + k].to(m.device)).cpu()
                       for i in range(0, n, 16)])
        z = z.reshape(len(z), -1)
        bm = np.asarray(c["body_motion"])[:, :3]
        lab = np.stack([bm[t:t + k].mean(0) for t in range(n)])
        seg = np.asarray(c.get("segment", np.zeros(len(bm))))
        for t in range(0, n - P + 1, stride_win):
            if len(set(seg[t:t + P + k - 1].tolist())) > 1:     # never across a command switch
                continue
            Zw.append(z[t:t + P].mean(0).numpy()); Fw.append(lab[t:t + P].mean(0))
            Zs.append(z[t + P // 2].numpy()); Fs.append(lab[t + P // 2])
    return np.array(Zs), np.array(Fs), np.array(Zw), np.array(Fw)


def main():
    ckpt = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    held = {}

    def enc():
        if "e" not in held:
            from vjepa2_encoder import VJEPA2FrameEncoder
            held["e"] = VJEPA2FrameEncoder(dtype=torch.float32)
        return held["e"]
    D = {}
    for body, emb in (("c10", "hexapod"), ("b1", "b1")):
        m = Models(os.path.join(ROOT, ckpt), emb, 18 if emb == "hexapod" else 12, dev); m.device = dev
        for split in ("train", "heldout"):
            D[body, split] = windows(m, enc, f"{CW}/rr_{body}_clips_{split}", emb)
            print(body, split, "windows", len(D[body, split][2]), flush=True)
        del m
    allF = np.concatenate([D[b, s][3] for b in ("c10", "b1") for s in ("train", "heldout")])
    mu, sd = allF.mean(0), allF.std(0)
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    print(f"\n1. motion kept? ridge z -> Froude, fit train, R2 on held-out (fwd / lat / yaw); window P={P}")
    for body in ("c10", "b1"):
        for name, zi, fi in (("single z", 0, 1), ("window-mean z", 2, 3)):
            Xtr, Ytr = D[body, "train"][zi], D[body, "train"][fi]; Xte, Yte = D[body, "heldout"][zi], D[body, "heldout"][fi]
            sc = StandardScaler().fit(Xtr); r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(Xtr), Ytr)
            pr = r.predict(sc.transform(Xte))
            r2 = 1 - ((pr - Yte) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
            print(f"   {body:<4} {name:<14} " + " / ".join(f"{x:+.2f}" for x in r2))
    print("\n2. head-only alignment on window-mean z (c10 <-> B1), sigma 0.25, tau 0.1")
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    S = {s: (t(D["c10", s][2]), t((D["c10", s][3] - mu) / sd), t(D["b1", s][2]), t((D["b1", s][3] - mu) / sd))
         for s in ("train", "heldout")}
    torch.manual_seed(0); head = AlignHead(S["train"][0].shape[1]); opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    def L(s):
        za, fa, zb, fb = S[s]; ua, ub = head(za), head(zb)
        return 0.5 * (soft_infonce(ua, fa, ub, fb, 0.1, 0.25) + soft_infonce(ub, fb, ua, fa, 0.1, 0.25))
    for it in range(2001):
        opt.zero_grad(); l = L("train"); l.backward(); opt.step()
        if it % 500 == 0:
            with torch.no_grad():
                print(f"   step {it:4d}  train {l.item():.3f}  test {L('heldout').item():.3f}", flush=True)
    za, fa, zb, fb = S["heldout"]
    fl = 0.5 * sum(float(-(w * w.clamp_min(1e-12).log()).sum(1).mean()) for w in (soft_targets(fa, fb, .25), soft_targets(fb, fa, .25)))
    print(f"   test best possible {fl:.3f}; chance {0.5 * (np.log(len(zb)) + np.log(len(za))):.3f}")


if __name__ == "__main__":
    main()
