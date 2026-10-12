"""WHY does a z read-out transfer across bodies on forward but not on lateral / yaw? (2026-10-12)

Uses exactly the z of `scripts/figures/shared_latent_figure.py` (same bodies, held-out clips, ITM z of real frame pairs,
stride k, every 2nd frame, label = mean Froude over the pair) and separates three explanations per channel:
  (1) the body's own z does not carry the channel        -> own-body read-out R2 (5-fold by clip) is low
  (2) carried, but along another direction than c10's     -> own R2 high, c10 read-out on it low; cosine between the
                                                             two bodies' read-out weight vectors (standardised z) low
  (3) same direction, other scale / offset                -> c10 read-out applied to the body: correlation r high but R2
                                                             low; slope / offset of that read against the truth
    .venv/bin/python3 scripts/diagnostics/cross_embodiment/z_channel_transfer.py --model NAME=wm/runs/NAME/best.pt \\
        --c10 DIR=CACHE --c08 DIR=CACHE --b1 DIR=CACHE      # the shared-latent step's arguments
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "scripts/figures"))
import shared_latent_figure as SL  # noqa: E402


def main():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--c10", required=True); ap.add_argument("--c08", required=True); ap.add_argument("--b1", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    for i, (lab, emb, key) in enumerate((("c10 (pretrain)", "hexapod", a.c10), ("c08 (held-out)", "hexapod", a.c08),
                                         ("B1", "b1", a.b1))):
        d, c = key.split("=", 1); SL.BODIES[i] = (lab, emb, d, c)
    name, path = a.model.split("=", 1)
    Z, B, F, G, _ = SL.latents(path, a.device)
    Zs = StandardScaler().fit_transform(Z)
    al = np.logspace(-2, 4, 13)
    fits = {b: RidgeCV(alphas=al).fit(Zs[B == b], F[B == b]) for b in (0, 1, 2)}
    ch = ("fwd", "lat", "yaw")
    L = [f"{name}: held-out clips; z standardised over all bodies; per channel {' / '.join(ch)}",
         "label spread (std): " + "; ".join(f"{SL.BODIES[b][0]} " + " / ".join(f"{v:.3f}" for v in F[B == b].std(0))
                                            for b in (0, 1, 2))]
    for b in (0, 1, 2):
        m = B == b
        own = np.zeros_like(F[m])
        for tr, te in GroupKFold(5).split(Zs[m], F[m], G[m]):
            own[te] = RidgeCV(alphas=al).fit(Zs[m][tr], F[m][tr]).predict(Zs[m][te])
        r2_own = 1 - ((own - F[m]) ** 2).sum(0) / ((F[m] - F[m].mean(0)) ** 2).sum(0)
        L.append(f"(1) {SL.BODIES[b][0]:16s} own read-out R2 (5-fold by clip): " + " / ".join(f"{v:+.2f}" for v in r2_own))
    for b in (1, 2):
        m = B == b; p = fits[0].predict(Zs[m]); t = F[m]
        r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
        r = [np.corrcoef(p[:, j], t[:, j])[0, 1] for j in range(3)]
        sl = [np.polyfit(t[:, j], p[:, j], 1) for j in range(3)]
        cos = [float(np.dot(fits[0].coef_[j], fits[b].coef_[j]) /
                     (np.linalg.norm(fits[0].coef_[j]) * np.linalg.norm(fits[b].coef_[j]))) for j in range(3)]
        L.append(f"c10 read-out on {SL.BODIES[b][0]}: R2 " + " / ".join(f"{v:+.2f}" for v in r2)
                 + " | (3) correlation r " + " / ".join(f"{v:+.2f}" for v in r)
                 + " | slope " + " / ".join(f"{s[0]:+.2f}" for s in sl) + " | offset " + " / ".join(f"{s[1]:+.3f}" for s in sl))
        L.append(f"(2) cosine between the c10 and {SL.BODIES[b][0]} read-out directions: " + " / ".join(f"{v:+.2f}" for v in cos))
    # (4) per-body CENTRED z: subtract each body's own mean z (no training; a body-level constant shift removed)
    Zc = Zs.copy()
    for b in (0, 1, 2):
        Zc[B == b] -= Zc[B == b].mean(0)
    fc = RidgeCV(alphas=al).fit(Zc[B == 0], F[B == 0])
    for b in (1, 2):
        m = B == b; p = fc.predict(Zc[m]); t = F[m]
        r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
        L.append(f"(4) per-body centred z, c10 read-out on {SL.BODIES[b][0]}: R2 " + " / ".join(f"{v:+.2f}" for v in r2))
    # (5) GAIT-AVERAGED z: within each clip, average z and its label over a 1 s window (10 samples = 20 frames, ~ one
    # gait cycle, the label's own length) -- removes the gait sway from z as the 1 s label removes it from the motion
    # (F61); then raw vs per-body centred
    def win_avg(X, groups, n=10):
        out = np.empty_like(X)
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            cs = np.cumsum(np.r_[np.zeros((1, X.shape[1])), X[idx]], 0)
            for k, i in enumerate(idx):
                lo, hi = max(0, k - n // 2), min(len(idx), k - n // 2 + n)
                out[i] = (cs[hi] - cs[lo]) / (hi - lo)
        return out
    Zw, Fw = win_avg(Zs, G), win_avg(F, G)
    for tag, Zx in (("gait-averaged z", Zw), ("gait-averaged + per-body centred z", None)):
        if Zx is None:
            Zx = Zw.copy()
            for b in (0, 1, 2):
                Zx[B == b] -= Zx[B == b].mean(0)
        fw = RidgeCV(alphas=al).fit(Zx[B == 0], Fw[B == 0])
        for b in (1, 2):
            m = B == b; p = fw.predict(Zx[m]); t = Fw[m]
            r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
            L.append(f"(5) {tag}, c10 read-out on {SL.BODIES[b][0]}: R2 " + " / ".join(f"{v:+.2f}" for v in r2))
    # (6) GAIT-SHAKE REMOVAL as F155's note: per clip, find the stride frequency (dominant frequency of the clip's z,
    # first principal component, 0.5-4 Hz), fit sin / cos at it + 2 harmonics to every z dimension and subtract the fit
    # (clip mean and slow changes kept). Samples are every 2 frames (0.1 s).
    def deshake(X, groups, dt=0.1, nh=3):
        out = X.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g); Xc = X[idx] - X[idx].mean(0); n = len(idx)
            if n < 8:
                continue
            pc = np.linalg.svd(Xc, full_matrices=False)[2][0]; sig = Xc @ pc
            fr = np.fft.rfftfreq(n, dt); amp = np.abs(np.fft.rfft(sig - sig.mean()))
            ok = (fr >= 0.5) & (fr <= 4.0)
            if not ok.any():
                continue
            f0 = fr[ok][np.argmax(amp[ok])]; t = np.arange(n) * dt
            Hm = np.column_stack([fn(2 * np.pi * k * f0 * t) for k in range(1, nh + 1) for fn in (np.sin, np.cos)])
            coef, *_ = np.linalg.lstsq(Hm, Xc, rcond=None)
            out[idx] = X[idx] - Hm @ coef
        return out
    Zd = deshake(Zs, G)
    for tag, Zx in (("gait-shake removed (F155 method)", Zd), ("gait-shake removed + per-body centred", None)):
        if Zx is None:
            Zx = Zd.copy()
            for b in (0, 1, 2):
                Zx[B == b] -= Zx[B == b].mean(0)
        fd = RidgeCV(alphas=al).fit(Zx[B == 0], F[B == 0])
        for b in (1, 2):
            m = B == b; p = fd.predict(Zx[m]); t = F[m]
            r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
            L.append(f"(6) {tag}, c10 read-out on {SL.BODIES[b][0]}: R2 " + " / ".join(f"{v:+.2f}" for v in r2))
    out = os.path.join(ROOT, "results/eval", name, "z_channel_transfer.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
