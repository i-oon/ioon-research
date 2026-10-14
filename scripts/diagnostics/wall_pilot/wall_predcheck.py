"""Wall prediction check D1-D3 on the balanced wall data, FROZEN world model (protocol LOCKED 2026-10-14 in
`doc/WALL_SELECTION_PLAN.md` section 9; the F334 / F338 sets keep `b1_wall_predcheck.py`).

Per body (one per call; the c10 collector re-sets the shared wall settings on import):
  D1  does the REAL future embedding carry the wall outcome?        R_real (train real frames) on true e_{t+i}
  D2  does the frozen imagination carry it as real frames do?      R_real unchanged on imagined e^_i (+ shift)
  D3  is it decodable from the imagined embeddings?                R_imag,i (train imagined e^_i, detached)
  references: copy (R_real(e_t), one value per state), command-only (R_real(e_t) + the train mean change of that command),
  command-only with the TRUE start clearance (privileged).
Common 24 commands (indices 0-23); clearance in body lengths; imagined step i = 0.25 i s, i = 1..5; branches that
crossed before a step are masked there (ranking: worse than every present command, value = last stored clearance);
detection = NEAR-or-cross within 3 s, scored by the minimum predicted clearance over steps 1..i. Bootstrap over episodes.

    .venv/bin/python3 scripts/diagnostics/wall_pilot/wall_predcheck.py --body b1  --ckpt round1_branches_s0_gray_imread
    .venv/bin/python3 scripts/diagnostics/wall_pilot/wall_predcheck.py --body c10 --ckpt round1_branches_s0_gray_imread
    -> results/check/wall_predcheck_v3/<body>_<model>.txt (+ _emb.pkl cache: re-analysis without re-encoding)
"""
import argparse
import collections
import glob
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/dataset", "scripts/diagnostics/objective_experiments", "sim/collect", "sim/control",
          "sim/render"):
    sys.path.insert(0, os.path.join(ROOT, p))
CHK = os.path.join(ROOT, "results/check/wall_predcheck_v3")
STEPS = (1, 2, 3, 4, 5)
N_CMD = 24
DET_FRAMES = 60                                   # NEAR-or-cross within 3 s after the branch point


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return np.nan
    return float(spearmanr(a, b).correlation)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


@torch.no_grad()
def collect(m, enc, W, split, BL, body, fit_rows):
    """per branch file (commands 0-23): pooled e_t, real e_{t+i}, imagined e^_i, clearance (body lengths); for train also
    every 5th real row (R_real fit)"""
    from wm.evaluate import encode_clip
    import check_wall_data as G
    akey = "actions" if body == "c10" else "action"
    D = os.path.join(W.DATA, split)
    files = sorted(f for f in glob.glob(os.path.join(D, "ep*_b*_c*.npz"))
                   if int(os.path.basename(f).split("_c")[1][:2]) < N_CMD)
    out, fitX, fitY = [], [], []
    for n, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        P = int(d["first_pair"]); T = P - 1
        cl = np.asarray(d["clearance"], float) / BL
        nrow = len(cl)
        rows = [T] + [T + 5 * i for i in STEPS if T + 5 * i < nrow]
        if fit_rows:
            rows = sorted(set(rows) | set(range(T % 5, nrow, 5)))
        E = encode_clip(enc, np.asarray(d["frames"])[rows], 8).float()
        if m.offset is not None:
            from wm.evaluate import offset_as
            E = E - offset_as(m.offset.float(), E.shape[1:])
        pooled = {r: E[j].mean(0).cpu().numpy() for j, r in enumerate(rows)}
        if fit_rows:
            for r in rows:
                fitX.append(pooled[r]); fitY.append(cl[r])
        A = np.asarray(d[akey], np.float32)
        e = E[rows.index(T)][None].cuda()
        imag = {}
        for i in STEPS:
            s0 = T + 1 + 5 * (i - 1)
            ch = A[min(s0, len(A) - 1):min(s0, len(A) - 1) + 5]
            if len(ch) < 5:
                ch = np.concatenate([ch, np.repeat(ch[-1:], 5 - len(ch), 0)])
            e = m.ftm_step(e, m.proj(torch.as_tensor(ch[None], device="cuda"), m.embodiment))
            imag[i] = e.float().mean(1).cpu().numpy()[0]
        grp, side, cs = G.state_geometry(W, W.F, d, P)
        crossed_at = int(np.argmax(cl < 0)) if (cl < 0).any() else -1
        out.append(dict(file=os.path.basename(f), ep=int(d["episode"]), state=os.path.basename(f).rsplit("_c", 1)[0],
                        cmd=int(d["cf_command_index"]), band=W.band_of(float(d["start_clearance"])), grp=grp,
                        e0=pooled[T], real={i: pooled.get(T + 5 * i) for i in STEPS}, imag=imag,
                        true={i: (cl[T + 5 * i] if T + 5 * i < nrow else np.nan) for i in STEPS},
                        last=float(cl[-1]), c0=float(cl[T]),
                        det=bool((cl[P:P + DET_FRAMES] < float(d["margin"]) / BL).any()), crossed_at=crossed_at))
        if (n + 1) % 200 == 0:
            print(f"  {split}: {n + 1}/{len(files)} files", flush=True)
    return out, (np.asarray(fitX), np.asarray(fitY))


def by_state(recs):
    g = collections.defaultdict(list)
    for r in recs:
        g[r["state"]].append(r)
    return {k: sorted(v, key=lambda r: r["cmd"]) for k, v in g.items()}


def main():
    from sklearn.linear_model import RidgeCV
    from rollout_state_action_anova import Models
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=("b1", "c10"))
    ap.add_argument("--ckpt", default="round1_branches_s0_gray_imread")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.body == "c10":
        import collect_c10_wall_data as C
        W = C.W; BL = C.BL; emb, adim, ck = "hexapod", 18, "hex.pt"
    else:
        import collect_b1_wall_data as W
        BL = 1.13; emb, adim, ck = "b1", 12, "b1.pt"
    os.makedirs(CHK, exist_ok=True)
    cache = os.path.join(CHK, f"{a.body}_{a.ckpt}_emb.pkl")
    m = Models(os.path.join(ROOT, "results/eval", a.ckpt, "ckpt", ck), emb, adim, "cuda")   # gray switch before encoder
    if os.path.exists(cache):
        data = pickle.load(open(cache, "rb"))
    else:
        from vjepa2_encoder import VJEPA2FrameEncoder
        print("VJEPA_GRAY =", os.environ.get("VJEPA_GRAY"), "| offset", m.offset is not None, flush=True)
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        data = {}
        for split in ("train", "val", "test"):
            data[split] = collect(m, enc, W, split, BL, a.body, fit_rows=(split == "train"))
        pickle.dump(data, open(cache + ".tmp", "wb")); os.replace(cache + ".tmp", cache)
    tr, (Xr, yr) = data["train"]
    alphas = np.logspace(-1, 6, 15)
    R_real = RidgeCV(alphas=alphas).fit(Xr, yr)
    R_imag = {}
    for i in STEPS:
        ok = [r for r in tr if np.isfinite(r["true"][i])]
        R_imag[i] = RidgeCV(alphas=alphas).fit(np.stack([r["imag"][i] for r in ok]), np.array([r["true"][i] for r in ok]))
    dchg = {}                                                       # train mean change of true clearance per command
    for i in STEPS:
        acc = collections.defaultdict(list)
        for r in tr:
            if np.isfinite(r["true"][i]):
                acc[r["cmd"]].append(r["true"][i] - r["c0"])
        dchg[i] = np.array([np.mean(acc[c]) if acc[c] else 0.0 for c in range(N_CMD)])

    def preds(name, recs, i):
        """per-file prediction of clearance at step i (body lengths)"""
        if name == "real-future":
            return np.array([R_real.predict(r["real"][i][None])[0] if r["real"][i] is not None else np.nan for r in recs])
        if name == "D2 imagined, R_real":
            return R_real.predict(np.stack([r["imag"][i] for r in recs]))
        if name == "D3 imagined, R_imag":
            return R_imag[i].predict(np.stack([r["imag"][i] for r in recs]))
        c0 = R_real.predict(recs[0]["e0"][None])[0]
        if name == "copy":
            return np.full(len(recs), c0)
        if name == "command-only":
            return np.array([c0 + dchg[i][r["cmd"]] for r in recs])
        if name == "command-only, true start":
            return np.array([r["c0"] + dchg[i][r["cmd"]] for r in recs])

    ROWS = ("real-future", "D2 imagined, R_real", "D3 imagined, R_imag", "copy", "command-only", "command-only, true start")
    L = [f"{a.body} wall prediction check D1-D3 (LOCKED protocol, WALL_SELECTION_PLAN 9); model {a.ckpt} frozen; common "
         f"24 commands; clearance in body lengths (BL {BL}); fit on train only",
         f"states: train {len(by_state(tr))} | val {len(by_state(data['val'][0]))} | test {len(by_state(data['test'][0]))};"
         f" R_real alpha {R_real.alpha_:.3g} on {len(yr)} train real frames"]
    rng = np.random.default_rng(0)
    for split in ("test", "val"):
        S = by_state(data[split][0])
        eps = sorted({v[0]["ep"] for v in S.values()})
        L.append(f"=== {split} ({len(S)} states, {len(eps)} episodes)")
        per = {}                                                   # (row, i) -> per-state (ep, spearman, auc_within)
        for i in STEPS:
            L.append(f"--- step {i} = {0.25 * i:.2f} s | Spearman within state | MAE BL | detection AUC pooled / within "
                     f"(NEAR-or-cross within 3 s, score = min predicted over steps 1..{i})")
            for rn in ROWS:
                sp, err, aw, Y, Sc = [], [], [], [], []
                for k, recs in S.items():
                    if len(recs) < N_CMD:
                        continue
                    pr = {j: preds(rn, recs, j) for j in range(1, i + 1)}
                    p_i = pr[i]
                    # true value; a branch that crossed before step i has no frame there: ranked below every present
                    # command (its last stored clearance, <= 0), masked in MAE
                    tv = np.array([r["true"][i] if np.isfinite(r["true"][i]) else min(r["last"], 0.0) for r in recs])
                    if rn == "real-future":                          # no real frame after a crossing: rank it lowest
                        p_i = np.where(np.isfinite(p_i), p_i, np.nanmin(p_i) - 1.0 if np.isfinite(p_i).any() else 0.0)
                    sp.append(spearman(p_i, tv))
                    msk = np.array([np.isfinite(r["true"][i]) for r in recs]) & np.isfinite(p_i)
                    err.append(np.abs(p_i[msk] - tv[msk]).mean())
                    pj = [np.where(np.isfinite(pr[j]), pr[j], (np.nanmin(pr[j]) - 1.0) if np.isfinite(pr[j]).any() else 0.0)
                          for j in pr]
                    score = -np.min(np.stack(pj), 0)                  # riskier = lower minimum predicted clearance
                    y = [r["det"] for r in recs]
                    aw.append(auc(y, score)); Y += y; Sc += list(score)
                    per.setdefault((rn, i), []).append((recs[0]["ep"], sp[-1], aw[-1]))
                spm = np.nanmean(sp) if np.isfinite(sp).any() else np.nan
                L.append(f"  {rn:26s} | {spm:+.3f} | {np.nanmean(err):.3f} | {auc(Y, Sc):.3f} / "
                         f"{np.nanmean(aw) if np.isfinite(aw).any() else np.nan:.3f}")
            sh = np.array([R_real.predict(r["imag"][i][None])[0] - R_real.predict(r["real"][i][None])[0]
                           for recs in S.values() for r in recs if r["real"][i] is not None])
            L.append(f"  real-vs-imagined shift (R_real(imag) - R_real(real)): mean {sh.mean():+.3f} BL, std {sh.std():.3f}")

        def boot(rA, rB, col):
            """episode-bootstrap 95 % CI of mean(A) - mean(B) at step 5 (col 1 = Spearman, 2 = within AUC)"""
            A_ = collections.defaultdict(list); B_ = collections.defaultdict(list)
            for ep, *v in per[(rA, 5)]:
                A_[ep].append(v[col - 1])
            for ep, *v in per[(rB, 5)]:
                B_[ep].append(v[col - 1])
            ds = []
            for _ in range(a.boot):
                pick = rng.choice(eps, len(eps))
                av = np.concatenate([A_[e] for e in pick]); bv = np.concatenate([B_[e] for e in pick])
                ds.append(np.nanmean(av) - np.nanmean(bv))
            return np.nanmean(np.concatenate(list(A_.values()))) - np.nanmean(np.concatenate(list(B_.values()))), \
                np.percentile(ds, 2.5), np.percentile(ds, 97.5)
        if split == "test":
            L.append("=== PASS rules (test, 1.25 s; 95 % episode bootstrap)")
            d1 = boot("real-future", "command-only", 1)
            L.append(f"D1 Spearman real-future - command-only: {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] -> "
                     f"{'PASS' if d1[1] > 0 else 'FAIL'}")
            d3 = boot("D3 imagined, R_imag", "command-only", 1)
            d3a = boot("D3 imagined, R_imag", "command-only", 2)
            L.append(f"D3 Spearman imagined - command-only: {d3[0]:+.3f} [{d3[1]:+.3f}, {d3[2]:+.3f}]; within-state "
                     f"detection AUC imagined - command-only: {d3a[0]:+.3f} [{d3a[1]:+.3f}, {d3a[2]:+.3f}]")
            pooled = {}
            for rn in ("D3 imagined, R_imag", "copy"):
                Y, Sc = [], []
                for recs in S.values():
                    if len(recs) < N_CMD:
                        continue
                    pj = np.stack([preds(rn, recs, j) for j in STEPS]); Y += [r["det"] for r in recs]; Sc += list(-pj.min(0))
                pooled[rn] = auc(Y, Sc)
            ok3 = d3[1] > 0 and d3a[0] > 0 and pooled["D3 imagined, R_imag"] > pooled["copy"]
            L.append(f"D3 pooled detection AUC imagined {pooled['D3 imagined, R_imag']:.3f} vs copy {pooled['copy']:.3f} -> "
                     f"D3 {'PASS' if ok3 else 'FAIL'} (needs: Spearman lower bound > 0, within-AUC gain > 0, pooled > copy)")
    txt = "\n".join(L)
    out = os.path.join(CHK, f"{a.body}_{a.ckpt}.txt")
    open(out, "w").write(txt + "\n"); print(txt); print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
