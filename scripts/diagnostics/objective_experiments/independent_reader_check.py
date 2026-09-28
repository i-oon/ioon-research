"""Is the cycle term's gain real motion, or a code only the frozen ITM can read?

`lambda_cycle` scores the FTM through the same frozen ITM that later reads its rollouts at selection
time, so a high cycle score could mean the FTM writes whatever the ITM decodes rather than a
realistic future. This reads the FTM's predictions with a judge that never saw the FTM, the ITM or
the cycle term: a ridge regression from the REAL frame change (e_{t+1} - e_t, spatially pooled) to
the clip's Froude, fit on training clips of the pretraining body.

At each step on held-out clips, the full grid over states s and actions a with the true latent
z_a = ITM(e_a[t], e_a[t+1]):
    pred[s, a] = reader(FTM(e_s, z_a) - e_s)
  r_row     across-action correlation with the actions' true Froude, from a fixed state (what
            selection needs), averaged over states and steps
  act/state share of variance from the action vs state marginal
  real      the same reader on each action's REAL change e_a[t+1] - e_a[t]: the ceiling

    .venv/bin/python3 scripts/diagnostics/objective_experiments/independent_reader_check.py \\
        --ckpt ctrl_fz=wm/runs/beh24_ft_ctrl_fz/last.pt --ckpt cycle_fz=wm/runs/beh24_ft_cycle_fz/last.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import load_candidates  # noqa: E402
from rollout_state_action_anova import Models, corr  # noqa: E402

CH = ("forward", "lateral", "yaw")


def pool(delta, grid=16, q=2):
    """(n, tokens, dim) -> (n, q*q*dim): mean over each spatial quadrant of the token grid."""
    n, t, d = delta.shape
    x = delta.reshape(n, -1, grid, grid, d) if t % (grid * grid) == 0 else None
    if x is None:
        return delta.mean(1)
    x = x.mean(1)                                             # average over time tubelets if any
    s = grid // q
    parts = [x[:, i * s:(i + 1) * s, j * s:(j + 1) * s].mean((1, 2)) for i in range(q) for j in range(q)]
    return torch.cat(parts, -1)


def embeddings(dirpath, cache, emb_name):
    cands = load_candidates(dirpath, emb_name, per_condition=999)
    emb = torch.load(cache, map_location="cpu") if os.path.exists(cache) else {}
    missing = [c["path"] for c in cands if c["path"] not in emb]
    if missing:
        from vjepa2_encoder import VJEPA2FrameEncoder
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        for p in missing:
            emb[p] = encode_clip(enc, load(p, REGISTRY[emb_name])["frames"], 2).cpu().half()
        del enc
        torch.cuda.empty_cache()
        torch.save(emb, cache)
    motion = [np.asarray(load(c["path"], REGISTRY[emb_name])["body_motion"])[:, :3] for c in cands]
    return cands, emb, motion


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--train_dir", default="data/egocentric/beh24_c10f10t10_ego_flat_cleantrain")
    ap.add_argument("--eval_dir", default="data/egocentric/beh24_c10f10t10_ego_flat_cleanval")
    ap.add_argument("--train_cache", default="results/wm/cache/reader_hex_beh24train.pt")
    ap.add_argument("--eval_cache", default="results/wm/cache/anova_hex_beh24val.pt")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, E = args.device, args.embodiment

    tr_c, tr_e, tr_m = embeddings(os.path.join(ROOT, args.train_dir), os.path.join(ROOT, args.train_cache), E)
    ev_c, ev_e, ev_m = embeddings(os.path.join(ROOT, args.eval_dir), os.path.join(ROOT, args.eval_cache), E)

    # the judge: real frame change -> Froude, fit on training clips only, raw (uncentred) embeddings
    def real_xy(cands, emb, motion):
        X, Y = [], []
        for c, m in zip(cands, motion):
            e = emb[c["path"]].float()
            n = min(len(e) - 1, len(m))
            X.append(pool(e[1:n + 1] - e[:n]).numpy())
            Y.append(m[:n])
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = real_xy(tr_c, tr_e, tr_m)
    Xev, Yev = real_xy(ev_c, ev_e, ev_m)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    reader = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit((Xtr - mu) / sd, Ytr)
    pr = reader.predict((Xev - mu) / sd)
    r2 = 1 - ((pr - Yev) ** 2).sum(0) / ((Yev - Yev.mean(0)) ** 2).sum(0)
    print(f"independent reader (real delta -> Froude), held-out R2: "
          + "  ".join(f"{c} {v:.3f}" for c, v in zip(CH, r2)) + f"   alpha {reader.alpha_:.0f}")
    read = lambda d: reader.predict((pool(d.float().cpu()).numpy() - mu) / sd)

    n = len(ev_c)
    T = min(min(len(ev_e[c["path"]]) for c in ev_c), min(len(m) for m in ev_m)) - 2
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        md = Models(os.path.join(ROOT, path), E, ev_c[0]["actions"].shape[1], dev)
        rows = {k: [] for k in ("r_row", "act", "state", "real_r")}
        for t in range(0, T, args.step):
            e0 = torch.stack([ev_e[c["path"]][t].float() for c in ev_c])
            e1 = torch.stack([ev_e[c["path"]][t + 1].float() for c in ev_c])
            off = md.offset.float().reshape(e0.shape[1:]) if md.offset is not None else 0
            e0d, e1d = (e0 - off).to(dev), (e1 - off).to(dev)
            z = md.itm(e0d, e1d)
            truth = np.stack([m[t] for m in ev_m])
            P = np.stack([read(md.ftm_step(e0d[s:s + 1].expand(n, -1, -1), z) - e0d[s:s + 1])
                          for s in range(n)])                              # (state, action, ch)
            realp = read(e1 - e0)
            r_row, act, state, real_r = [], [], [], []
            for j in range(3):
                p = P[..., j]
                tot = p.var() + 1e-12
                r_row.append(np.nanmean([corr(p[s], truth[:, j]) for s in range(n)]))
                act.append(p.mean(0).var() / tot)
                state.append(p.mean(1).var() / tot)
                real_r.append(corr(realp[:, j], truth[:, j]))
            for k, v in zip(rows, (r_row, act, state, real_r)):
                rows[k].append(v)
        print(f"\n=== {name} ===  judged by the independent reader, {n} states x {n} actions")
        print(f"{'':10}" + "".join(f"{k:>10}" for k in rows))
        for j, c in enumerate(CH):
            print(f"{c:<10}" + "".join(f"{np.nanmean(np.asarray(v)[:, j]):>10.3f}" for v in rows.values()))
        del md
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
