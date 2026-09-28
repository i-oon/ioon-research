"""Is z a shared action space across bodies? Figure + the numbers that decide it (not the picture).

Pre-registered expectation (2026-09-26, before running): at stride 1, z separates by BODY (a body can
be identified from z, a Froude read-out fit on the hexapod does not transfer); at stride 5, z groups by
BEHAVIOUR across bodies (body identification near chance, the hexapod-fit read-out transfers). F262 is
the indirect evidence for this.

Latents: z = ITM(e_t, e_{t+k}) with each model's B1-adapted ITM (`ckpt_lib_s4.pt`, the ITM the B1 test
uses), on three bodies' cached egocentric clips: hexapod c10f10t10 (beh24 val, 24 clips), hexapod
c08f09t09 (beh12, 48), B1 (beh12 library, 24).

Numbers (decide the question; the picture only illustrates them):
  body-ID acc   logistic regression z -> body, grouped CV by clip, bodies subsampled to equal size
                (chance 0.33; lower = more shared)
  Froude xfer   ridge z -> Froude fit on c10 only; R2 on c08 and on B1 (higher = more shared)
  kNN mixing    share of each point's 10 nearest neighbours (standardised z) from another body,
                divided by that share under random mixing (1 = fully mixed, 0 = separated)
Pictures: PCA (deterministic) and UMAP with THREE seeds side by side (no seed chosen), coloured by
body and by behaviour family.

    .venv/bin/python3 scripts/figures/shared_latent_figure.py
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.policy.planner import condition_of, load_candidates  # noqa: E402

BODIES = [("c10 (pretrain)", "hexapod", "data/egocentric/beh24_c10f10t10_ego_flat_cleanval",
           "results/wm/cache/anova_hex_beh24val.pt"),
          ("c08 (held-out)", "hexapod", "data/egocentric/beh12_c08f09t09_ego_flat",
           "results/wm/cache/selection_eval_c08.pt"),
          ("B1", "b1", "data/egocentric/beh12_b1_ego_flat_cleantrain",
           "results/wm/cache/selection_eval_cands.pt")]
MODELS = [("stride 1", "wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_s4.pt"),
          ("stride 5", "wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt")]


def family(cond):
    c = str(cond)
    if c.startswith("side_L"):
        return "side L"
    if c.startswith("side_R"):
        return "side R"
    if c.startswith("turn"):
        return "turn -" if c.endswith("_neg") else "turn +"
    if "-" in c or c.endswith("_bwd"):
        return "backward"
    return "forward"


@torch.no_grad()
def latents(model_path, dev):
    Z, B, F, G, C = [], [], [], [], []
    for bi, (name, emb, d, cache) in enumerate(BODIES):
        m = Models(os.path.join(ROOT, model_path), emb, 18 if emb == "hexapod" else 12, dev)
        k = m.stride
        E = torch.load(os.path.join(ROOT, cache), map_location="cpu")
        for ci, c in enumerate(load_candidates(os.path.join(ROOT, d), emb, per_condition=999)):
            e = E[c["path"]].float()
            if m.offset is not None:
                e = e - m.offset.float().reshape(e.shape[1:])
            bm = np.asarray(load(c["path"], REGISTRY[emb])["body_motion"])[:, :3]
            ts = list(range(0, min(len(e), len(bm)) - k, 2))
            z = torch.cat([m.itm(e[ts[i:i + 16]].to(dev), e[[t + k for t in ts[i:i + 16]]].to(dev)).cpu()
                           for i in range(0, len(ts), 16)])
            Z.append(z.numpy()); B += [bi] * len(ts); C += [family(condition_of(c["path"]))] * len(ts)
            F.append(np.stack([bm[t:t + k].mean(0) for t in ts])); G += [f"{bi}_{ci}"] * len(ts)
        del m
    return np.concatenate(Z), np.array(B), np.concatenate(F), np.array(G), np.array(C)


def numbers(Z, B, F, G, rng):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    n = min(np.bincount(B))
    idx = np.concatenate([rng.choice(np.where(B == b)[0], n, replace=False) for b in np.unique(B)])
    Zs = StandardScaler().fit_transform(Z)
    acc = []
    for tr, te in GroupKFold(5).split(Zs[idx], B[idx], G[idx]):
        clf = LogisticRegression(max_iter=2000).fit(Zs[idx][tr], B[idx][tr])
        acc.append((clf.predict(Zs[idx][te]) == B[idx][te]).mean())
    r = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Zs[B == 0], F[B == 0])
    xfer = {b: 1 - ((r.predict(Zs[B == b]) - F[B == b]) ** 2).sum(0) / ((F[B == b] - F[B == b].mean(0)) ** 2).sum(0)
            for b in (1, 2)}
    nn = NearestNeighbors(n_neighbors=11).fit(Zs[idx])
    nb = nn.kneighbors(Zs[idx], return_distance=False)[:, 1:]
    other = (B[idx][nb] != B[idx][:, None]).mean()
    return float(np.mean(acc)), xfer, float(other / (2 / 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/deck/shared_latent")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    body_col = ["#1565c0", "#2e7d32", "#c62828"]
    fams = ["forward", "backward", "turn +", "turn -", "side L", "side R"]
    fam_col = dict(zip(fams, ["#1b5e20", "#b71c1c", "#e65100", "#f9a825", "#0d47a1", "#6a1b9a"]))
    rows = []
    for mname, mpath in MODELS:
        Z, B, F, G, C = latents(mpath, args.device)
        acc, xfer, mix = numbers(Z, B, F, G, np.random.default_rng(0))
        rows.append((mname, acc, xfer, mix))
        Zs = StandardScaler().fit_transform(Z)
        embeds = [("PCA", PCA(2, random_state=0).fit_transform(Zs))] + \
                 [(f"UMAP seed {s}", umap.UMAP(random_state=s).fit_transform(Zs)) for s in (0, 1, 2)]
        fig, ax = plt.subplots(2, 4, figsize=(18, 8.5))
        for j, (title, Y) in enumerate(embeds):
            for b in range(3):
                ax[0, j].scatter(*Y[B == b].T, s=3, c=body_col[b], alpha=0.5, label=BODIES[b][0])
            for f in fams:
                ax[1, j].scatter(*Y[C == f].T, s=3, c=fam_col[f], alpha=0.5, label=f)
            ax[0, j].set_title(title); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
            ax[1, j].set_xticks([]); ax[1, j].set_yticks([])
        ax[0, 0].set_ylabel("coloured by BODY"); ax[1, 0].set_ylabel("coloured by BEHAVIOUR")
        ax[0, 0].legend(markerscale=4, fontsize=8); ax[1, 0].legend(markerscale=4, fontsize=8, ncol=2)
        fig.suptitle(f"{mname}: z = ITM(e_t, e_t+k) of three bodies  |  body-ID acc {acc:.2f} (chance 0.33)  "
                     f"| Froude read-out fit on c10 -> c08 R2 {xfer[1].mean():+.2f}, B1 R2 {xfer[2].mean():+.2f}  "
                     f"| kNN mixing {mix:.2f}", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"latent_{mname.replace(' ', '')}.png"), dpi=110)
        plt.close(fig)
    print(f"{'model':<10}{'body-ID acc':>13}{'c08 xfer R2 (fwd/lat/yaw)':>30}{'B1 xfer R2 (fwd/lat/yaw)':>30}{'kNN mix':>10}")
    for mname, acc, xfer, mix in rows:
        f = lambda v: " / ".join(f"{x:+.2f}" for x in v)
        print(f"{mname:<10}{acc:>13.2f}{f(xfer[1]):>30}{f(xfer[2]):>30}{mix:>10.2f}")
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
