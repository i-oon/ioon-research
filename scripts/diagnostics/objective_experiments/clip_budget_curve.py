"""How many clips of a new body does grounding actually need, and what does pretraining buy?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/clip_budget_curve.py

**Why this exists when F45 already ran a budget sweep.** F45 swept 1/3/5/7/9 clips, pretrained
against scratch, and concluded "insect pretraining is worth roughly 7x fewer target clips" -- but it
scored the FORWARD MODEL's rollout ratio. F189 then measured that this ratio does not predict
downstream success at all: B1's stage 1 made the rollout WORSE at every horizon and B1 still grounded
(stage 4 = 0.751), while gecko's stage 1 improved it and gecko failed (1.010). So F45's curve cannot
support the "9 clips is enough" claim the deck makes, because it never measured grounding.

This sweeps the same budgets and scores **the metric the claim is actually stated in**: the shared
Froude head's held-out ratio, and the per-channel rho on the projector path -- the number Slide 29
quotes as 0.572.

**The two arms differ only in whether OUR pretraining is present.**
  pretrained : ITM/FTM/body_head warm-started from `beh12_hexonly_stopgrad` (hexapod only, never B1)
  scratch    : the same three modules randomly re-initialised, everything else identical
The frozen V-JEPA2 encoder is present in BOTH arms -- it is not ours, and removing it would measure
a different question ("does video pretraining help") than the one asked here.

Equal optimisation per cell: `adapt()` samples one batch per step, so a 1-clip cell and a 9-clip
cell both get exactly `--steps` updates. Without that the budget and the compute would be confounded.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from diagnostics.cross_embodiment.finetune_ftm import adapt  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CHAN = ["forward", "lateral", "yaw"]


def fresh(mod):
    for m in mod.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hexonly_stopgrad/best.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_b1_ego_flat")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--cache", default="results/wm/cache/b1.pt")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--proj_epochs", type=int, default=300)
    ap.add_argument("--head_epochs", type=int, default=400)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    bmean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
    bstd = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    enc = None
    E, A, Y, G = [], [], [], []
    for i, p in enumerate(paths):
        c = load(p, REGISTRY[args.embodiment])
        if p not in cache:
            enc = enc or VJEPA2FrameEncoder(dtype=torch.float32)
            cache[p] = encode_clip(enc, c["frames"], 2).cpu().half()
        e = cache[p].float()
        m = np.asarray(c["body_motion"])[:, channels]
        a = np.asarray(c["actions"], np.float32)
        n = min(len(e) - 1, len(m) - 1, len(a))
        E.append(e[:n + 1])
        A.append(a[:n])
        Y.append(torch.tensor(m[:n], dtype=torch.float32))
        G.append(torch.full((n,), i))
    del enc
    torch.cuda.empty_cache()
    print(f"{len(paths)} clips cached\n")

    y_all = ((torch.cat(Y) - bmean) / bstd)
    g_all = torch.cat(G)
    ids = torch.unique(g_all)
    # held-out clips fixed across every cell, so budgets are compared on identical test data
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
    val_ids = ids[order[:max(1, int(0.2 * len(ids)))]]
    val = torch.isin(g_all, val_ids)
    # `ids` holds tensors; testing `tensor in set_of_ints` is always False, which silently
    # let held-out clips into the adaptation set. Compare ints to ints.
    _val = set(val_ids.tolist())
    train_pool = [i for i in ids.tolist() if i not in _val]
    print(f"{int(val.sum())} of {len(y_all)} transitions held out ({len(val_ids)} clips); "
          f"{len(train_pool)} clips available to adapt on\n")

    adim = A[0].shape[1]
    print(f"{'arm':<11}{'clips':>6}{'seed':>5}{'stage4 held-out':>17}"
          + "".join(f"{c:>10}" for c in CHAN) + f"{'median rho':>12}")
    print("-" * 88)
    results = {}
    for arm in ("pretrained", "scratch"):
        for budget in args.budgets:
            cells = []
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                pick = rng.choice(train_pool, size=min(budget, len(train_pool)), replace=False)

                itm = InverseTransitionModel(cfg).to(dev)
                ftm = ForwardTransitionModel(cfg).to(dev)
                itm.load_state_dict(ck["itm"])
                ftm.load_state_dict(ck["ftm"])
                md = MotionDecoder(cfg, {args.embodiment: adim}).to(dev)
                md.load_state_dict(ck["md"], strict=False)
                if arm == "scratch":
                    fresh(itm), fresh(ftm), fresh(md.body_head)

                # ---- stage 1
                adapt(itm, ftm, [E[i] for i in pick], args.steps, 1e-4, seed, dev)
                for p_ in list(itm.parameters()) + list(ftm.parameters()):
                    p_.requires_grad_(False)
                itm.eval()

                # ---- z from the adapted ITM, on every clip
                with torch.no_grad():
                    z = torch.cat([torch.cat([itm(E[i][t:t + 1].to(dev), E[i][t + 1:t + 2].to(dev))
                                              for t in range(len(A[i]))]) for i in range(len(A))])
                X = torch.tensor(np.concatenate(A))

                # ---- stage 2: projector
                proj = ActionProjector(cfg, {args.embodiment: adim}).to(dev)
                proj.set_stats(args.embodiment, X[~val].mean(0), X[~val].std(0))
                o1 = torch.optim.Adam(proj.parameters(), lr=1e-3)
                Xd, vd = X.to(dev), val.to(dev)
                for _ in range(args.proj_epochs):
                    o1.zero_grad()
                    nn.functional.mse_loss(proj(Xd[~vd], args.embodiment), z[~vd]).backward()
                    o1.step()
                proj.eval()
                with torch.no_grad():
                    zp = proj(Xd, args.embodiment)

                # ---- stage 4: body head on the union, exactly as the pipeline does
                for p_ in md.parameters():
                    p_.requires_grad_(False)
                for p_ in md.body_head.parameters():
                    p_.requires_grad_(True)
                zu, yu = torch.cat([z, zp.detach()]), torch.cat([y_all, y_all]).to(dev)
                vu = torch.cat([vd, vd])
                o2 = torch.optim.Adam(md.body_head.parameters(), lr=1e-3)
                for _ in range(args.head_epochs):
                    o2.zero_grad()
                    nn.functional.mse_loss(md.body(None, zu[~vu]), yu[~vu]).backward()
                    o2.step()
                md.eval()
                with torch.no_grad():
                    err = nn.functional.mse_loss(md.body(None, zu[vu]), yu[vu]).item()
                    base = nn.functional.mse_loss(
                        yu[vu].mean(0, keepdim=True).expand_as(yu[vu]), yu[vu]).item()
                    pred = md.body(None, zp[vd]).cpu().numpy()
                truth = y_all[val].numpy()
                rhos = [spearmanr(pred[:, c], truth[:, c]).statistic for c in range(3)]
                cells.append((err / base, rhos))
                del itm, ftm, md, proj, z, zp, zu
                torch.cuda.empty_cache()

            ratio = float(np.mean([c[0] for c in cells]))
            per = np.mean([c[1] for c in cells], axis=0)
            results[(arm, budget)] = (ratio, per)
            print(f"{arm:<11}{budget:>6}{'avg':>5}{ratio:>17.3f}"
                  + "".join(f"{v:>+10.3f}" for v in per) + f"{np.median(per):>+12.3f}")
        print()

    print("\n=== what pretraining buys, per budget ===")
    print(f"{'clips':>6}{'pretrained rho':>16}{'scratch rho':>13}{'gap':>8}")
    for b in args.budgets:
        p = np.median(results[("pretrained", b)][1])
        s = np.median(results[("scratch", b)][1])
        print(f"{b:>6}{p:>+16.3f}{s:>+13.3f}{p - s:>+8.3f}")


if __name__ == "__main__":
    main()
