"""Stage-2-only refit: does a projector actually FIT on B1 babble's own actions read candidates
correctly, instead of testing the trained-policy-fit projector zero-shot on a different action
distribution (the mistake the previous version of this test made)?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/refit_b1_babble_projector.py

**Why this is the right procedure, not a new idea.** This is exactly F189's own playbook for
gecko: stage 1 (ITM/FTM adaptation) is about VISUAL dynamics and does not need to change --
B1's ITM/FTM are already adapted (`wm/runs/b1_adapt/`) and that adaptation has nothing to do with
which action distribution feeds the projector. Stage 2 (the projector, action -> z) is what is
specific to an action distribution, and it was only ever fit on B1's trained-policy actions
(`data/egocentric/beh12_b1_ego_flat`). Applying that projector zero-shot to CPG babble's actions
-- a completely different region of action space (measured: calf mean +2.3 policy vs +0.3 babble)
-- is the same zero-shot-fails pattern this project has hit before (F181 gecko zero-shot), not
evidence about babble itself. `body_head` and the ITM stay FROZEN and UNCHANGED throughout --
only a fresh projector is fit, mirroring gecko's own stage-2 refit exactly.

Held out by clip (not frame), same discipline as every other stage-2/4 fit in this project, to
avoid the near-duplicate-frame leak F189/F76 already document.
"""
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

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CKPT = os.path.join(ROOT, "wm/runs/b1_adapt/body_head_b1.pt")
BABBLE_DIR = os.path.join(ROOT, "results/wm/dataset/b1_babble/batch_rendered")
EXPERT_DIR = os.path.join(ROOT, "data/egocentric/beh12_b1_ego_flat")
GOAL_DIR = os.path.join(ROOT, "data/egocentric/beh12_c10f10t10_ego_flat")
CACHE = os.path.join(ROOT, "results/wm/cache/b1_babble_rendered.pt")
EMB = "b1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    bmean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
    bstd = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)

    itm = InverseTransitionModel(cfg).to(dev).eval()
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    md = MotionDecoder(cfg, {EMB: 12}).to(dev).eval()
    md.load_state_dict(ck["md"], strict=False)
    for p in md.parameters():
        p.requires_grad_(False)

    paths = sorted(glob.glob(os.path.join(BABBLE_DIR, "*.npz")))
    cache = torch.load(CACHE, map_location="cpu") if os.path.exists(CACHE) else {}
    enc = None
    Z, A, Y, G = [], [], [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            c = load(p, REGISTRY[EMB])
            if p not in cache:
                enc = enc or VJEPA2FrameEncoder(dtype=torch.float32)
                cache[p] = encode_clip(enc, c["frames"], 2).cpu().half()
            e = cache[p].float().to(dev)
            m = np.asarray(c["body_motion"])[:, channels]
            a = np.asarray(c["actions"], np.float32)
            n = min(len(e) - 1, len(m), len(a))
            Z.append(torch.cat([itm(e[t:t + 1], e[t + 1:t + 2]) for t in range(n)]).cpu())
            A.append(a[:n])
            Y.append(torch.tensor(m[:n], dtype=torch.float32))
            G.append(torch.full((n,), i))
    if enc is not None:
        torch.save(cache, CACHE)
        del enc
        torch.cuda.empty_cache()

    z = torch.cat(Z).to(dev)
    y = ((torch.cat(Y) - bmean[:len(channels)]) / bstd[:len(channels)]).to(dev)
    X = torch.tensor(np.concatenate(A)).to(dev)
    group = torch.cat(G)

    ids = torch.unique(group)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))
    val_ids = ids[order[:max(1, int(0.2 * len(ids)))]]
    val = torch.isin(group, val_ids)
    print(f"{len(ids)} babble clips, {int(val.sum())} of {len(z)} transitions held out "
          f"({len(val_ids)} clips)\n")

    # ---- stage 2: fit a FRESH projector on babble's own actions, ITM/body_head frozen ----
    proj = ActionProjector(cfg, {EMB: 12}).to(dev)
    proj.set_stats(EMB, X[~val].mean(0), X[~val].std(0))
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)
    for ep in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(proj(X[~val], EMB), z[~val])
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1:4d}  train {loss.item():.4f}")
    proj.eval()
    with torch.no_grad():
        zp = proj(X, EMB)
        mse = nn.functional.mse_loss(zp[val], z[val]).item()
        base = nn.functional.mse_loss(z[~val].mean(0, keepdim=True).expand_as(z[val]), z[val]).item()
    print(f"\nstage-2 z-MSE ratio (babble-fit projector): {mse/base:.3f}  "
          f"(below 1.0 beats predicting the mean)\n")

    # ---- evaluate: forward-Froude rho on held-out babble, babble-fit projector vs body_head ----
    with torch.no_grad():
        pred_proj = md.body(None, zp[val]).cpu().numpy()
        pred_itm = md.body(None, z[val]).cpu().numpy()
    truth = y[val].cpu().numpy()
    for name, pred in (("proj(babble_action), NEW projector", pred_proj),
                       ("itm(e_t,e_next), for reference", pred_itm)):
        rho = spearmanr(pred[:, 0], truth[:, 0]).statistic
        print(f"{name:<38} held-out forward rho = {rho:+.3f}")

    # ---- the actual comparison test: candidate scoring with the NEW projector ----
    print("\n=== candidate scoring, babble-fit projector vs the original policy-fit one ===")

    def score_pool(pool_paths, proj_to_use, tag):
        cands = []
        for p in pool_paths:
            c = load(p, REGISTRY[EMB])
            a = torch.tensor(c["actions"], dtype=torch.float32, device=dev)
            with torch.no_grad():
                zc = proj_to_use(a, EMB)
                pred = md.body(None, zc).mean(0).cpu().numpy() * bstd[:3].numpy() + bmean[:3].numpy()
            true = np.asarray(c["body_motion"])[:, :3].mean(0)
            cands.append({"path": p, "pred": pred, "true": true,
                         "fam": int(np.argmax(np.abs(true)))})

        by_cond = {}
        for p in sorted(glob.glob(os.path.join(GOAL_DIR, "*.npz"))):
            with np.load(p, allow_pickle=True) as d:
                cond = str(d["condition"])
            by_cond.setdefault(cond, p)
        goals = []
        for cond, p in sorted(by_cond.items()):
            fr = np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3].mean(0)
            goals.append(fr)

        fams = np.array([c["fam"] for c in cands])
        hits, dists = 0, []
        for g in goals:
            gfam = int(np.argmax(np.abs(g)))
            pick = cands[int(np.argmin([np.abs(c["pred"] - g).sum() for c in cands]))]
            hits += int(pick["fam"] == gfam)
            dists.append(np.abs(pick["true"] - g).sum())
        chance = np.mean([(fams == int(np.argmax(np.abs(g)))).mean() for g in goals])
        print(f"{tag:<30} n={len(cands):<3} accuracy={hits/len(goals):.0%}  "
              f"chance={chance:.0%}  mean pick error={np.mean(dists):.4f}")

    expert_paths = sorted(glob.glob(os.path.join(EXPERT_DIR, "*.npz")))
    by_cond = {}
    for p in expert_paths:
        with np.load(p, allow_pickle=True) as d:
            by_cond.setdefault(str(d["condition"]), p)
    score_pool(list(by_cond.values()), proj, "expert (new-proj sanity)")
    score_pool(paths, proj, "babble (NEW babble-fit projector)")


if __name__ == "__main__":
    main()
