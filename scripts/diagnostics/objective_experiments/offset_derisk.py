"""Does the frozen per-embodiment offset track a drifting FTM, or was the diagnosis wrong?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/offset_derisk.py

`state_head_transfer_confirm.py` found leak 0.704 pre-training -> 0.947 after 2000 steps of PLAIN
L_recon (no state head at all) -- so the drift is the FTM's, not the state head's. Before proposing
periodic re-fitting as the cure, three things need confirming, because the numbers carry an
ambiguity that changes the fix entirely.

**0. Reconcile 0.704 against the offline 0.464.** `offset_fix_check.py` reported 0.464 on this same
un-fine-tuned checkpoint; this script's own pre-training number was 0.704. The two scripts split
clips differently -- `offset_fix_check.py` shuffles all 96 clips together (test composition across
embodiments can land uneven), this file's harness shuffles each embodiment's 48 separately (a
stratified 70/30 per body). Both are run here, on the identical base checkpoint, so the gap is
attributed to the split rather than assumed.

**1. Does the frozen offset still work on the UN-fine-tuned checkpoint?** Re-measured directly,
under both split methodologies, so the baseline this whole confirm chain rests on is nailed down
before asking whether it degrades.

**2. Does a FRESH offset (re-fit on train clips of the FINE-TUNED model, same frozen-buffer
discipline) bring leak back down on that model's own held-out clips?** This is the one gate
everything else depends on: recompute is worth building only if a fresh fit actually works on the
moved model.

**3. How far does the offset move?** Cosine similarity and relative norm change between the
original and the freshly-fit offset, per embodiment -- a slow drift and a reshaped identity signal
need different fixes, and "leak dropped in (2)" does not by itself distinguish them.

**Diagnosis only.** Trains the same L_recon-only FTM `state_head_transfer_confirm.py`'s control arm
already used (same seed, same steps, so this reproduces rather than duplicates that arm) and adds
measurements it did not take.
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
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
    ap.add_argument("--caches", nargs="+",
                    default=["results/wm/cache/ego_hex.pt", "results/wm/cache/ego_b1.pt"])
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_ftm", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    sources = [tuple(s.split("=", 1)) for s in args.sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    base_ftm = ForwardTransitionModel(cfg).to(device).eval()
    base_ftm.load_state_dict(ck["ftm"])

    clips_by_e, paths_by_e, cid_all, eid_all_data = {}, {}, [], []
    offset_ctr = 0
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, args.caches[eid]), map_location="cpu", mmap=True)
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        clips_by_e[eid], paths_by_e[eid] = clips, paths
        print(f"  {name}: {len(clips)} clips", flush=True)

    # ---- STRATIFIED split: each embodiment's clips shuffled and split independently -----------
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = {}, {}
    for eid in range(len(names)):
        order = list(range(len(clips_by_e[eid])))
        rng.shuffle(order)
        n_te = max(1, int(len(order) * args.test_frac))
        test_idx[eid] = set(order[:n_te])
        train_idx[eid] = set(order[n_te:])

    def gather_pts(eid, ids):
        out = []
        clips, paths = clips_by_e[eid], paths_by_e[eid]
        for ci in ids:
            e = clips[ci]["e"].float()
            for t in range(1, len(e) - 2, args.stride):
                out.append((ci, t))
        return out

    train_pts = {eid: gather_pts(eid, train_idx[eid]) for eid in range(len(names))}
    test_pts = {eid: gather_pts(eid, test_idx[eid]) for eid in range(len(names))}

    @torch.no_grad()
    def fit_offset(ftm, pts_by_e):
        offs = {}
        for eid, name in enumerate(names):
            total, count = None, 0
            for ci, t in pts_by_e[eid]:
                e = clips_by_e[eid][ci]["e"].float()
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                d = (ftm(e_t, z) - e_t).mean(1)[0]
                total = d if total is None else total + d
                count += 1
            offs[name] = (total / max(count, 1)).cpu()
        return offs

    @torch.no_grad()
    def leak(ftm, offs, pts_by_e):
        pooled, eids = [], []
        for eid, name in enumerate(names):
            for ci, t in pts_by_e[eid]:
                e = clips_by_e[eid][ci]["e"].float()
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                d = (ftm(e_t, z) - e_t).mean(1)[0]
                pooled.append((d - offs[name].to(device)).cpu().numpy())
                eids.append(eid)
        pooled, eids = np.stack(pooled), np.array(eids)
        return cross_val_score(LogisticRegression(max_iter=500), pooled, eids, cv=5).mean()

    print(f"\n{args.ckpt}")
    print(f"stratified split: " + ", ".join(f"{names[e]} {len(train_idx[e])}tr/{len(test_idx[e])}te"
                                            for e in range(len(names))))

    # ============ 0 & 1: baseline, two split methodologies, base (un-fine-tuned) checkpoint =====
    off_base_strat = fit_offset(base_ftm, train_pts)
    leak_base_strat = leak(base_ftm, off_base_strat, test_pts)
    print(f"\n[1] BASE checkpoint, STRATIFIED split (this script's own methodology)")
    print(f"    leak with frozen offset: {leak_base_strat:.3f}")

    # reproduce offset_fix_check.py's own joint-shuffle split exactly, on the same base checkpoint
    Dpool_all, EID_all, cid_joint = [], [], []
    ctr = 0
    for eid, name in enumerate(names):
        with torch.no_grad():
            for ci, c in enumerate(clips_by_e[eid]):
                e = c["e"].float()
                for t in range(1, len(e) - 2, args.stride):
                    e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                    z = itm(e_t, e1)
                    d = (base_ftm(e_t, z) - e_t).mean(1)[0]
                    Dpool_all.append(d.cpu().numpy())
                    EID_all.append(eid)
                    cid_joint.append(ctr + ci)
        ctr += len(clips_by_e[eid])
    Dpool_all = np.stack(Dpool_all); EID_all = np.array(EID_all); cid_joint = np.array(cid_joint)
    rng2 = np.random.default_rng(args.seed)
    clips_joint = sorted(set(cid_joint.tolist()))
    rng2.shuffle(clips_joint)
    n_te_j = max(1, int(len(clips_joint) * args.test_frac))
    te_j = set(clips_joint[:n_te_j])
    te_mask = np.array([c in te_j for c in cid_joint]); tr_mask = ~te_mask
    print(f"\n[1b] BASE checkpoint, JOINT-SHUFFLE split (offset_fix_check.py's methodology)")
    comp = ", ".join(f"{names[e]}: {int(((EID_all==e)&te_mask).sum())}te/"
                     f"{int(((EID_all==e)&tr_mask).sum())}tr" for e in range(len(names)))
    print(f"    test-clip embodiment composition: {comp}")
    off_j = {names[e]: Dpool_all[tr_mask & (EID_all == e)].mean(0) for e in range(len(names))}
    Dcorr_j = Dpool_all - np.stack([off_j[names[e]] for e in EID_all])
    acc_j = cross_val_score(LogisticRegression(max_iter=500), Dcorr_j[te_mask], EID_all[te_mask],
                            cv=min(5, int(te_mask.sum() // 20) or 2)).mean()
    print(f"    leak with frozen offset: {acc_j:.3f}  (offset_fix_check.py reported 0.464)")

    # ============ train the L_recon-only FTM (same as the earlier control arm) =================
    print(f"\n[2] Training L_recon-only FTM, {args.steps} steps (reproduces the control arm)")
    ftm_c = copy.deepcopy(base_ftm)
    for p in ftm_c.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ftm_c.parameters(), lr=args.lr_ftm)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        eid = step % len(names)
        pool = train_pts[eid]
        pick = [pool[i] for i in torch.randint(len(pool), (args.batch,), generator=g).tolist()]
        e_t = torch.stack([clips_by_e[eid][ci]["e"][t] for ci, t in pick]).float().to(device)
        e1 = torch.stack([clips_by_e[eid][ci]["e"][t + 1] for ci, t in pick]).float().to(device)
        with torch.no_grad():
            z = itm(e_t, e1)
        pred = ftm_c(e_t, z)
        loss = F.mse_loss(pred, e1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ftm_c.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 4) == 0 or step == args.steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.5f}  "
                  f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
    ftm_c.eval()

    print(f"\n[2a] FINE-TUNED FTM, stale offset (fit on base checkpoint)")
    leak_stale = leak(ftm_c, off_base_strat, test_pts)
    print(f"    leak: {leak_stale:.3f}  (state_head_transfer_confirm.py reported ~0.95)")

    print(f"\n[2b] FINE-TUNED FTM, FRESH offset (re-fit on train clips of the moved model)")
    off_fresh = fit_offset(ftm_c, train_pts)
    leak_fresh = leak(ftm_c, off_fresh, test_pts)
    print(f"    leak: {leak_fresh:.3f}  (chance ~0.50-0.51)")

    print(f"\n[3] How far did the offset move?")
    for name in names:
        old, new = off_base_strat[name], off_fresh[name]
        cos = float(torch.nn.functional.cosine_similarity(old.unsqueeze(0), new.unsqueeze(0)))
        rel = float((new - old).norm() / old.norm().clamp_min(1e-9))
        print(f"    {name:>10}: cosine(old,new) {cos:.3f}   "
              f"||new-old||/||old|| {rel:.3f}   "
              f"|old| {old.norm().item():.3f}  |new| {new.norm().item():.3f}")

    print(f"\n  READ:")
    print(f"  fresh offset drops leak near chance -> periodic recompute is the fix, low risk.")
    print(f"  fresh offset stays high -> per-body-mean is the wrong model of the leak now;")
    print(f"  cosine near 1 with small ||new-old|| -> slow drift (recompute should track it);")
    print(f"  cosine far from 1 or large relative change -> restructuring, not drift.")


if __name__ == "__main__":
    main()
