"""Does a learned per-embodiment offset remove the pooled-delta leak without killing Delta-state?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/offset_fix_check.py --stage check
    .venv/bin/python3 scripts/diagnostics/objective_experiments/offset_fix_check.py --stage derisk
    .venv/bin/python3 scripts/diagnostics/objective_experiments/offset_fix_check.py --stage confirm

**Merged 2026-09-11 from three scripts that were one sequential investigation** (`offset_fix_check.py`,
`offset_derisk.py`, `state_head_transfer_confirm.py`) into one file with a `--stage` flag. Unlike a
one-shot-diagnosis cluster, these three build directly on each other's own reported numbers (stage 2
explicitly reconciles a discrepancy against stage 1's number; stage 3 uses stage 1/2's offset
methodology inside a real training run) -- so they are kept in stage ORDER, not side by side, and
each stage's own computation is preserved verbatim (only renamed/namespaced), not rewritten.

`frame_vs_delta_classify.py` found the state head's actual input -- `delta.mean(1)`, the FTM's
predicted change pooled over patch tokens -- reads embodiment identity at 0.977, and that an ORACLE
per-embodiment mean removal (using the true label on the whole set) drops it to 0.114, pointing at
F35's mechanism: an additive per-embodiment offset, not something distributed and adversarial-shaped.

    --stage check    (was offset_fix_check.py) The non-oracle version: closed-form offset fit on
                     TRAIN clips only, leak (embodiment accuracy) and signal (Delta-state ridge R2)
                     both read on TEST clips the offset never saw. Diagnosis only, no gradients.
    --stage derisk   (was offset_derisk.py) Does the frozen offset survive an FTM that keeps
                     training under plain L_recon (no state head)? Reconciles this script's own
                     stratified-split baseline against `check`'s joint-shuffle baseline (they use
                     different splits, reported explicitly rather than assumed equal), trains an
                     L_recon-only FTM, then compares a STALE offset (fit on the base checkpoint)
                     against a FRESH offset (re-fit on the moved model) -- the gate everything after
                     this depends on.
    --stage confirm  (was state_head_transfer_confirm.py) The real thing: trains the actual
                     `StateHead` with `L_state` added, control (no state head) vs treatment (+
                     state head), and watches per-embodiment held-out Delta-state R2 for an F57-style
                     collapse (-10.5/-57.2), not just a modest change. The offset is fit ONCE before
                     fine-tuning starts and never recomputed from an eval batch (that would be the
                     oracle leak this whole chain exists to avoid).

**The read, per stage.** `check`: leak toward chance (0.501) with signal (R2) preserved means the fix
is targeted and cheap. `derisk`: a fresh offset dropping leak back toward chance on the fine-tuned
model means periodic recompute is a viable fix; if it doesn't, per-body-mean is the wrong model of
the leak. `confirm`: both embodiments' R2 staying comparable to the offline ridge ceiling (0.852)
with leak still near chance means the design is de-risked and justifies spending real training
compute; a sharp, F57-shaped collapse means the offset fixed the static proxy, not the training
dynamics.
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
from wm.models.state_head import StateHead  # noqa: E402


# ===================================================================================== stage: check
def stage_check(args, device):
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


# ==================================================================================== stage: derisk
def stage_derisk(args, device):
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    sources = [tuple(s.split("=", 1)) for s in args.multi_sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    base_ftm = ForwardTransitionModel(cfg).to(device).eval()
    base_ftm.load_state_dict(ck["ftm"])

    clips_by_e, paths_by_e = {}, {}
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, args.caches[eid]), map_location="cpu", mmap=True)
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        clips_by_e[eid], paths_by_e[eid] = clips, paths
        print(f"  {name}: {len(clips)} clips", flush=True)

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
        clips = clips_by_e[eid]
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
    print("stratified split: " + ", ".join(f"{names[e]} {len(train_idx[e])}tr/{len(test_idx[e])}te"
                                           for e in range(len(names))))

    off_base_strat = fit_offset(base_ftm, train_pts)
    leak_base_strat = leak(base_ftm, off_base_strat, test_pts)
    print("\n[1] BASE checkpoint, STRATIFIED split (this script's own methodology)")
    print(f"    leak with frozen offset: {leak_base_strat:.3f}")

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
    print("\n[1b] BASE checkpoint, JOINT-SHUFFLE split (offset_fix_check.py's methodology)")
    comp = ", ".join(f"{names[e]}: {int(((EID_all==e)&te_mask).sum())}te/"
                     f"{int(((EID_all==e)&tr_mask).sum())}tr" for e in range(len(names)))
    print(f"    test-clip embodiment composition: {comp}")
    off_j = {names[e]: Dpool_all[tr_mask & (EID_all == e)].mean(0) for e in range(len(names))}
    Dcorr_j = Dpool_all - np.stack([off_j[names[e]] for e in EID_all])
    acc_j = cross_val_score(LogisticRegression(max_iter=500), Dcorr_j[te_mask], EID_all[te_mask],
                            cv=min(5, int(te_mask.sum() // 20) or 2)).mean()
    print(f"    leak with frozen offset: {acc_j:.3f}  (offset_fix_check.py reported 0.464)")

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

    print("\n[2a] FINE-TUNED FTM, stale offset (fit on base checkpoint)")
    leak_stale = leak(ftm_c, off_base_strat, test_pts)
    print(f"    leak: {leak_stale:.3f}  (state_head_transfer_confirm.py reported ~0.95)")

    print("\n[2b] FINE-TUNED FTM, FRESH offset (re-fit on train clips of the moved model)")
    off_fresh = fit_offset(ftm_c, train_pts)
    leak_fresh = leak(ftm_c, off_fresh, test_pts)
    print(f"    leak: {leak_fresh:.3f}  (chance ~0.50-0.51)")

    print("\n[3] How far did the offset move?")
    for name in names:
        old, new = off_base_strat[name], off_fresh[name]
        cos = float(F.cosine_similarity(old.unsqueeze(0), new.unsqueeze(0)))
        rel = float((new - old).norm() / old.norm().clamp_min(1e-9))
        print(f"    {name:>10}: cosine(old,new) {cos:.3f}   "
              f"||new-old||/||old|| {rel:.3f}   "
              f"|old| {old.norm().item():.3f}  |new| {new.norm().item():.3f}")

    print("\n  READ:")
    print("  fresh offset drops leak near chance -> periodic recompute is the fix, low risk.")
    print("  fresh offset stays high -> per-body-mean is the wrong model of the leak now;")
    print("  cosine near 1 with small ||new-old|| -> slow drift (recompute should track it);")
    print("  cosine far from 1 or large relative change -> restructuring, not drift.")


# =================================================================================== stage: confirm
def stage_confirm(args, device):
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    sources = [tuple(s.split("=", 1)) for s in args.multi_sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    base_ftm = ForwardTransitionModel(cfg).to(device).eval()
    base_ftm.load_state_dict(ck["ftm"])

    clips_by_e, paths_by_e = {}, {}
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, args.caches[eid]), map_location="cpu", mmap=True)
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        clips_by_e[eid], paths_by_e[eid] = clips, paths
        print(f"  {name}: {len(clips_by_e[eid])} clips", flush=True)

    def split_clips(seed, test_frac):
        rng = np.random.default_rng(seed)
        train_idx, test_idx = {}, {}
        for eid, clips in clips_by_e.items():
            order = list(range(len(clips)))
            rng.shuffle(order)
            n_te = max(1, int(len(order) * test_frac))
            test_idx[eid] = set(order[:n_te])
            train_idx[eid] = set(order[n_te:])
        return train_idx, test_idx

    train_idx, test_idx = split_clips(args.seed, args.test_frac)

    @torch.no_grad()
    def compute_offset(clips, ftm, train_ids):
        total, count = None, 0
        for ci in train_ids:
            e = clips[ci]["e"].float()
            for t in range(1, len(e) - 2, args.stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                d = (ftm(e_t, z) - e_t).mean(1)[0]
                total = d if total is None else total + d
                count += 1
        return (total / max(count, 1)).cpu()

    def gather_transitions(clips, paths, ids, name):
        out = []
        for ci in ids:
            c, p = clips[ci], paths[ci]
            bm = np.asarray(load(p, REGISTRY[name])["body_motion"])[:, channels]
            e = c["e"].float()
            for t in range(1, min(len(e) - 2, len(bm)), args.stride):
                out.append((ci, t, torch.tensor(bm[t], dtype=torch.float32)))
        return out

    print("\n  fitting per-embodiment offset from TRAIN clips, base checkpoint", flush=True)
    offsets = {}
    for eid, name in enumerate(names):
        offsets[name] = compute_offset(clips_by_e[eid], base_ftm, train_idx[eid])
        print(f"    {name}: norm {offsets[name].norm().item():.3f}", flush=True)

    train_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], train_idx[eid], names[eid])
                for eid in range(len(names))}
    test_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], test_idx[eid], names[eid])
               for eid in range(len(names))}
    for eid, name in enumerate(names):
        print(f"  {name}: {len(train_pts[eid])} train / {len(test_pts[eid])} test transitions",
              flush=True)

    def r2_of(ftm, state):
        out = {}
        pooled_all, eid_all = [], []
        for eid, name in enumerate(names):
            preds, truths = [], []
            with torch.no_grad():
                for ci, t, y in test_pts[eid]:
                    e = clips_by_e[eid][ci]["e"].float()
                    e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                    z = itm(e_t, e1)
                    pred = ftm(e_t, z)
                    delta = pred - e_t
                    pooled = delta.mean(1)[0]
                    pooled_all.append((pooled - offsets[name].to(device)).cpu().numpy())
                    eid_all.append(eid)
                    sp = state(delta, z, name)[0].cpu().numpy()
                    preds.append(sp); truths.append(y.numpy())
            preds, truths = np.stack(preds), np.stack(truths)
            mu, sd = truths.mean(0), truths.std(0) + 1e-9
            ss = (((preds - (truths - mu) / sd)) ** 2).sum()
            ss_tot = (((truths - mu) / sd - 0) ** 2).sum()
            out[name] = 1 - ss / max(ss_tot, 1e-9)
        pooled_all, eid_all = np.stack(pooled_all), np.array(eid_all)
        leak = cross_val_score(LogisticRegression(max_iter=500), pooled_all, eid_all, cv=5).mean()
        return out, leak

    def leak_only(ftm):
        pooled_all, eid_all = [], []
        with torch.no_grad():
            for eid, name in enumerate(names):
                for ci, t, y in test_pts[eid]:
                    e = clips_by_e[eid][ci]["e"].float()
                    e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                    z = itm(e_t, e1)
                    pooled = (ftm(e_t, z) - e_t).mean(1)[0]
                    pooled_all.append((pooled - offsets[name].to(device)).cpu().numpy())
                    eid_all.append(eid)
        pooled_all, eid_all = np.stack(pooled_all), np.array(eid_all)
        return cross_val_score(LogisticRegression(max_iter=500), pooled_all, eid_all, cv=5).mean()

    def run(use_state, tag):
        ftm = copy.deepcopy(base_ftm)
        for p in ftm.parameters():
            p.requires_grad_(True)
        state = StateHead(cfg, cfg.body_dim, names).to(device)
        for name in names:
            state.set_offset(name, offsets[name])
        params = list(ftm.parameters())
        groups = [{"params": params, "lr": args.lr_ftm}]
        if use_state:
            groups.append({"params": state.parameters(), "lr": args.lr_head})
        opt = torch.optim.AdamW(groups)
        g = torch.Generator().manual_seed(args.seed)
        t0 = time.time()
        for step in range(args.steps):
            eid = step % len(names)
            pool = train_pts[eid]
            pick = [pool[i] for i in torch.randint(len(pool), (args.batch,), generator=g).tolist()]
            e_t = torch.stack([clips_by_e[eid][ci]["e"][t] for ci, t, _ in pick]).float().to(device)
            e1 = torch.stack([clips_by_e[eid][ci]["e"][t + 1] for ci, t, _ in pick]).float().to(device)
            y = torch.stack([yy for _, _, yy in pick]).to(device)
            with torch.no_grad():
                z = itm(e_t, e1)
            pred = ftm(e_t, z)
            loss = F.mse_loss(pred, e1)
            if use_state:
                sp = state(pred - e_t, z, names[eid])
                sloss = F.mse_loss(sp, y)
                loss = loss + args.lambda_state * sloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
                print(f"    [{tag}] step {step:5d}  loss {loss.item():.5f}  "
                      f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
        return ftm, state

    print("\n  BEFORE any fine-tune (base checkpoint's own state head, offset just fit)")
    base_state = StateHead(cfg, cfg.body_dim, names).to(device)
    for name in names:
        base_state.set_offset(name, offsets[name])
    _r2_before, leak_before = r2_of(base_ftm, base_state)
    print(f"    leak on offset-corrected pooled delta (pre-fine-tune FTM): {leak_before:.3f}")

    print("\n  ARM control (no state head, L_recon only)")
    ftm_c, _ = run(False, "control")
    leak_c = leak_only(ftm_c)
    print(f"    leak with the STATIC offset applied to this L_recon-only FTM: {leak_c:.3f}  "
          f"-- isolates whether L_recon alone already staled the offset")

    print("\n  ARM treatment (+ StateHead, offset-corrected, L_state)")
    ftm_t, state_t = run(True, "treatment")

    r2_t, leak_t = r2_of(ftm_t, state_t)
    print("\n  COMPARISON -- does L_recon alone explain the drift, or does L_state add to it?")
    print(f"    leak, control  (L_recon only)   : {leak_c:.3f}")
    print(f"    leak, treatment (+ L_state)     : {leak_t:.3f}")
    print("\n  RESULT -- treatment arm, held-out per embodiment")
    for name in names:
        print(f"    {name:>10}  Delta-state R2 {r2_t[name]:+.3f}")
    print(f"    leak on trained head's own pooled-delta input: {leak_t:.3f}  "
          f"(pre-fine-tune {leak_before:.3f}; un-corrected pooled delta was 0.961-0.977; "
          f"chance is the majority class fraction, ~0.50-0.51 with balanced embodiments)")
    print("\n  READ: F57's collapse was -10.5/-57.2 -- catastrophically negative, not merely lower.")
    print("  Both R2 well above 0 and comparable across embodiments, with leak still near chance,")
    print("  means transfer held under real gradient pressure. Either R2 deeply negative or leak")
    print("  reopening (rising back toward the un-corrected 0.977-0.961) means it did not.")


# =========================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["check", "derisk", "confirm"], required=True)
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"],
                    help="stage=check only: two-source form, cache picked by name (hexapod/other)")
    ap.add_argument("--multi_sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"],
                    help="stage=derisk/confirm: sources paired positionally with --caches")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt", help="stage=check only")
    ap.add_argument("--extra_cache", default="results/wm/cache/ego_b1.pt", help="stage=check only")
    ap.add_argument("--caches", nargs="+",
                    default=["results/wm/cache/ego_hex.pt", "results/wm/cache/ego_b1.pt"],
                    help="stage=derisk/confirm: one cache path per --multi_sources entry, in order")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=2000, help="stage=derisk/confirm")
    ap.add_argument("--batch", type=int, default=8, help="stage=derisk/confirm")
    ap.add_argument("--lr_ftm", type=float, default=1e-5, help="stage=derisk/confirm")
    ap.add_argument("--lr_head", type=float, default=1e-3, help="stage=confirm only")
    ap.add_argument("--lambda_state", type=float, default=1.5, help="stage=confirm only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"check": stage_check, "derisk": stage_derisk, "confirm": stage_confirm}[args.stage](args, device)


if __name__ == "__main__":
    main()
