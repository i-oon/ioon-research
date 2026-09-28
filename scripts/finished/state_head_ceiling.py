"""Is the state head's weak R2 (0.059/0.099) under-training, or does it inherit the FTM's weak
predicted delta -- and does that survive augmentation and a ridge-shaped, regularised head?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_ceiling.py --stage ceiling
    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_ceiling.py --stage aug
    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_ceiling.py --stage regularized

**Merged 2026-09-11 from three scripts that were sequential arms of one investigation**
(`state_head_ceiling.py`, `state_head_aug_confirm.py`, `state_head_regularized.py`) into one file
with a `--stage` flag. Each stage explicitly builds on and cites the prior stage's own numbers, so
they are kept in stage order rather than treated as independent measurements. Each stage's own
computation is preserved verbatim (only renamed/namespaced), not rewritten.

The identity-leak chase turned out to be cosmetic (`cross_embodiment_swap.py`: wrong-embodiment
offset changes R2 by nothing). That leaves the actual number exposed: R2 +0.059/+0.099, far under
the offline ridge ceiling (0.852) or the trained body head's (+0.430).

    --stage ceiling      (was state_head_ceiling.py) Three non-exclusive causes tested together:
                         under-training, structural head/hyperparameter cap, or inherited FTM
                         weakness. The discriminator is an ORACLE arm -- same head, fed the TRUE
                         `e_t+1 - e_t` instead of the FTM's own prediction. Found: training on
                         CACHED embeddings (fixed views of 34 clips/body), train state-loss
                         collapses to ~0.0001 while held-out R2 stays ~0.06-0.10 -- overfitting.
    --stage aug          (was state_head_aug_confirm.py) Does real cross-augmentation (fresh
                         random crop + jitter every step, via `wm/data/augment.py`, matching the
                         real training pipeline) fix the overfitting `ceiling` found? Slower by
                         design -- the encoder can no longer be cached.
    --stage regularized  (was state_head_regularized.py) Shrinks the head to `LinearStateHead` --
                         ONE linear layer, no hidden layer -- the closest a trained network gets to
                         the offline ridge's own function class, plus what ridge had and the earlier
                         arms did not: strong weight decay on the head, and early stopping on a
                         validation slice carved from TRAIN clips. Also fixes a real standardisation
                         bug found while building this arm (see its own section below) and adds
                         `--freeze_ftm`, the moving-target test: with the FTM frozen, pooled delta is
                         a static function of `(e_t, e_t+1)` alone, the same input the offline ridge
                         oracle saw.

**The read, common thread.** R2 climbing toward the ridge ceiling (0.852) at any stage means
Delta-state was alive and the prior stages' weak numbers were a fixable artifact (of confirm setup,
of clip count without augmentation, or of function-class mismatch, respectively) -- proceed to real
training. R2 staying ~0.06-0.15 despite each stage's fix means the wall is more fundamental than
that stage's hypothesis and the next stage exists to rule out the next cause.
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.augment import sample_params, apply, identity_params  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402


class LinearStateHead(nn.Module):
    """The closest a trained network gets to ridge's own function class: one linear layer over
    [pooled delta - offset, z], nothing else. (stage=regularized only)"""

    def __init__(self, cfg, state_dim, embodiments):
        super().__init__()
        self.linear = nn.Linear(cfg.token_dim + cfg.z_dim, state_dim)
        for name in embodiments:
            self.register_buffer(f"offset_{name}", torch.zeros(cfg.token_dim))

    def set_offset(self, embodiment, mean):
        with torch.no_grad():
            getattr(self, f"offset_{embodiment}").copy_(torch.as_tensor(mean, dtype=torch.float32))

    def forward(self, delta, z, embodiment="default"):
        d = delta.mean(1) if delta.dim() == 3 else delta
        d = d - getattr(self, f"offset_{embodiment}")
        return self.linear(torch.cat([d, z], dim=-1))


# ==================================================================================== stage: ceiling
def _ceiling_split_clips(clips_by_e, seed, test_frac):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = {}, {}
    for eid, clips in clips_by_e.items():
        order = list(range(len(clips)))
        rng.shuffle(order)
        n_te = max(1, int(len(order) * test_frac))
        test_idx[eid] = set(order[:n_te])
        train_idx[eid] = set(order[n_te:])
    return train_idx, test_idx


@torch.no_grad()
def _ceiling_compute_offset(clips, train_ids, itm, ftm, device, stride):
    total, count = None, 0
    for ci in train_ids:
        e = clips[ci]["e"].float()
        for t in range(1, len(e) - 2, stride):
            e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
            z = itm(e_t, e1)
            d = (ftm(e_t, z) - e_t).mean(1)[0]
            total = d if total is None else total + d
            count += 1
    return (total / max(count, 1)).cpu()


@torch.no_grad()
def _ceiling_compute_offset_true(clips, train_ids, device, stride):
    total, count = None, 0
    for ci in train_ids:
        e = clips[ci]["e"].float()
        for t in range(1, len(e) - 2, stride):
            d = (e[t + 1] - e[t]).to(device).mean(0)
            total = d if total is None else total + d
            count += 1
    return (total / max(count, 1)).cpu()


def _ceiling_gather_transitions(clips, paths, ids, channels, stride, name):
    out = []
    for ci in ids:
        c, p = clips[ci], paths[ci]
        bm = np.asarray(load(p, REGISTRY[name])["body_motion"])[:, channels]
        e = c["e"].float()
        for t in range(1, min(len(e) - 2, len(bm)), stride):
            out.append((ci, t, torch.tensor(bm[t], dtype=torch.float32)))
    return out


def stage_ceiling(args, device):
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

    clips_by_e, paths_by_e = {}, {}
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, args.caches[eid]), map_location="cpu", mmap=True)
        paths_by_e[eid] = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips_by_e[eid] = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                                 max(1, cfg.action_lag), device)
        print(f"  {name}: {len(clips_by_e[eid])} clips", flush=True)

    train_idx, test_idx = _ceiling_split_clips(clips_by_e, args.seed, args.test_frac)
    train_pts = {eid: _ceiling_gather_transitions(clips_by_e[eid], paths_by_e[eid], train_idx[eid],
                                                  channels, args.stride, names[eid])
                for eid in range(len(names))}
    test_pts = {eid: _ceiling_gather_transitions(clips_by_e[eid], paths_by_e[eid], test_idx[eid],
                                                 channels, args.stride, names[eid])
               for eid in range(len(names))}

    print("\n  fitting offsets (predicted-delta and true-delta variants)", flush=True)
    off_pred, off_true = {}, {}
    for eid, name in enumerate(names):
        off_pred[name] = _ceiling_compute_offset(clips_by_e[eid], train_idx[eid], itm, base_ftm,
                                                  device, args.stride)
        off_true[name] = _ceiling_compute_offset_true(clips_by_e[eid], train_idx[eid], device,
                                                       args.stride)

    def run(oracle, tag):
        ftm = copy.deepcopy(base_ftm)
        for p in ftm.parameters():
            p.requires_grad_(not oracle)
        state = StateHead(cfg, cfg.body_dim, names).to(device)
        offs = off_true if oracle else off_pred
        for name in names:
            state.set_offset(name, offs[name])
        groups = [{"params": state.parameters(), "lr": args.lr_head}]
        if not oracle:
            groups.append({"params": ftm.parameters(), "lr": args.lr_ftm})
        opt = torch.optim.AdamW(groups)
        g = torch.Generator().manual_seed(args.seed)
        t0 = time.time()
        recon_hist, state_hist = [], []
        for step in range(args.steps):
            eid = step % len(names)
            pool = train_pts[eid]
            pick = [pool[i] for i in torch.randint(len(pool), (args.batch,), generator=g).tolist()]
            e_t = torch.stack([clips_by_e[eid][ci]["e"][t] for ci, t, _ in pick]).float().to(device)
            e1 = torch.stack([clips_by_e[eid][ci]["e"][t + 1] for ci, t, _ in pick]).float().to(device)
            y = torch.stack([yy for _, _, yy in pick]).to(device)
            with torch.no_grad():
                z = itm(e_t, e1)
            if oracle:
                with torch.no_grad():
                    delta = e1 - e_t
                recon_loss = torch.tensor(0.0)
            else:
                pred = ftm(e_t, z)
                delta = pred - e_t
                recon_loss = F.mse_loss(pred, e1)
            sp = state(delta, z, names[eid])
            state_loss = F.mse_loss(sp, y)
            loss = recon_loss + args.lambda_state * state_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = list(state.parameters()) + ([] if oracle else list(ftm.parameters()))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            recon_hist.append(float(recon_loss.detach())); state_hist.append(float(state_loss.detach()))
            if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
                print(f"    [{tag}] step {step:5d}  recon {float(recon_loss):.5f}  "
                      f"state {float(state_loss):.5f}  "
                      f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
        ftm.eval(); state.eval()
        return ftm, state, recon_hist, state_hist

    @torch.no_grad()
    def r2_of(ftm, state, oracle):
        out = {}
        for eid, name in enumerate(names):
            preds, truths = [], []
            for ci, t, y in test_pts[eid]:
                e = clips_by_e[eid][ci]["e"].float()
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                delta = (e1 - e_t) if oracle else (ftm(e_t, z) - e_t)
                sp = state(delta, z, name)[0].cpu().numpy()
                preds.append(sp); truths.append(y.numpy())
            preds, truths = np.stack(preds), np.stack(truths)
            mu, sd = truths.mean(0), truths.std(0) + 1e-9
            ss = (((preds - (truths - mu) / sd)) ** 2).sum()
            ss_tot = (((truths - mu) / sd) ** 2).sum()
            out[name] = 1 - ss / max(ss_tot, 1e-9)
        return out

    print("\n  ARM predicted-delta (reproduces the earlier treatment run)")
    ftm_p, state_p, recon_p, state_hist_p = run(False, "predicted")
    r2_p = r2_of(ftm_p, state_p, oracle=False)

    print("\n  ARM oracle (state head fed the TRUE delta, not FTM's prediction)")
    _ftm_o, state_o, recon_o, state_hist_o = run(True, "oracle")
    r2_o = r2_of(None, state_o, oracle=True)

    def trend(hist, k=200):
        return np.mean(hist[:k]), np.mean(hist[-k:])

    print(f"\n  LOSS TRAJECTORY -- state-loss mean of first vs last {200} steps")
    for tag, hist in (("predicted", state_hist_p), ("oracle", state_hist_o)):
        a, b = trend(hist)
        print(f"    {tag:>10}: {a:.4f} -> {b:.4f}  ({'still falling' if b < a * 0.95 else 'flat/plateaued'})")

    print("\n  RESULT -- held-out Delta-state R2")
    print(f"  {'embodiment':>12}{'predicted-delta':>18}{'oracle (true delta)':>22}")
    for name in names:
        print(f"  {name:>12}{r2_p[name]:>+18.3f}{r2_o[name]:>+22.3f}")

    print("\n  READ: oracle >> predicted -> the head is fine, it inherits the FTM's weak prediction")
    print("  (this session's +0.055 problem, now shown to reach the state head too). Oracle ALSO")
    print("  weak -> the head/hyperparameters cap out regardless of input quality -- structural,")
    print("  not inherited. Either combined with 'still falling' above means more steps may still help.")


# ======================================================================================== stage: aug
def stage_aug(args, device):
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    sources = [tuple(s.split("=", 1)) for s in args.sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(ck["ftm"])
    for p in ftm.parameters():
        p.requires_grad_(True)

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    clips_by_e = {}
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = {}, {}
    for eid, (name, path) in enumerate(sources):
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = []
        for p in paths:
            d = load(p, REGISTRY[name])
            bm = np.asarray(d["body_motion"])[:, channels]
            clips.append({"frames": d["frames"], "bm": bm})
        clips_by_e[eid] = clips
        order = list(range(len(clips)))
        rng.shuffle(order)
        n_te = max(1, int(len(order) * args.test_frac))
        test_idx[eid] = order[:n_te]
        train_idx[eid] = order[n_te:]
        print(f"  {name}: {len(clips)} clips, {len(train_idx[eid])} train / {len(test_idx[eid])} test",
              flush=True)

    def train_pts(eid):
        out = []
        for ci in train_idx[eid]:
            n = min(len(clips_by_e[eid][ci]["frames"]) - 2, len(clips_by_e[eid][ci]["bm"]))
            out.extend((ci, t) for t in range(1, n, args.stride))
        return out

    def test_pts(eid):
        out = []
        for ci in test_idx[eid]:
            n = min(len(clips_by_e[eid][ci]["frames"]) - 2, len(clips_by_e[eid][ci]["bm"]))
            out.extend((ci, t) for t in range(1, n, args.stride))
        return out

    tr_pts = {eid: train_pts(eid) for eid in range(len(names))}
    te_pts = {eid: test_pts(eid) for eid in range(len(names))}

    def encode_pair(eid, ci, t, augmented, arng):
        frames = clips_by_e[eid][ci]["frames"]
        f_t, f_1 = frames[t], frames[t + 1]
        if augmented:
            par = sample_params(arng, *f_t.shape[:2])
        else:
            par = identity_params(*f_t.shape[:2])
        return apply(f_t, par), apply(f_1, par)

    @torch.no_grad()
    def compute_offset():
        offs = {}
        for eid, name in enumerate(names):
            total, count = None, 0
            for ci, t in tr_pts[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                e_t, e1 = e[0:1], e[1:2]
                z = itm(e_t, e1)
                d = (ftm(e_t, z) - e_t).mean(1)[0]
                total = d if total is None else total + d
                count += 1
            offs[name] = (total / max(count, 1)).cpu()
        return offs

    print("\n  fitting per-embodiment offset (un-augmented, train clips only)", flush=True)
    offsets = compute_offset()

    state = StateHead(cfg, cfg.body_dim, names).to(device)
    for name in names:
        state.set_offset(name, offsets[name])
    opt = torch.optim.AdamW([{"params": ftm.parameters(), "lr": args.lr_ftm},
                             {"params": state.parameters(), "lr": args.lr_head}])
    g = torch.Generator().manual_seed(args.seed)
    arng = np.random.default_rng(args.seed + 1)

    @torch.no_grad()
    def r2_held_out():
        out = {}
        for eid, name in enumerate(names):
            preds, truths = [], []
            for ci, t in te_pts[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                e_t, e1 = e[0:1], e[1:2]
                z = itm(e_t, e1)
                pred = ftm(e_t, z)
                sp = state(pred - e_t, z, name)[0].cpu().numpy()
                preds.append(sp)
                truths.append(clips_by_e[eid][ci]["bm"][t])
            preds, truths = np.stack(preds), np.stack(truths)
            mu, sd = truths.mean(0), truths.std(0) + 1e-9
            ss = (((preds - (truths - mu) / sd)) ** 2).sum()
            ss_tot = (((truths - mu) / sd) ** 2).sum()
            out[name] = 1 - ss / max(ss_tot, 1e-9)
        return out

    print(f"\n  training WITH cross-augmentation, {args.steps} steps", flush=True)
    t0 = time.time()
    state_hist = []
    for step in range(args.steps):
        eid = step % len(names)
        pool = tr_pts[eid]
        pick_idx = torch.randint(len(pool), (args.batch,), generator=g).tolist()
        pairs = [pool[i] for i in pick_idx]
        frames_batch = []
        for ci, t in pairs:
            a, b = encode_pair(eid, ci, t, True, arng)
            frames_batch.extend([a, b])
        e = encoder.encode(frames_batch).float()
        e = e.view(len(pairs), 2, *e.shape[1:])
        e_t, e1 = e[:, 0], e[:, 1]
        y = torch.tensor(np.stack([clips_by_e[eid][ci]["bm"][t] for ci, t in pairs]),
                         dtype=torch.float32, device=device)
        with torch.no_grad():
            z = itm(e_t, e1)
        pred = ftm(e_t, z)
        recon_loss = F.mse_loss(pred, e1)
        sp = state(pred - e_t, z, names[eid])
        state_loss = F.mse_loss(sp, y)
        loss = recon_loss + args.lambda_state * state_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ftm.parameters()) + list(state.parameters()), 1.0)
        opt.step()
        state_hist.append(float(state_loss.detach()))
        if step % args.eval_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t0
            print(f"    step {step:5d}  recon {float(recon_loss.detach()):.5f}  "
                  f"state {float(state_loss.detach()):.5f}"
                  f"  ({elapsed / max(step + 1, 1):.2f}s/step)", flush=True)

    def trend(hist, k=100):
        k = min(k, len(hist) // 2) or 1
        return np.mean(hist[:k]), np.mean(hist[-k:])
    a, b = trend(state_hist)
    print(f"\n  state-loss trend: {a:.4f} -> {b:.4f}  "
          f"({'collapsed (memorising anyway)' if b < a * 0.1 else 'did not collapse the same way'})")

    print("\n  final held-out R2 (un-augmented eval, same protocol as every prior confirm)")
    r2 = r2_held_out()
    for name in names:
        print(f"    {name:>10}  R2 {r2[name]:+.3f}")
    print("\n  for reference, WITHOUT augmentation (stage=ceiling): hexapod +0.057, b1 +0.095")
    print("\n  READ: R2 clearly above the unaugmented reference -> augmentation helps, more steps")
    print("  and the real pipeline's full augmentation may close the gap further -- com7 is worth")
    print("  trying. R2 still ~0.06-0.10 -> clip-count is the wall regardless of augmentation.")


# ================================================================================ stage: regularized
def stage_regularized(args, device):
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    sources = [tuple(s.split("=", 1)) for s in args.sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(ck["ftm"])
    if args.freeze_ftm:
        ftm.eval()
    for p in ftm.parameters():
        p.requires_grad_(not args.freeze_ftm)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    clips_by_e = {}
    rng = np.random.default_rng(args.seed)
    fit_idx, val_idx, test_idx = {}, {}, {}
    for eid, (name, path) in enumerate(sources):
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = []
        for p in paths:
            d = load(p, REGISTRY[name])
            clips.append({"frames": d["frames"], "bm": np.asarray(d["body_motion"])[:, channels]})
        clips_by_e[eid] = clips
        order = list(range(len(clips)))
        rng.shuffle(order)
        n_te = max(1, int(len(order) * args.test_frac))
        test_idx[eid] = order[:n_te]
        rest = order[n_te:]
        n_va = max(1, int(len(rest) * args.val_frac))
        val_idx[eid] = rest[:n_va]
        fit_idx[eid] = rest[n_va:]
        print(f"  {name}: {len(clips)} clips -> {len(fit_idx[eid])} fit / {len(val_idx[eid])} val "
              f"/ {len(test_idx[eid])} test", flush=True)

    def pts_of(idx_map, eid):
        out = []
        for ci in idx_map[eid]:
            n = min(len(clips_by_e[eid][ci]["frames"]) - 2, len(clips_by_e[eid][ci]["bm"]))
            out.extend((ci, t) for t in range(1, n, args.stride))
        return out

    fit_pts = {eid: pts_of(fit_idx, eid) for eid in range(len(names))}
    val_pts = {eid: pts_of(val_idx, eid) for eid in range(len(names))}
    test_pts = {eid: pts_of(test_idx, eid) for eid in range(len(names))}

    def encode_pair(eid, ci, t, augmented, arng):
        frames = clips_by_e[eid][ci]["frames"]
        f_t, f_1 = frames[t], frames[t + 1]
        par = sample_params(arng, *f_t.shape[:2]) if augmented else identity_params(*f_t.shape[:2])
        return apply(f_t, par), apply(f_1, par)

    @torch.no_grad()
    def compute_offset():
        offs = {}
        for eid, name in enumerate(names):
            total, count = None, 0
            for ci, t in fit_pts[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                d = (ftm(e[0:1], z) - e[0:1]).mean(1)[0]
                total = d if total is None else total + d
                count += 1
            offs[name] = (total / max(count, 1)).cpu()
        return offs

    print("\n  fitting per-embodiment offset (un-augmented, FIT clips only)", flush=True)
    offsets = compute_offset()

    # **The bug this run exists to fix.** Every earlier confirm trained on RAW body_motion (e.g.
    # magnitude ~0.02-0.15) but evaluated R2 by standardising the TRUTHS only, comparing them
    # against the model's raw-scale output -- a units mismatch that manufactures a large residual
    # regardless of model quality. Standardised here, matching ridge's own preprocessing, and used
    # consistently for both the training loss and the R2 evaluation below.
    all_bm = np.concatenate([clips_by_e[eid][ci]["bm"][t] .reshape(1, -1)
                             for eid in range(len(names)) for ci, t in fit_pts[eid]], axis=0)
    y_mu = all_bm.mean(0); y_sd = all_bm.std(0) + 1e-9
    print(f"  target standardisation (fit clips only): mu {y_mu}, sd {y_sd}", flush=True)

    state = LinearStateHead(cfg, cfg.body_dim, names).to(device)
    for name in names:
        state.set_offset(name, offsets[name])
    groups = [{"params": state.parameters(), "lr": args.lr_head, "weight_decay": args.wd_head}]
    if not args.freeze_ftm:
        groups.append({"params": ftm.parameters(), "lr": args.lr_ftm, "weight_decay": args.wd_ftm})
    opt = torch.optim.AdamW(groups)
    g = torch.Generator().manual_seed(args.seed)
    arng = np.random.default_rng(args.seed + 1)

    @torch.no_grad()
    def r2_on(pts_by_e):
        out = {}
        for eid, name in enumerate(names):
            preds, truths = [], []
            for ci, t in pts_by_e[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                pred = ftm(e[0:1], z)
                sp = state(pred - e[0:1], z, name)[0].cpu().numpy()
                preds.append(sp); truths.append(clips_by_e[eid][ci]["bm"][t])
            preds, truths = np.stack(preds), np.stack(truths)
            truths_std = (truths - y_mu) / y_sd
            ss = ((preds - truths_std) ** 2).sum()
            ss_tot = (truths_std ** 2).sum()
            out[name] = 1 - ss / max(ss_tot, 1e-9)
        return out

    @torch.no_grad()
    def val_loss():
        losses = []
        for eid, name in enumerate(names):
            for ci, t in val_pts[eid][:60]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                pred = ftm(e[0:1], z)
                sp = state(pred - e[0:1], z, name)
                y = torch.tensor((clips_by_e[eid][ci]["bm"][t] - y_mu) / y_sd, dtype=torch.float32,
                                 device=device).unsqueeze(0)
                losses.append(F.mse_loss(sp, y).item())
        return float(np.mean(losses))

    print(f"\n  training regularised LINEAR head, wd_head={args.wd_head}, {args.steps} steps",
          flush=True)
    t0 = time.time()
    best_val, best_state, best_ftm, best_step = float("inf"), None, None, -1
    for step in range(args.steps):
        eid = step % len(names)
        pool = fit_pts[eid]
        pick = [pool[i] for i in torch.randint(len(pool), (args.batch,), generator=g).tolist()]
        frames_batch = []
        for ci, t in pick:
            a, b = encode_pair(eid, ci, t, True, arng)
            frames_batch.extend([a, b])
        e = encoder.encode(frames_batch).float()
        e = e.view(len(pick), 2, *e.shape[1:])
        e_t, e1 = e[:, 0], e[:, 1]
        y = torch.tensor(np.stack([(clips_by_e[eid][ci]["bm"][t] - y_mu) / y_sd for ci, t in pick]),
                         dtype=torch.float32, device=device)
        with torch.no_grad():
            z = itm(e_t, e1)
        pred = ftm(e_t, z)
        recon_loss = F.mse_loss(pred, e1)
        sp = state(pred - e_t, z, names[eid])
        state_loss = F.mse_loss(sp, y)
        loss = recon_loss + args.lambda_state * state_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ftm.parameters()) + list(state.parameters()), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = val_loss()
            tag = ""
            if vl < best_val:
                best_val, best_step = vl, step
                best_state = {k: v.clone() for k, v in state.state_dict().items()}
                best_ftm = {k: v.clone() for k, v in ftm.state_dict().items()}
                tag = "  <- best so far"
            print(f"    step {step:5d}  recon {float(recon_loss.detach()):.5f}  "
                  f"state(train) {float(state_loss.detach()):.5f}  state(val) {vl:.5f}"
                  f"  ({(time.time() - t0) / max(step + 1, 1):.2f}s/step){tag}", flush=True)

    print(f"\n  best validation snapshot: step {best_step}, val state-loss {best_val:.5f}")
    state.load_state_dict(best_state)
    ftm.load_state_dict(best_ftm)

    print("\n  held-out TEST R2, best-validation snapshot (never touched by early stopping)")
    r2 = r2_on(test_pts)
    for name in names:
        print(f"    {name:>10}  R2 {r2[name]:+.3f}")
    print("\n  for reference: unregularised MLP (no aug) hexapod +0.057 b1 +0.095")
    print("                 unregularised MLP (aug)    hexapod +0.063 b1 +0.150")
    print("                 offline ridge oracle        0.852 (different validation split)")
    print("\n  READ: climbing toward 0.852 -> Delta-state alive, regularised design for com7.")
    print("  staying ~0.06-0.15 -> ridge's number does not reproduce under a trained, early-")
    print("  stopped model here -- Delta-state is capped on this data regardless of head shape.")


# =========================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ceiling", "aug", "regularized"], required=True)
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--caches", nargs="+",
                    default=["results/wm/cache/ego_hex.pt", "results/wm/cache/ego_b1.pt"],
                    help="stage=ceiling only")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--val_frac", type=float, default=0.2, help="stage=regularized only, carved "
                    "from TRAIN clips for early stopping, never touches test")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000, help="default 2000 for ceiling, 800 for "
                    "aug/regularized (their original, different, defaults) unless overridden")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_ftm", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd_head", type=float, default=0.1, help="stage=regularized only")
    ap.add_argument("--wd_ftm", type=float, default=0.0, help="stage=regularized only")
    ap.add_argument("--lambda_state", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100, help="stage=aug/regularized only")
    ap.add_argument("--freeze_ftm", action="store_true", help="stage=regularized only. **The "
                    "moving-target test.** With FTM frozen, pooled delta is a static function of "
                    "(e_t, e_t+1) alone -- the same input the offline ridge oracle saw. If R2 closes "
                    "toward 0.852 here but not with FTM trainable, the earlier ceiling was FTM "
                    "drift, not head capacity or Delta-state itself.")
    args = ap.parse_args()
    if args.steps == 2000 and args.stage in ("aug", "regularized"):
        args.steps = 800   # their original default; ceiling's 2000 default left untouched otherwise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"ceiling": stage_ceiling, "aug": stage_aug, "regularized": stage_regularized}[args.stage](
        args, device)


if __name__ == "__main__":
    main()
