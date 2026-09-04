"""Short fine-tune confirm: does the offset-corrected state head hold cross-body transfer?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_transfer_confirm.py

The offline proxy (`offset_fix_check.py`) is a closed-form ridge on a static snapshot: leak to
chance (0.464), Delta-state R2 unchanged (0.852=0.852), held-out clips, offset fit on train only.
**That is not the same claim as "training under `L_state` is safe."** F64's collapse was a trained
network finding a shortcut over many gradient steps; a one-shot linear probe on a frozen checkpoint
cannot rule that out. This trains the real `StateHead` and watches the number that matters: per
embodiment held-out Delta-state R2, before and after -- collapse looks like F64's -10.5/-57.2, not
like a modest change.

    control      FTM trained on L_recon alone (state head absent) -- matches every run before this
    treatment    + StateHead(pooled delta - frozen per-embodiment offset, z), L_state added

Both arms: same start weights, same seed, same batches, same steps -- differ in exactly the term
under test. **ITM is frozen** in both arms, so `z` is identical throughout and any change is the FTM
and the state head, not a relabelled latent -- same isolation `multistep_derisk.py` used.

The offset is fit ONCE, from training clips, using the checkpoint before any fine-tuning starts --
never recomputed from an evaluation batch (that would be the oracle leak `center_embeddings` and this
whole check exist to avoid). It stays fixed for the training run, per `state_head.py`'s documented
limitation.

**Pre-registered read.**

  transfer holds     both embodiments' held-out Delta-state R2 stay comparable to the offline
                     ridge ceiling (0.852) and to each other -- no embodiment craters. Leak,
                     re-measured on the trained head's own pooled-delta input, stays near chance.
                     -> the design is de-risked, com7 is justified.
  transfer collapses  one or both embodiments' R2 drops sharply, or the leak re-opens under
                     continued FTM drift away from where the offset was fit.
                     -> the offset fixed the static proxy, not the training dynamics; rethink
                     before spending com7.

**A short confirm, not the retrain.** Few thousand steps from a converged checkpoint; a null result
here is informative, a positive result bounds what the real run could do rather than replacing it.
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402


def load_all(ck, cfg, sources, caches, stride, channels, device):
    """One flat list of (embodiment_id, clip_id, t) pointers plus the tensors they index into."""
    clips_by_e, paths_by_e = {}, {}
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, caches[eid]), map_location="cpu", mmap=True)
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                       max(1, cfg.action_lag), device)
        clips_by_e[eid] = clips
        paths_by_e[eid] = paths
    return clips_by_e, paths_by_e


def split_clips(clips_by_e, seed, test_frac):
    """Per-embodiment held-out clip ids, disjoint from train, same recipe as offset_fix_check.py."""
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
def compute_offset(clips, paths, itm, ftm, train_ids, device, channels, stride):
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


def gather_transitions(clips, paths, ids, itm, device, channels, stride, name):
    """(e_t, e1, action_stride_target=body_motion) triples for the given clip ids."""
    out = []
    for ci in ids:
        c, p = clips[ci], paths[ci]
        bm = np.asarray(load(p, REGISTRY[name])["body_motion"])[:, channels]
        e = c["e"].float()
        for t in range(1, min(len(e) - 2, len(bm)), stride):
            out.append((ci, t, torch.tensor(bm[t], dtype=torch.float32)))
    return out


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
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lambda_state", type=float, default=1.5)
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

    clips_by_e, paths_by_e = load_all(ck, cfg, sources, args.caches, args.stride, channels, device)
    for eid, name in enumerate(names):
        print(f"  {name}: {len(clips_by_e[eid])} clips", flush=True)
    train_idx, test_idx = split_clips(clips_by_e, args.seed, args.test_frac)

    print("\n  fitting per-embodiment offset from TRAIN clips, base checkpoint", flush=True)
    offsets = {}
    for eid, name in enumerate(names):
        offsets[name] = compute_offset(clips_by_e[eid], paths_by_e[eid], itm, base_ftm,
                                       train_idx[eid], device, channels, args.stride)
        print(f"    {name}: norm {offsets[name].norm().item():.3f}", flush=True)

    train_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], train_idx[eid], itm,
                                         device, channels, args.stride, names[eid])
                for eid in range(len(names))}
    test_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], test_idx[eid], itm,
                                        device, channels, args.stride, names[eid])
               for eid in range(len(names))}
    for eid, name in enumerate(names):
        print(f"  {name}: {len(train_pts[eid])} train / {len(test_pts[eid])} test transitions",
              flush=True)

    def r2_of(ftm, state, offset_mode=True):
        """Held-out state-head R2 per embodiment, plus leak accuracy on its own pooled input."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
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
                    pooled_all.append((pooled - (offsets[name].to(device) if offset_mode
                                                 else 0)).cpu().numpy())
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
        """Same leak check as r2_of, without needing a state head -- for the control arm, which
        has none. Isolates whether the static offset drifts stale under L_recon alone."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
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
            eid = step % len(names)                    # alternate embodiments, balanced exposure
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
    # untrained head has random weights -- report leak on its INPUT (offset-corrected pooled
    # delta), which does not depend on the head's own untrained parameters
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
    print(f"\n  COMPARISON -- does L_recon alone explain the drift, or does L_state add to it?")
    print(f"    leak, control  (L_recon only)   : {leak_c:.3f}")
    print(f"    leak, treatment (+ L_state)     : {leak_t:.3f}")
    print(f"\n  RESULT -- treatment arm, held-out per embodiment")
    for name in names:
        print(f"    {name:>10}  Delta-state R2 {r2_t[name]:+.3f}")
    print(f"    leak on trained head's own pooled-delta input: {leak_t:.3f}  "
          f"(pre-fine-tune {leak_before:.3f}; un-corrected pooled delta was 0.961-0.977; "
          f"chance is the majority class fraction, ~0.50-0.51 with balanced embodiments)")
    print("\n  READ: F64's collapse was -10.5/-57.2 -- catastrophically negative, not merely lower.")
    print("  Both R2 well above 0 and comparable across embodiments, with leak still near chance,")
    print("  means transfer held under real gradient pressure. Either R2 deeply negative or leak")
    print("  reopening (rising back toward the un-corrected 0.977-0.961) means it did not.")


if __name__ == "__main__":
    main()
