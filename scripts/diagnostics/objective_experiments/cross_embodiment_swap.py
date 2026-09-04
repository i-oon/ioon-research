"""Does the state head's identity leak actually break cross-body prediction, or is that assumed?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/cross_embodiment_swap.py

Three identity-removal attempts have now failed (delta, frozen offset, recompute), each chased on
the premise that F64's -10.5/-57.2 collapse means high embodiment-leak is dangerous here too. That
premise was never measured on THIS head. F64's head read body motion from a shared trunk *fed the
raw frame*; this head reads a Froude-shared quantity from a *predicted change*, and the leak measured
so far (0.750, post-fine-tune) is on its INPUT, not shown to control its OUTPUT.

**The test: reproduce the treatment arm exactly (same seed), then swap embodiment identity at
inference.** The state head takes an explicit `embodiment` argument only to pick which offset buffer
to subtract. Scoring hexapod's own held-out transitions through the WRONG buffer (`b1`'s offset)
answers the load-bearing question directly: if Delta-state R2 barely moves, the offset -- and by
extension the identity information it removes -- is not what the shared trunk is using to predict
body motion, and the leak is cosmetic. If R2 collapses the way F64's did, the correction is
load-bearing and the leak is fatal after all.

    correct    hexapod transitions, hexapod's own offset   (already measured: R2 +0.057)
    swapped    hexapod transitions, B1's offset instead    -- the load-bearing test
    (mirrored for B1 against hexapod's offset)

**Reads:**

  R2 barely moves under the swap        the leak is cosmetic; stop chasing removal mechanisms and
                                        re-test whether the state head actually ranks (F179-style)
  R2 collapses toward F64's shape       the leak is fatal; additive removal has failed three ways,
                                        an adversarial term or an architectural rethink is next

com7 stays blocked regardless of this result -- it answers whether the removal chase is the right
chase, not whether the head is ready.
"""
import argparse
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


def split_clips(clips_by_e, seed, test_frac):
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
def compute_offset(clips, train_ids, itm, ftm, device, stride):
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


def gather_transitions(clips, paths, ids, channels, stride, name):
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

    clips_by_e, paths_by_e = {}, {}
    for eid, (name, path) in enumerate(sources):
        cache = torch.load(os.path.join(ROOT, args.caches[eid]), map_location="cpu", mmap=True)
        paths_by_e[eid] = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips_by_e[eid] = gather(os.path.join(ROOT, path), name, None, ck, cache, 2,
                                 max(1, cfg.action_lag), device)
        print(f"  {name}: {len(clips_by_e[eid])} clips", flush=True)

    train_idx, test_idx = split_clips(clips_by_e, args.seed, args.test_frac)
    print("\n  fitting per-embodiment offset from TRAIN clips, base checkpoint", flush=True)
    offsets = {}
    for eid, name in enumerate(names):
        offsets[name] = compute_offset(clips_by_e[eid], train_idx[eid], itm, base_ftm, device,
                                       args.stride)
        print(f"    {name}: norm {offsets[name].norm().item():.3f}", flush=True)

    train_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], train_idx[eid],
                                         channels, args.stride, names[eid])
                for eid in range(len(names))}
    test_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], test_idx[eid],
                                        channels, args.stride, names[eid])
               for eid in range(len(names))}
    for eid, name in enumerate(names):
        print(f"  {name}: {len(train_pts[eid])} train / {len(test_pts[eid])} test transitions",
              flush=True)

    print("\n  reproducing the treatment arm (same seed as state_head_transfer_confirm.py)",
          flush=True)
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(base_ftm.state_dict())
    for p in ftm.parameters():
        p.requires_grad_(True)
    state = StateHead(cfg, cfg.body_dim, names).to(device)
    for name in names:
        state.set_offset(name, offsets[name])
    opt = torch.optim.AdamW([{"params": ftm.parameters(), "lr": args.lr_ftm},
                             {"params": state.parameters(), "lr": args.lr_head}])
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
        sp = state(pred - e_t, z, names[eid])
        loss = F.mse_loss(pred, e1) + args.lambda_state * F.mse_loss(sp, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ftm.parameters()) + list(state.parameters()), 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.5f}  "
                  f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
    ftm.eval(); state.eval()

    @torch.no_grad()
    def r2_swapped(eid_data, eid_offset):
        """Score eid_data's held-out transitions using eid_offset's embodiment id (its offset)."""
        preds, truths = [], []
        for ci, t, y in test_pts[eid_data]:
            e = clips_by_e[eid_data][ci]["e"].float()
            e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
            z = itm(e_t, e1)
            pred = ftm(e_t, z)
            sp = state(pred - e_t, z, names[eid_offset])[0].cpu().numpy()
            preds.append(sp); truths.append(y.numpy())
        preds, truths = np.stack(preds), np.stack(truths)
        mu, sd = truths.mean(0), truths.std(0) + 1e-9
        ss = (((preds - (truths - mu) / sd)) ** 2).sum()
        ss_tot = (((truths - mu) / sd) ** 2).sum()
        return 1 - ss / max(ss_tot, 1e-9)

    print(f"\n  RESULT -- Delta-state R2, correct vs swapped embodiment id at inference")
    print(f"  {'data':>10}{'offset used':>14}{'R2':>9}")
    for eid, name in enumerate(names):
        other = names[1 - eid] if len(names) == 2 else names[(eid + 1) % len(names)]
        r_correct = r2_swapped(eid, eid)
        r_swapped = r2_swapped(eid, names.index(other))
        print(f"  {name:>10}{'own (correct)':>14}{r_correct:>+9.3f}")
        print(f"  {name:>10}{other + ' (WRONG)':>14}{r_swapped:>+9.3f}")

    print("\n  READ: if 'WRONG' R2 is close to 'correct' R2, the offset/identity correction is not")
    print("  load-bearing for the prediction -- the leak is cosmetic. If 'WRONG' collapses sharply")
    print("  (strongly negative, F64-shape), the correction matters and the leak is fatal.")


if __name__ == "__main__":
    main()
