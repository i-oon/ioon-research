"""Is the state head's weak R2 (0.059/0.099) under-training, or does it inherit the FTM's weak
predicted delta?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_ceiling.py

The identity-leak chase turned out to be cosmetic (`cross_embodiment_swap.py`: wrong-embodiment
offset changes R2 by nothing). That leaves the actual number exposed: R2 +0.059/+0.099, far under
the offline ridge ceiling (0.852) or the trained body head's (+0.430). Three non-exclusive causes,
tested together in one run:

    under-training     2000 steps is a short confirm; L_state may not have converged
    structural         the head/hyperparameters cap out regardless of input quality
    inherited weakness the head reads FTM(e_t,z) - e_t, and this whole session's finding is that
                       the FTM's predicted delta buries the action (+0.055 cosine, not the ~0.69
                       ceiling) -- if that weak signal is the input, no amount of head capacity
                       fixes it

**The discriminating test: oracle arm.** Same state head architecture, same steps, same optimiser --
but fed `e_t+1(true) - e_t`, the ground truth the FTM is trying to predict, instead of the FTM's own
output. This is not deployable (control time has no true next frame), but it isolates the question
cleanly: if the head learns the mapping well from a clean delta, the architecture is fine and the
predicted-delta arm's weakness is inherited from the FTM. If the oracle arm ALSO stays weak, the
head or its hyperparameters are the bottleneck, independent of the FTM.

**Loss logged as recon and state separately**, not combined, so under-training and gradient balance
are visible directly rather than inferred from a single number the way the earlier confirm run's
summary line obscured.

**Diagnosis only. The oracle arm's result bounds what a real rebuild could do; it does not train
anything meant to run at control time.**
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


@torch.no_grad()
def compute_offset_true(clips, train_ids, device, stride):
    """Same offset, computed from the TRUE delta -- the oracle arm's own input distribution."""
    total, count = None, 0
    for ci in train_ids:
        e = clips[ci]["e"].float()
        for t in range(1, len(e) - 2, stride):
            d = (e[t + 1] - e[t]).to(device).mean(0)
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
    train_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], train_idx[eid],
                                         channels, args.stride, names[eid])
                for eid in range(len(names))}
    test_pts = {eid: gather_transitions(clips_by_e[eid], paths_by_e[eid], test_idx[eid],
                                        channels, args.stride, names[eid])
               for eid in range(len(names))}

    print("\n  fitting offsets (predicted-delta and true-delta variants)", flush=True)
    off_pred, off_true = {}, {}
    for eid, name in enumerate(names):
        off_pred[name] = compute_offset(clips_by_e[eid], train_idx[eid], itm, base_ftm, device,
                                        args.stride)
        off_true[name] = compute_offset_true(clips_by_e[eid], train_idx[eid], device, args.stride)

    def run(oracle, tag):
        """oracle=False: predicted-delta arm (reproduces the earlier treatment run).
        oracle=True: state head fed the TRUE delta instead of FTM's prediction."""
        ftm = copy.deepcopy(base_ftm)
        for p in ftm.parameters():
            p.requires_grad_(not oracle)   # no reason to move the FTM when it isn't supplying the input
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

    print(f"\n  RESULT -- held-out Delta-state R2")
    print(f"  {'embodiment':>12}{'predicted-delta':>18}{'oracle (true delta)':>22}")
    for name in names:
        print(f"  {name:>12}{r2_p[name]:>+18.3f}{r2_o[name]:>+22.3f}")

    print("\n  READ: oracle >> predicted -> the head is fine, it inherits the FTM's weak prediction")
    print("  (this session's +0.055 problem, now shown to reach the state head too). Oracle ALSO")
    print("  weak -> the head/hyperparameters cap out regardless of input quality -- structural,")
    print("  not inherited. Either combined with 'still falling' above means more steps may still help.")


if __name__ == "__main__":
    main()
