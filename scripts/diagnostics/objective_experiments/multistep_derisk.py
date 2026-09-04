"""Does a 2-step consistency term reduce the FTM's action stamp? The cheap test before a rebuild.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/multistep_derisk.py \\
        --arm both --steps 800

**What this de-risks.** `where_action_lives.py` measured that on egocentric the FTM's output carries
action R2 **0.452** while the true `e_t+1` it is trained to match carries only **0.310** -- the model
writes `z` into its prediction instead of predicting the future. `L_pred` is single-step
(`wm/train.py:141`, one FTM call), and a stamp satisfies a single step perfectly. The proposed fix is
a multi-step term, on the argument that a stamp which cannot roll forward makes step 2 wrong.

**That argument is unproven, and proving it is the point of this file.** The full version needs a
loader change -- the training loader carries two timesteps (`VIEW_KEYS`) because cross-augmentation
re-encodes fresh views every epoch -- and then a pretrain. This runs the same question on cached
embeddings for a few minutes instead.

    arm 1step    loss = ||FTM(e_t, z1) - e_t+1||^2                        today's objective
    arm 2step    the same + ||FTM(FTM(e_t, z1), z2) - e_t+2||^2           the proposed term

**Three things are held fixed so the comparison means what it says.**

  the ITM is frozen        `z1`, `z2` are identical in both arms, so any change in the stamp is the
                           forward model's doing and not a relabelled latent
  same start, same order   both arms begin from the same checkpoint and draw the same batches under
                           the same seed, differing in exactly one term
  train and test disjoint  fitted on the body the teacher pretrained on, measured on the held-out
                           body, which is the split the 0.452 was measured under

**Pre-registered read**, written before the run:

  2step moves the stamp toward 0.310 and 1step does not   the term targets the measured cause;
                                                          the rebuild is justified
  both move it down by a similar amount                   it is fine-tuning on un-augmented cached
                                                          embeddings doing the work, not the term
  neither moves it                                        multi-step is not the fix; do not spend
                                                          the loader change and the pretrain

**Scope limits this cannot escape.** No cross-augmentation, since cached embeddings are fixed views
-- with the ITM frozen that removes the shortcut it exists to block, but it is not the pretraining
condition. The ITM is frozen, where a rebuild would train it. And this is a short fine-tune from a
converged model, so a null result bounds what the term does *to this checkpoint*, not what training
from scratch under it would produce. **Diagnosis only; writes no checkpoint into wm/runs.**
"""
import argparse
import collections
import copy
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
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def triples(clips, reach=2):
    """(clip, t) where e_t, e_t+1 and e_t+2 all exist."""
    out = []
    for ci, c in enumerate(clips):
        for t in range(1, len(c["e"]) - reach - 1):
            out.append((ci, t))
    return out


def load_clips(ckpt, data, embodiment, cache_path, lag, device):
    cache = torch.load(cache_path, map_location="cpu", mmap=True)
    before = len(cache)
    clips = gather(data, embodiment, None, ckpt, cache, 2, lag, device)
    if len(cache) > before:
        raise SystemExit(f"clips missing from {cache_path}; this file encodes nothing")
    return clips


@torch.no_grad()
def stamp_and_fidelity(ftm, itm, clips, device, stride=3):
    """Action R2 from the FTM's output and from the true next frame, plus rollout errors."""
    cols, A, clip_id, fam = collections.defaultdict(list), [], [], []
    one = two = hold = 0.0
    n = 0
    for ci, c in enumerate(clips):
        e = c["e"].float()
        for t in range(1, len(e) - 3, stride):
            e_t, e1, e2 = (e[t:t + 1].to(device), e[t + 1:t + 2].to(device),
                           e[t + 2:t + 3].to(device))
            z1 = itm(e_t, e1)
            z2 = itm(e1, e2)
            p1 = ftm(e_t, z1)
            p2 = ftm(p1, z2)
            one += F.mse_loss(p1, e1).item()
            two += F.mse_loss(p2, e2).item()
            hold += F.mse_loss(e_t, e1).item()
            n += 1
            cols["pred"].append(p1[0].flatten().half().cpu())
            cols["gt"].append(e[t + 1].flatten().half())
            A.append(c["a"][t].flatten().float())
            clip_id.append(ci); fam.append(FAMILY(c["cond"]))
    cols = {k: torch.stack(v) for k, v in cols.items()}
    A = torch.stack(A).numpy()
    clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id]); tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    y = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)

    out = {}
    for name, v in cols.items():
        g = gram(v, v, device).numpy()
        g = g / max(np.mean(np.diag(g)), 1e-12)
        out[name], _p, _a = ridge_r2(g[np.ix_(tr, tr)], g[np.ix_(te, tr)], y[tr], y[te], folds)
    out["mse_1step"] = one / max(n, 1)
    out["mse_2step"] = two / max(n, 1)
    out["hold_still"] = hold / max(n, 1)
    return out


def finetune(ftm, itm, clips, idx, arm, steps, batch, lr, seed, device):
    ftm.train()
    opt = torch.optim.AdamW(ftm.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        pick = [idx[i] for i in torch.randint(len(idx), (batch,), generator=g).tolist()]
        e_t = torch.stack([clips[c]["e"][t] for c, t in pick]).float().to(device)
        e1 = torch.stack([clips[c]["e"][t + 1] for c, t in pick]).float().to(device)
        e2 = torch.stack([clips[c]["e"][t + 2] for c, t in pick]).float().to(device)
        with torch.no_grad():
            z1, z2 = itm(e_t, e1), itm(e1, e2)
        p1 = ftm(e_t, z1)
        loss = F.mse_loss(p1, e1)
        if arm == "2step":
            loss = loss + F.mse_loss(ftm(p1, z2), e2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ftm.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"    step {step:5d}  loss {loss.item():.5f}  "
                  f"({(time.time() - t0) / max(step + 1, 1):.2f}s/step)", flush=True)
    ftm.eval()
    return ftm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat",
                    help="the body the teacher pretrained on")
    ap.add_argument("--test_data", default="data/egocentric/beh12_c08f09t09_ego_flat",
                    help="the held-out body the 0.452 was measured on")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--arm", choices=("1step", "2step", "both"), default="both")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="the rate stage 3 nudges a pretrained forward model at, not the "
                         "pretraining rate: 77M converged parameters are being moved, not learned")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    lag = max(1, cfg.action_lag)

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)

    train_clips = load_clips(ck, os.path.join(ROOT, args.train_data), args.embodiment,
                             os.path.join(ROOT, args.cache), lag, device)
    test_clips = load_clips(ck, os.path.join(ROOT, args.test_data), args.embodiment,
                            os.path.join(ROOT, args.cache), lag, device)
    idx = triples(train_clips)
    print(f"{args.ckpt}\ntrain {len(train_clips)} clips ({len(idx)} triples) from {args.train_data}"
          f"\ntest  {len(test_clips)} clips from {args.test_data}\n")

    base_ftm = ForwardTransitionModel(cfg).to(device).eval()
    base_ftm.load_state_dict(ck["ftm"])

    print("  BEFORE (the checkpoint as it stands)")
    before = stamp_and_fidelity(base_ftm, itm, test_clips, device, args.stride)
    print(f"    stamp: action R2 from FTM output {before['pred']:.3f}   "
          f"from true e_t+1 {before['gt']:.3f}")
    print(f"    mse 1-step {before['mse_1step']:.4f}  2-step {before['mse_2step']:.4f}  "
          f"hold-still {before['hold_still']:.4f}\n")

    arms = ("1step", "2step") if args.arm == "both" else (args.arm,)
    results = {"before": before}
    for arm in arms:
        print(f"  ARM {arm}")
        ftm = copy.deepcopy(base_ftm)
        for p in ftm.parameters():
            p.requires_grad_(True)
        finetune(ftm, itm, train_clips, idx, arm, args.steps, args.batch, args.lr, args.seed,
                 device)
        r = stamp_and_fidelity(ftm, itm, test_clips, device, args.stride)
        results[arm] = r
        print(f"    stamp: action R2 from FTM output {r['pred']:.3f}   "
              f"from true e_t+1 {r['gt']:.3f}")
        print(f"    mse 1-step {r['mse_1step']:.4f}  2-step {r['mse_2step']:.4f}\n")
        del ftm
        torch.cuda.empty_cache()

    print("  SUMMARY -- the stamp is `action R2 from the FTM output`; the target it should "
          "approach is\n  the true next frame's own value.")
    print(f"  {'arm':>10}{'stamp':>9}{'vs before':>11}{'gap to gt':>11}"
          f"{'mse 1step':>11}{'mse 2step':>11}")
    gt = before["gt"]
    for name in ("before",) + arms:
        r = results[name]
        print(f"  {name:>10}{r['pred']:>9.3f}{r['pred'] - before['pred']:>+11.3f}"
              f"{r['pred'] - gt:>+11.3f}{r['mse_1step']:>11.4f}{r['mse_2step']:>11.4f}")
    if len(arms) == 2:
        d1 = results["1step"]["pred"] - before["pred"]
        d2 = results["2step"]["pred"] - before["pred"]
        print(f"\n  2step moved the stamp {d2:+.3f}, 1step {d1:+.3f}; "
              f"the term's own effect is {d2 - d1:+.3f}")
        print("  **Read that difference, not the 2step column alone.** A drop both arms share is "
              "fine-tuning\n  on un-augmented cached embeddings, which is not the term under test.")


if __name__ == "__main__":
    main()
