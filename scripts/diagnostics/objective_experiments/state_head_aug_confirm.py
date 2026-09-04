"""Does cross-augmentation prevent the overfitting `state_head_ceiling.py` found?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_aug_confirm.py

`state_head_ceiling.py` trained on CACHED embeddings -- fixed views of 34 clips per body -- and
found train state-loss collapse to ~0.0001 while held-out R2 stayed at +0.057/+0.095: overfitting,
not under-training. But cross-augmentation exists in the real pipeline precisely to prevent this
(fresh random crop + brightness/contrast jitter every step), and none of that session's confirms
used it. This is the discriminator: run the identical setup with real augmentation and see which of
two very different situations holds.

    fixed         R2 climbs toward the ridge ceiling (0.852)          -> augmentation-fixable,
                                                                         the earlier result was a
                                                                         confirm-setup artifact,
                                                                         com7 (which uses
                                                                         cross_augment) is fine
    clip-count    R2 stays ~0.06-0.10 despite augmentation            -> the wall is 48 clips a
                                                                         body, not the lack of
                                                                         augmented views of them;
                                                                         no amount of com7 time
                                                                         fixes it

**Frames are loaded raw from the npz and encoded fresh every step**, a random crop/jitter sampled
per step via `wm/data/augment.py` -- the same augmentation the real training loop uses, just without
the two-view ITM/FTM split (irrelevant to this question; one view per pair is enough to test whether
view diversity closes the generalisation gap). Held-out evaluation stays un-augmented (identity
params), matching how validation works in the real pipeline.

**Slower than the cached-embedding confirms by design** -- the encoder is the dominant cost once
frames cannot be cached, exactly as `direction_plan.md` documents. Step count is reduced accordingly;
the train-loss-collapse signature was already visible by step 400 in the unaugmented run, so it does
not need 2000 steps to show up again if it is still happening.
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

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.augment import sample_params, apply, identity_params  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_ftm", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lambda_state", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=200)
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
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(ck["ftm"])
    for p in ftm.parameters():
        p.requires_grad_(True)

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    # ---- load raw clips (frames + body_motion), split by clip, same recipe as every prior confirm
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
    print(f"\n  for reference, WITHOUT augmentation (state_head_ceiling.py): hexapod +0.057, b1 +0.095")
    print("\n  READ: R2 clearly above the unaugmented reference -> augmentation helps, more steps")
    print("  and the real pipeline's full augmentation may close the gap further -- com7 is worth")
    print("  trying. R2 still ~0.06-0.10 -> clip-count is the wall regardless of augmentation.")


if __name__ == "__main__":
    main()
