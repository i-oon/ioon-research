"""Which loss term is actually pulling on the latent, measured as a gradient rather than a loss.

**Loss magnitude is not gradient influence, and the two are easy to conflate.** At convergence the
objective reads `recon` 1.6963, `motion` 0.0582, `body` 0.0362, so with the shipped weights the
total is 95.7% recon, 3.3% motion, 1.0% body. That invites the conclusion that reconstruction owns
the gradient -- but what updates the inverse model is `dL/dz`, and its size depends on the scale of
each term's output space rather than on the value of the loss. `recon` lives on unnormalised V-JEPA2
embeddings while `motion` and `body` live on standardised targets, so their losses are not on a
common scale to begin with.

This measures the thing directly: build one real batch, take each term's gradient with respect to
the **same** `z`, and report the norms.

**Why it decides what to fix**, and the two cases need opposite interventions:

    recon dominates dL/dz     the weighting is wrong -> lower lambda_recon, or normalise recon
    the terms are balanced    the weighting is fine and the *task* is too easy -> act on 2i and
                              predict further ahead, which F54 already measured to be better

**Context for reading it.** The forward model beats the do-nothing baseline by only 27.5%: copying
`e_t` gives an MSE of 2.338 (the variance of a one-step change) against the trained model's 1.696.
So most of the reconstruction problem is solved without reading `z` at all (F87), and a large
`recon` gradient would mostly be pulling `z` toward a solution it is not needed for.

  .venv/bin/python3 scripts/diagnostics/loss_gradient_balance.py --ckpt wm/runs/beh12_body_fwd/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load as load_clip  # noqa: E402
from wm.evaluate import encode_clip, offset_for, upgrade_decoder_state  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", default="data/beh12_c10f10t10_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64,
                    help="transitions to estimate the gradient on. Training uses batch_size 8, so "
                         "this is already generous -- and the whole set will not fit: retaining "
                         "the graph for three backward passes over 520 transitions of 256x1408 "
                         "tokens needs about 15 GB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    stats = ck.get("action_stats", {})
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).to(device).eval()
    md.load_state_dict(upgrade_decoder_state(ck["md"]))

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    enc = None
    channels = tuple(int(c) for c in cfg.body_channels)
    mean, std = stats[args.embodiment]
    mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(std, dtype=torch.float32, device=device)

    E, N, A, Bm = [], [], [], []
    for p in sorted(glob.glob(os.path.join(ROOT, args.dir, "*.npz")))[:args.clips]:
        if p not in cache:
            if enc is None:
                enc = VJEPA2FrameEncoder(dtype=torch.float32)
            with np.load(p, allow_pickle=True) as c:
                cache[p] = encode_clip(enc, c["frames"], 2).cpu().half()
        e = cache[p].float().to(device)
        off = offset_for(ck, args.embodiment)
        if off is not None:
            e = e - off.to(device)
        clip = load_clip(p, REGISTRY[args.embodiment])
        n = len(e) - 1
        lag = max(1, cfg.action_lag)
        if len(clip["actions"]) < n + lag:
            continue
        E.append(e[:n]); N.append(e[1:n + 1])
        A.append(torch.as_tensor(clip["actions"][lag:lag + n], dtype=torch.float32, device=device))
        Bm.append(torch.as_tensor(clip["body_motion"][:n][:, channels],
                                  dtype=torch.float32, device=device))
    e_t, e_next = torch.cat(E), torch.cat(N)
    a = (torch.cat(A) - mean) / std
    b = torch.cat(Bm)
    # **The checkpoint's own statistics, never this batch's.** `train.py` stores them because they
    # are pooled across *both* embodiments and so cannot be recomputed from one robot's clips.
    # Standardising here on a hexapod-only batch put the body loss at 0.7837 against the training
    # log's 0.0362 -- 21x too large -- and inflated its share of the gradient with it.
    if "body_stats" not in ck:
        raise SystemExit("checkpoint has no body_stats; it is not a cross-embodiment run")
    bm, bs = ck["body_stats"]
    b = (b - torch.as_tensor(bm, dtype=torch.float32, device=device)) / \
        torch.as_tensor(bs, dtype=torch.float32, device=device)

    if len(e_t) > args.batch:
        idx = torch.as_tensor(
            np.random.default_rng(args.seed).choice(len(e_t), args.batch, replace=False),
            device=device)
        e_t, e_next, a, b = e_t[idx], e_next[idx], a[idx], b[idx]

    z = itm(e_t, e_next)
    z.retain_grad()
    terms = {
        "recon": (cfg.lambda_recon, F.mse_loss(ftm(e_t, z, None), e_next)),
        "motion": (cfg.lambda_motion, F.mse_loss(md(e_t, z, args.embodiment), a)),
    }
    if md.body_head is not None and cfg.lambda_body > 0:
        terms["body"] = (cfg.lambda_body, F.mse_loss(md.body(e_t, z), b))

    print(f"{args.ckpt}   {len(e_t)} transitions, {args.embodiment}\n")
    print(f"{'term':<9}{'lambda':>8}{'loss':>10}{'share of loss':>15}"
          f"{'|dL/dz| raw':>14}{'x lambda':>11}{'share of grad':>15}")
    raw, weighted, losses = {}, {}, {}
    for name, (lam, val) in terms.items():
        if z.grad is not None:
            z.grad = None
        val.backward(retain_graph=True)
        raw[name] = float(z.grad.norm())
        weighted[name] = raw[name] * lam
        losses[name] = float(val) * lam
    lt, gt = sum(losses.values()), sum(weighted.values())
    for name, (lam, val) in terms.items():
        print(f"{name:<9}{lam:>8.2f}{float(val):>10.4f}{100 * losses[name] / lt:>14.1f}%"
              f"{raw[name]:>14.4f}{weighted[name]:>11.4f}{100 * weighted[name] / gt:>14.1f}%")

    print("\nIf `share of grad` differs sharply from `share of loss`, the loss values were never the")
    print("right thing to read. recon dominating the gradient means the weighting is wrong; the")
    print("terms being balanced means the weighting is fine and the prediction task is too easy,")
    print("which is step 2i and F54's finding rather than a tuning problem.")
    if enc is not None:
        torch.save(cache, cache_path)


if __name__ == "__main__":
    main()
