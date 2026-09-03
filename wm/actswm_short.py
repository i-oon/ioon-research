"""A short run of the ActSWM objective, to see whether it is safe to spend five hours on it.

    .venv/bin/python3 -m wm.actswm_short --steps 400

**This is not a pretrain and is not a substitute for one.** It continues from the existing
three-channel checkpoint, so it answers "does this objective put gradient into the latent, move
`/mean-z`, and leave the prediction standing" -- and not "what does pretraining from scratch under
it produce". Those are different questions and only the first is cheap.

**The objective, with ActSWM's Table 3 values where they are not physics-dependent:**

    alpha_pred = 1.0    the prediction loss, unchanged
    lambda_hinge = 0.5  separation between the real-action and null-action rollouts, hinged at
                        margin m = 0.3, accumulated over K steps
    lambda_readout = 1.0  the frozen readout must recover the real action from the *predicted*
                        transition, which is what forces the prediction to carry it

**Two of their settings are deliberately not copied.**

`K`: they use 12. **Ours is 5**, because F140 and F150 measure the rolled prediction crossing
"worse than a frozen frame" by five steps on this checkpoint. Hinging separation at horizons where
the prediction is already broken trains on noise. Ready to drop to 3.

`H`, context length: they use 32 frames. **Our forward model is conditioned on a single frame** --
`FTM(e_t, z)`, 256 tokens, no temporal stack -- so 32 is not a hyperparameter here, it is a
different architecture. Nothing in F138 or F150 implicates context length, so it stays at 1.

**The null is in latent space, not action space, and it has to be.** F148 established the null as
each body's standing *stance* -- an action. **Pretraining has no action projector**: `wm/train.py`
drives the forward model with the ITM's `z`, and the projector is fitted afterwards against a frozen
ITM. Mapping the stance into `z` is therefore impossible at this stage, and a hinge built on
`proj(stance)` puts **zero gradient into `z`** -- measured, both bodies, before this was corrected.
The pretraining-compatible null is `ITM(e_t, e_t)`: the latent of *nothing happened*, which needs no
projector and carries the same meaning.

**`lambda_sig = 0.09` is not applied**, and that is deliberate: which term it scales cannot be
determined from the number alone, and guessing would put a value in the log under a name that may
not mean what the paper means. Flagged rather than assumed.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from diagnostics.objective_experiments.check_actswm_wiring import FrozenActionReadout, stance_action  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def rollouts(ftm, e0, z_real, z_null, K):
    """K steps from the same frame on the real action and on the null action."""
    real, null, pair = e0, e0, []
    for k in range(K):
        real = ftm(real, z_real[:, min(k, z_real.shape[1] - 1)])
        null = ftm(null, z_null)
        pair.append((real, null))
    return pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hex-b1_body3/best.pt")
    ap.add_argument("--projector", default="wm/runs/beh12_hex-b1_body3/projector_b1_adapted.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2,
                    help="**2, not more.** Rolling K steps through the forward model keeps every "
                         "intermediate 256x1408 activation for the backward pass; batch 4 at K=5 "
                         "exhausts an 11 GB card")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--K", type=int, default=5, help="**not ActSWM's 12** -- our reliable horizon")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--lambda_hinge", type=float, default=0.5)
    ap.add_argument("--lambda_readout", type=float, default=1.0)
    ap.add_argument("--alpha_pred", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25,
                    help="the separation curve is the diagnostic, not the endpoint: F151 read "
                         "0.019, 0.137, 0.496, 0.008 at 100-step spacing, which is an oscillation "
                         "that a coarser log would have shown as a single number")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device); ftm.load_state_dict(ck["ftm"])
    saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
    proj.load_state_dict(saved["projector"])
    for p in proj.parameters():
        p.requires_grad_(False)

    bodies = {"hexapod": ("data/allocentric/beh12_c10f10t10_flat", "results/wm/cache/hex_c10.pt", 18),
              "b1": ("data/allocentric/beh12_b1_flat", "results/wm/cache/b1_body3.pt", 12)}
    readout = {n: FrozenActionReadout(cfg.token_dim, d, hidden=args.hidden).to(device)
               for n, (_, _, d) in bodies.items()}
    data, znull = {}, {}
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    for name, (d, cache_path, _dim) in bodies.items():
        cache = torch.load(os.path.join(ROOT, cache_path), map_location="cpu")
        data[name] = gather(os.path.join(ROOT, d), name, enc, ck, cache, 2,
                            max(1, cfg.action_lag), device)[:args.clips]
        with torch.no_grad():
            znull[name] = proj(torch.tensor(stance_action(d, name),
                                            device=device).unsqueeze(0), name)
    del enc
    torch.cuda.empty_cache()

    def batch(name, rng):
        clips = data[name]
        e_t, e_next, a_seq = [], [], []
        for _ in range(args.batch):
            c = clips[int(rng.integers(len(clips)))]
            t = int(rng.integers(2, max(3, c["n"] - args.K - 1)))
            e_t.append(c["e"][t].float()); e_next.append(c["e"][t + 1].float())
            a_seq.append(torch.stack([c["a"][min(t + k, c["n"] - 1)] for k in range(args.K)]))
        return (torch.stack(e_t).to(device), torch.stack(e_next).to(device),
                torch.stack(a_seq).to(device))

    def report(tag):
        """Per body: gradient into z from each term, sensitivity, prediction against a frozen frame."""
        itm.eval(); ftm.eval()
        print(f"\n  {tag}")
        print(f"    {'body':<9}{'|dL/dz| pred':>14}{'hinge':>10}{'readout':>10}"
              f"{'/mean-z':>10}{'pred/frozen':>13}")
        rng = np.random.default_rng(args.seed + 1)
        for name in bodies:
            e0, e1, aseq = batch(name, rng)
            z = itm(e0, e1); z.retain_grad()
            z_stat = itm(e0, e0)          # the latent of "nothing happened" -- the null, in z
            zs = z.unsqueeze(1).expand(-1, args.K, -1)
            pair = rollouts(ftm, e0, zs, z_stat, args.K)
            terms = {}
            terms["pred"] = args.alpha_pred * F.mse_loss(ftm(e0, z), e1)
            sep = torch.stack([1 - F.cosine_similarity(r.flatten(1), n.flatten(1), dim=1).mean()
                               for r, n in pair])
            terms["hinge"] = args.lambda_hinge * F.relu(args.margin - sep).mean()
            terms["readout"] = args.lambda_readout * F.mse_loss(
                readout[name](e0, pair[0][0]), aseq[:, 0])
            g = {}
            for k, v in terms.items():
                if z.grad is not None:
                    z.grad = None
                v.backward(retain_graph=True)
                g[k] = float(z.grad.norm()) if z.grad is not None else 0.0
            with torch.no_grad():
                real, null = e0, e0
                for k in range(args.K):
                    real = ftm(real, z); null = ftm(null, z_stat)
                mz = float((real - e1).pow(2).mean() / (null - e1).pow(2).mean())
                pf = float(F.mse_loss(ftm(e0, z), e1) / F.mse_loss(e0, e1))
            print(f"    {name:<9}{g['pred']:>14.5f}{g['hinge']:>10.5f}{g['readout']:>10.5f}"
                  f"{mz:>10.3f}{pf:>13.3f}")
            del e0, e1, aseq, z, z_stat, zs, pair, terms
            torch.cuda.empty_cache()
        itm.train(); ftm.train()

    report("BEFORE — the F150 baseline, re-read on this batch")
    curve = []
    opt = torch.optim.Adam(list(itm.parameters()) + list(ftm.parameters()), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    names = list(bodies)
    for step in range(args.steps):
        name = names[step % len(names)]
        e0, e1, aseq = batch(name, rng)
        z = itm(e0, e1)
        z_stat = itm(e0, e0)
        zs = z.unsqueeze(1).expand(-1, args.K, -1)
        pair = rollouts(ftm, e0, zs, z_stat, args.K)
        sep = torch.stack([1 - F.cosine_similarity(r.flatten(1), n.flatten(1), dim=1).mean()
                           for r, n in pair])
        loss = (args.alpha_pred * F.mse_loss(ftm(e0, z), e1)
                + args.lambda_hinge * F.relu(args.margin - sep).mean()
                + args.lambda_readout * F.mse_loss(readout[name](e0, pair[0][0]), aseq[:, 0]))
        l_val, s_val = float(loss), float(sep.mean())
        opt.zero_grad(); loss.backward(); opt.step()
        del e0, e1, aseq, z, z_stat, zs, pair, sep, loss
        curve.append((step + 1, name, l_val, s_val))
        if (step + 1) % args.log_every == 0:
            recent = [c for c in curve[-args.log_every:]]
            per = {n: np.mean([c[3] for c in recent if c[1] == n]) for n in names}
            print(f"  step {step + 1:4d}  loss {l_val:.4f}  separation  "
                  + "  ".join(f"{n} {per[n]:.4f}" for n in names), flush=True)
    report("AFTER")
    print("\n  separation, per body, over training — a working hinge holds it, F151's oscillated")
    for n in names:
        pts = [c[3] for c in curve if c[1] == n]
        block = max(1, len(pts) // 8)
        marks = [float(np.mean(pts[i:i + block])) for i in range(0, len(pts), block)][:8]
        print(f"    {n:<9}" + "  ".join(f"{v:.3f}" for v in marks)
              + f"   margin {args.margin}")

    if args.out:
        torch.save({**ck, "itm": itm.state_dict(), "ftm": ftm.state_dict(),
                    "actswm": vars(args)}, os.path.join(ROOT, args.out))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
