"""How far can the forward model be imagined forward before its *state* stops being right?

    .venv/bin/python3 scripts/diagnostics/forward_model/rollout_fidelity.py \\
        --ckpt wm/runs/beh12_hex-b1_body3/best.pt --data data/allocentric/beh12_c08f09t09_flat \\
        --embodiment hexapod

**The bar a distilled policy would be trained against, and it is tighter than any measured so far.**
F126 asked whether a rollout ranks behaviours correctly, which a model can do while its predicted
state drifts anywhere. Teacher-student trains on states the model *imagines*, so what matters is
whether the imagined embedding is the one the robot would actually have been in.

Three questions, deliberately separated, because the fix differs for each:

  1. **fidelity**   rolled `e_t+h` against the true `e_t+h`, as a ratio to holding `e_t` still.
                    Below 1.0 beats predicting no motion; the divergence horizon is where it
                    crosses.
  2. **cause**      the one-step error isolates prediction quality, i.e. the pretraining
                    objective; the growth of error with `h` isolates compounding. A model can be
                    excellent at one step and useless at ten, or mediocre at one and no worse at
                    ten, and those call for opposite responses.
  3. **manifold**   the same measurement with the action perturbed off the distribution the model
                    was fitted on. If fidelity holds on-manifold and collapses off it, distillation
                    that stays near recorded behaviour is unaffected.

**Actions are teacher-forced from the ITM on the true frames**, so this measures the forward model
alone; the projector's own limits (F97) would otherwise be folded in.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="clips the checkpoint never trained on")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--latent", choices=("itm", "projector"), default="itm",
                    help="which latent drives the roll. **`projector` is the path a distilled "
                         "policy would use** and the one F119's `/mean-z` was measured on; `itm` "
                         "is the teacher-forced upper bound. The forward model is action-sensitive "
                         "only inside the projector's region (F139), so a sweep has to report the "
                         "projector path or it measures a subspace nothing drives.")
    ap.add_argument("--mean_z", action="store_true",
                    help="also roll on a **mean latent**, and on two of them, because which mean "
                         "is used changes the answer by a factor of two. `wm/adapt3` averages "
                         "latents from across the dataset, which is what F119's 0.49 measures: the "
                         "real action against an *average behaviour*. Averaging within the clip "
                         "instead asks whether the action matters *inside* one behaviour, and on "
                         "the same checkpoint that reads 0.95. **The second is the stricter "
                         "question and the one control depends on**; both are printed.")
    ap.add_argument("--family_mean", action="store_true",
                    help="add a third `/mean-z` baseline: the mean latent of the **same behaviour "
                         "family at other magnitudes**. **This is what separates a task property "
                         "from a model failure.** Within one clip the gait is periodic at one "
                         "speed, so a frame fixes the phase and the action is genuinely near "
                         "redundant -- a high `/mean-z` there says nothing about the model. Across "
                         "the family the magnitude varies and the action is *not* redundant, so a "
                         "high `/mean-z` there is the model discarding information that exists.")
    ap.add_argument("--noise", type=float, default=1.0,
                    help="off-manifold perturbation, in units of the latent's own sd")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    projector = None
    if args.latent == "projector":
        from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
        projector = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
        projector.load_state_dict(ck["projector"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    print(f"{args.ckpt}\n{len(clips)} held-out clips of {args.embodiment} from {args.data}\n")
    print(f"  {'horizon':>8}{'model':>10}{'hold still':>12}{'ratio':>8}{'moves':>8}"
          f"{'off-manifold':>14}{'shuffled z':>12}{'n':>7}")

    z_sd = None
    rows = {}
    with torch.no_grad():
        # **The across-clips mean, computed once.** `wm/adapt3` takes it from its first batch; the
        # point is only that it is not the clip's own mean, which is a far stronger baseline.
        pool = []
        for c in clips:
            e = c["e"].float().to(device)
            pool.append((projector(c["a"][:len(e) - 1].to(device), args.embodiment)
                         if projector is not None else
                         torch.cat([itm(e[t:t + 1], e[t + 1:t + 2])
                                    for t in range(len(e) - 1)])).mean(0, keepdim=True))
        z_global = torch.cat(pool).mean(0, keepdim=True)
        # the family mean: same behaviour, the other magnitudes, excluding the clip itself
        fam_mean = {}
        if args.family_mean:
            import collections
            fam_of = lambda cond: ("side" if cond.startswith("side") else cond.split("_")[0])
            by = collections.defaultdict(list)
            for c_, m_ in zip(clips, pool):
                by[fam_of(c_["cond"])].append((c_["cond"], m_))
            for c_ in clips:
                f_ = fam_of(c_["cond"])
                others = [m_ for cond_, m_ in by[f_] if cond_ != c_["cond"]]
                if others:
                    fam_mean[c_["cond"]] = torch.cat(others).mean(0, keepdim=True)
        for h in args.horizons:
            tot = {"model": 0.0, "hold": 0.0, "moved": 0.0, "truth": 0.0, "off": 0.0,
                   "shuf": 0.0, "meanz": 0.0, "meanz_all": 0.0, "meanz_fam": 0.0}
            n = 0
            for c in clips:
                e = c["e"].float().to(device)
                if projector is None:
                    zs = torch.cat([itm(e[t:t + 1], e[t + 1:t + 2])
                                    for t in range(len(e) - 1)])
                else:
                    zs = projector(c["a"][:len(e) - 1].to(device), args.embodiment)
                if z_sd is None:
                    z_sd = zs.std(0, keepdim=True)
                # a real latent from elsewhere in the same clip: an action the model has seen,
                # paired with a state it never followed
                order = torch.randperm(len(zs), generator=torch.Generator().manual_seed(0))
                for t in range(1, len(e) - h - 1, args.stride):
                    truth = e[t + h]
                    seq = zs[t:t + h]
                    pred = e[t:t + 1]
                    off = e[t:t + 1]
                    shuf = e[t:t + 1]
                    mz = e[t:t + 1]
                    mz_all = e[t:t + 1]
                    mz_fam = e[t:t + 1]
                    z_bar = zs.mean(0, keepdim=True)
                    noise = args.noise * z_sd * torch.randn(
                        seq.shape, generator=torch.Generator(device=device).manual_seed(t),
                        device=device)
                    for i in range(h):
                        pred = ftm(pred, seq[i:i + 1])
                        off = ftm(off, seq[i:i + 1] + noise[i:i + 1])
                        shuf = ftm(shuf, zs[order[(t + i) % len(zs)]].unsqueeze(0))
                        mz = ftm(mz, z_bar)
                        mz_all = ftm(mz_all, z_global)
                        mz_fam = ftm(mz_fam, fam_mean.get(c["cond"], z_global))
                    tot["model"] += float(((pred[0] - truth) ** 2).mean())
                    tot["hold"] += float(((e[t] - truth) ** 2).mean())
                    tot["moved"] += float(((pred[0] - e[t]) ** 2).mean())
                    tot["truth"] += float(((truth - e[t]) ** 2).mean())
                    tot["off"] += float(((off[0] - truth) ** 2).mean())
                    tot["shuf"] += float(((shuf[0] - truth) ** 2).mean())
                    tot["meanz"] += float(((mz[0] - truth) ** 2).mean())
                    tot["meanz_all"] += float(((mz_all[0] - truth) ** 2).mean())
                    tot["meanz_fam"] += float(((mz_fam[0] - truth) ** 2).mean())
                    n += 1
            r = {k: v / max(n, 1) for k, v in tot.items()}
            rows[h] = r
            print(f"  {h:>8}{r['model']:>10.4f}{r['hold']:>12.4f}"
                  f"{r['model'] / max(r['hold'], 1e-9):>8.3f}"
                  f"{r['moved'] / max(r['truth'], 1e-9):>8.2f}"
                  f"{r['off'] / max(r['hold'], 1e-9):>14.3f}"
                  f"{r['shuf'] / max(r['hold'], 1e-9):>12.3f}{n:>7}"
                  + (f"   /mean-z within-clip {r['model'] / max(r['meanz'], 1e-9):.3f}"
                     + (f"  within-family {r['model'] / max(r['meanz_fam'], 1e-9):.3f}"
                        if args.family_mean else "")
                     + f"  across-all {r['model'] / max(r['meanz_all'], 1e-9):.3f}"
                     if args.mean_z else ""))

    h0 = min(rows)
    one = rows[h0]["model"] / max(rows[h0]["hold"], 1e-9)
    hs = sorted(rows)
    span = hs[-1] - hs[0]
    slope = (((rows[hs[-1]]["model"] / max(rows[hs[-1]]["hold"], 1e-9)) - one) / span
             if span else float("nan"))
    diverge = next((h for h in hs if rows[h]["model"] >= rows[h]["hold"]), None)
    print(f"\n  one-step ratio {one:.3f}   growth per step {slope:+.4f}   "
          f"divergence horizon {diverge if diverge else '> ' + str(hs[-1])}")
    print("\n  `ratio` is the rolled state's error over the error of holding `e_t` still: **below")
    print("  1.0 is better than predicting no motion at all**. `moves` is predicted displacement")
    print("  over actual -- near 0 is the collapse where the model copies its input and scores")
    print("  1.0 by construction. `off-manifold` perturbs the latent, `shuffled z` feeds a real")
    print("  latent from a state it never followed.")
    print("\n  **A high one-step ratio blames the pretraining objective; a low one with a steep")
    print("  slope blames compounding and is answered by capping the imagined horizon.**")


if __name__ == "__main__":
    main()
