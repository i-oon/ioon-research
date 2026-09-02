"""Is the action *readable* from what the decoder is given, or does the trained head fail to read it?

    .venv/bin/python3 scripts/diagnostics/motion_decoder_ceiling.py \\
        --ckpt wm/runs/beh12_ego/best.pt --data data/egocentric/beh12_c08f09t09_ego_flat \\
        --embodiment hexapod

**The gate teacher-student sits on.** Training on egocentric video, `MotionDecoder` reaches 0.076 on
train motion and never leaves 1.53 on validation (F172) -- above 1.0, so worse than predicting the
training mean, so `R2 = -0.53`. Teacher-student, the action projector and every path that emits a
joint command run through that head. **Two very different things produce that curve and they call
for opposite responses**, so it has to be separated before anything is built on it:

  1. the information is **not there**. Egocentric `e_t` no longer states the pose -- that is Q1's
     result, single-frame action R2 0.779 allocentric to 0.293 egocentric -- and if `z` does not
     supply the remainder, no head can recover the command and teacher-student has no target.
  2. the information **is** there and the trained head overfits to it. Then the curve is a training
     problem, not a representation problem, and the component is repairable.

A dual ridge on frozen features answers that directly, because it is the same question with the
architecture and the optimiser removed. **Its R2 is on the same scale as the training curve's
normalised MSE** -- `R2 = 1 - MSE` -- so the two are read against each other without conversion.

    e_t          the frame alone -- Q1's number, and the ceiling if `z` contributes nothing
    z            `ITM(e_t, e_t+1)` alone -- the latent, with no visual context
    [e_t, z]     **the MotionDecoder's own input.** This one is the gate

The two kernels are each normalised by their mean diagonal before being summed, or the 360,448-dim
embedding would drown a 64-dim latent and `[e_t, z]` would silently be `e_t`.

Split is **by clip**, in families, the same rule as `inverse_dynamics_r2.py` -- a within-clip split
on periodic locomotion scores the neighbouring frame, not generalisation.
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--bar", type=float, default=None,
                    help="the trained head's R2, `1 - val_motion` from the run being examined; "
                         "-0.53 for `beh12_ego`. **Omitted rather than defaulted**, because a "
                         "default would print a headroom against a placeholder and a placeholder "
                         "in a results table is how a number gets quoted that nothing measured.")
    ap.add_argument("--reference", type=float, nargs=3, default=None,
                    metavar=("E_T", "Z", "BOTH"),
                    help="the allocentric arm's three R2 values, to print a delta column: "
                         "`0.773 0.903 0.938` for the hexapod. **The comparison that matters is "
                         "per column, not on `[e_t, z]` alone** -- see the verdict text.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])

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

    E, Z, A, clip_id = [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < 4:
            continue
        for t in range(1, len(e) - 2, args.stride):
            E.append(e[t].flatten().half())
            with torch.no_grad():
                Z.append(itm(e[t:t + 1].to(device), e[t + 1:t + 2].to(device))[0].float().cpu())
            A.append(c["a"][t].flatten().float())
            clip_id.append(ci)
    E = torch.stack(E)
    Z = torch.stack(Z).numpy()
    A = torch.stack(A).numpy()
    clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id])
    tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    A = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)
    Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip, "
          f"action width {A.shape[1]}, z width {Z.shape[1]}\n")

    K_e = gram(E, E, device).numpy()
    K_z = Z @ Z.T
    K_e /= max(np.trace(K_e) / len(K_e), 1e-12)
    K_z /= max(np.trace(K_z) / len(K_z), 1e-12)
    feats = {"e_t  (frame alone)": K_e,
             "z    (latent alone)": K_z,
             "[e_t, z]  (the decoder's input)": K_e + K_z}

    print(f"  {'features':>34}{'action R2':>11}{'alpha':>9}")
    got = {}
    for name, Kf in feats.items():
        r2, _, alpha = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], A[tr], A[te], folds)
        got[name] = r2
        print(f"  {name:>34}{r2:>11.3f}{alpha:>9.3g}")

    e_only, z_only, both = (got["e_t  (frame alone)"], got["z    (latent alone)"],
                            got["[e_t, z]  (the decoder's input)"])
    if args.reference:
        r_e, r_z, r_b = args.reference
        print(f"\n  {'against the allocentric arm':>34}{'this run':>11}{'allo':>9}{'delta':>9}")
        for label, v, r in (("e_t", e_only, r_e), ("z", z_only, r_z), ("[e_t, z]", both, r_b)):
            print(f"  {label:>34}{v:>11.3f}{r:>9.3f}{v - r:>+9.3f}")
    if args.bar is not None:
        print(f"\n  trained MotionDecoder, same scale: R2 {args.bar:+.3f}   ridge on the same "
              f"input: {both:+.3f}   headroom {both - args.bar:+.3f}")

    print("\n  **Read the three columns, not the last one.** Egocentrically `e_t` is *expected* to")
    print("  fall -- that is Q1, single-frame action R2 0.779 to 0.293, and it is the thing that")
    print("  made the forward model use the action at all. So `[e_t, z]` landing under the")
    print("  allocentric 0.938 is not by itself a failure, and judging on that column alone would")
    print("  read the intended change as a defect.")
    print("\n  **The question is whether the burden shifted to `z`.** F168 has `z` carrying the")
    print("  action; if `z`-only holds near its allocentric 0.903 while `e_t`-only falls, the")
    print("  command still exists in what the decoder is shown, and the head has to be refitted to")
    print("  read it from `z` rather than from the frame. **That is a repair, and teacher-student")
    print("  opens.** If `e_t`-only and `z`-only *both* fall, neither input carries the command,")
    print("  no refit recovers it, and teacher-student needs rethinking rather than repairing.")


if __name__ == "__main__":
    main()
