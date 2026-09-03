"""Delta-JEPA's LDAD, measured on what we already have before anything is retrained.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/delta_action_decoding.py \\
        --ckpt wm/runs/beh12_ego/teacher_ego.pt \\
        --data data/egocentric/beh12_c08f09t09_ego_flat --embodiment hexapod \\
        --cache results/wm/cache/ego_hex.pt

**Two questions that must not be conflated, and only the second decides anything.**

Delta-JEPA (2606.31232) trains a world model to reconstruct the action from the *difference* of
consecutive state latents, `dz = z_t+1 - z_t`, with a large weight. Its stated target is
action-insensitive collapse, and it explains F168 precisely: given the endpoints `[z_t, z_t+1]` a
decoder can read the action off action-correlated cues in `z_t+1` without modelling the transition
at all, while a difference cannot be read that way. **It was tested on manipulation and navigation.
Neither is periodic.**

  **general**   can the action be read from `dz` at all? F158 measured the passive residual and
                found noise; that is a measurement of what is there, not of what training would put
                there. This run establishes the untrained baseline the LDAD arm has to beat.
  **fine**      can it be read for actions differing by the amounts that F145, F179 and F182 could
                not separate? **This is the deciding one.** Reconstruction is split into the part
                between behaviours -- which F145 already showed is available at 52% -- and the part
                *within* one behaviour, which is the wall.

**The architectural mapping, stated because getting it wrong would invalidate the finding.**
Delta-JEPA's `z` is a *state* latent; ours is the *action* latent, and the state is the V-JEPA2
embedding. So the faithful analogue of `dz` here is `e_t+1 - e_t`, the difference of consecutive
state embeddings, and the LDAD objective would decode the action from that. Using `z_t+1 - z_t`
instead would be decoding the action from a difference of action codes, which is a different and
much easier question.

**And the response-separation diagnostic, their Figure 6.** Hold `e_t` fixed, vary the action, and
ask whether the model's predicted responses are distinguishable. Reported for actions differing
finely and for actions from different behaviours, so our wall is measured in their terms.
"""
import argparse
import collections
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
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
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
    ap.add_argument("--projector", default="",
                    help="**a stage-1 pretrain carries no projector**, so an arm's `best.pt` cannot "
                         "supply one and the response-separation rows need it. Point this at a "
                         "projector fitted against THIS checkpoint -- fitted against another one it "
                         "is a different latent space, which is the F160 trap. Omit it and the "
                         "reconstruction rows still run; only the ratio is skipped.")
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="the fine perturbation, in units of each joint's own sd. **0.5 is the one "
                         "F144 ranked and F179 measured at 2.5% outcome separation**, so the "
                         "response-separation rows below are about exactly those candidates.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    proj = None
    saved = ck if "projector" in ck else (
        torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
        if args.projector else None)
    if saved is not None:
        proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
        proj.load_state_dict(saved.get("projector", saved))

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/fid_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    # **Two differences, and only the second can move.** `e_t+1 - e_t` is built from the FROZEN
    # encoder, so it is a property of V-JEPA2 on this data and is identical for every checkpoint --
    # the first LDAD run reported it unchanged to three decimals across lambda 0, 10 and 50, which
    # is what exposed the error rather than any argument. **The quantity LDAD actually trains is the
    # PREDICTED difference**, `FTM(e_t, z) - e_t`, and that is the one a trained arm changes.
    D, P, E, A, cond_id, clip_id = [], [], [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < 4:
            continue
        for t in range(1, len(e) - 2, args.stride):
            D.append((e[t + 1] - e[t]).flatten().half())   # encoder difference: fixed, a data property
            with torch.no_grad():
                z = itm(e[t:t + 1].to(device), e[t + 1:t + 2].to(device))
                P.append((ftm(e[t:t + 1].to(device), z)[0].cpu() - e[t]).flatten().half())
            E.append(e[t].flatten().half())                 # the endpoint, for the contrast
            A.append(c["a"][t].flatten().float())
            cond_id.append(c["cond"]); clip_id.append(ci)
    D = torch.stack(D); P = torch.stack(P); E = torch.stack(E); A = torch.stack(A)
    cond_id = np.array(cond_id); clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    held = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in held for c in clip_id]); tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    An = A.numpy()
    An = (An - An[tr].mean(0)) / (An[tr].std(0) + 1e-6)
    base_of = {c: An[tr][cond_id[tr] == c].mean(0) for c in set(cond_id[tr].tolist())}
    centre = np.stack([base_of.get(c, An[tr].mean(0)) for c in cond_id[te]])

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip\n")

    K_d = gram(D, D, device).numpy(); K_d /= max(np.trace(K_d) / len(K_d), 1e-12)
    K_p = gram(P, P, device).numpy(); K_p /= max(np.trace(K_p) / len(K_p), 1e-12)
    K_e = gram(E, E, device).numpy(); K_e /= max(np.trace(K_e) / len(K_e), 1e-12)

    print(f"  {'features':>44}{'action R2':>11}{'within cond':>13}")
    for name, Kf in (("dz_pred = FTM(e_t,z) - e_t   **LDAD's own target**", K_p),
                     ("dz = e_t+1 - e_t   (encoder only -- CANNOT move)", K_d),
                     ("e_t alone   (the endpoint contrast)", K_e),
                     ("[e_t, dz]   (endpoints, F168's setting)", K_e + K_d)):
        r2, pred, _ = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], An[tr], An[te], folds)
        ss = ((pred - An[te]) ** 2).sum()
        within = 1 - ss / max(float(((An[te] - centre) ** 2).sum()), 1e-9)
        print(f"  {name:>44}{r2:>11.3f}{within:>13.3f}")

    # ---- Delta-JEPA figure 6: are the model's responses to different actions distinguishable? ----
    if proj is None:
        print("\n  **response separation SKIPPED -- no projector.** This checkpoint carries none and")
        print("  `--projector` was not given, so one of the three numbers this run is supposed to")
        print("  report is missing. Fit one against THIS checkpoint with `wm.fit_projector` rather")
        print("  than borrowing another arm's: a projector from a different checkpoint lives in a")
        print("  different latent space and the ratio would be meaningless rather than absent.")
        return
    print(f"\n  response separation, `FTM(e_t, proj(a)) - e_t`, relative to the response's own size")
    sd = A[tr].std(0).to(device)
    idx = torch.nonzero(torch.tensor(te)).flatten()[:200]
    gen = torch.Generator(device=device).manual_seed(0)
    fine, coarse = [], []
    with torch.no_grad():
        # recover `e_t` for the sampled transitions by re-walking the clips; keeping every
        # 256x1408 frame around for this alone would cost several gigabytes
        pos, sample = 0, {int(v): None for v in idx}
        for ci, c in enumerate(clips):
            e = c["e"].float()
            if len(e) < 4:
                continue
            for t in range(1, len(e) - 2, args.stride):
                if pos in sample:
                    sample[pos] = (e[t:t + 1], A[pos])
                pos += 1
        for i in idx:
            e_t, a0 = sample[int(i)]
            e_t = e_t.to(device)
            a0 = a0.to(device)
            a_fine = a0 + args.sigma * sd * torch.randn(a0.shape, generator=gen, device=device)
            j = int(torch.randint(len(A), (1,), generator=gen, device=device))
            a_far = A[j].to(device)
            r0, rf, rc = (ftm(e_t, proj(x.unsqueeze(0), args.embodiment)) - e_t
                          for x in (a0, a_fine, a_far))
            scale = float(r0.norm()) + 1e-12
            fine.append(float((rf - r0).norm()) / scale)
            coarse.append(float((rc - r0).norm()) / scale)
    print(f"  {'a perturbed by ' + str(args.sigma) + ' sd (fine)':>44}{np.mean(fine):>11.3f}")
    print(f"  {'a from another transition (coarse)':>44}{np.mean(coarse):>11.3f}")
    ratio = np.mean(coarse) / max(np.mean(fine), 1e-9)
    print(f"  {'ratio coarse / fine':>44}{ratio:>11.1f}x")
    print(f"  {'the same ratio in PHYSICS (12.8% / 2.5%)':>44}{'5.1x':>11}")
    print(f"  {'model over physics':>44}{ratio / 5.12:>11.2f}")
    print("  **Below 1.0 means the model separates fine from coarse LESS than the world does** --")
    print("  it treats a fine perturbation as relatively more distinct than physics makes it. That")
    print("  is mis-proportion, not collapse, and an anti-collapse objective does not address it.")
    print("  **Re-read this line after any LDAD arm**: moving toward 5.1x is the term helping and")
    print("  moving further below it is the term making the mis-proportion worse.")

    print("\n  **Read `dz_pred`, not `dz`.** The encoder row is computed from frozen V-JEPA2")
    print("  embeddings and is identical for every checkpoint by construction; it is a property of")
    print("  the data and a reference, never a result. `dz_pred` is what the LDAD term trains.")
    print("\n  **`within cond` is the deciding column and `dz_pred` has to clear it, not the first one.**")
    print("  Between behaviours the action is already recoverable and F145 reads 52% on it; the")
    print("  wall is inside one behaviour. **A dz that reconstructs the action overall and not")
    print("  within a condition is the same wall relocated**, which is the outcome Delta-JEPA's")
    print("  own evidence cannot rule out -- manipulation and navigation are not periodic.")
    print("\n  These are the UNTRAINED numbers. LDAD trains dz to carry the action, so this is the")
    print("  baseline it must beat, not a verdict on the method.")


if __name__ == "__main__":
    main()
