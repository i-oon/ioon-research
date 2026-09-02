"""Does the student's pooling throw away the thing that makes egocentric work?

    .venv/bin/python3 scripts/diagnostics/pooled_student_check.py \\
        --ckpt wm/runs/beh12_ego/best.pt --data data/egocentric/beh12_c08f09t09_ego_flat \\
        --embodiment hexapod --cache results/wm/cache/ego_hex.pt

**P3: this decides the student's architecture, and it decides it before a student exists.** F174's
mechanism hypothesis is that egocentrically the joint command lives in *where things sit in the
frame*: a linear ridge on flattened tokens reads 0.334 on the B1 while cross-attention over the same
tokens reaches 0.778, a **2.3x** gap against **1.15x** allocentrically. `sim/control/teacher_student_insect.py`'s
`Student` takes `pooled(e) = e.mean(-2)` -- **one vector per frame, spatial layout discarded** -- and
its own docstring flags the trap and accepts it to hold 20 Hz.

**If pooling eats most of that gap, a student trains toward 0.334 rather than 0.778 and
teacher-student fails for a reason that has nothing to do with the teacher.** No amount of label
quality repairs a policy that pooled the signal away, so this runs before Q1 is interpreted.

Four rows, and the pooled ones are the question:

    e_t                    every token -- the reference F173/F174 were measured on
    pooled(e_t)            **the student's actual visual input**
    [pooled(e_t), z]       the student if it were also handed the latent
    pooled, MLP            the student's actual architecture, not a ridge: the same 512-wide MLP
                           `Student` uses, fitted on the same split

**The MLP row exists because a ridge would confound pooling with linearity.** `Student` is nonlinear;
if only the ridge rows were reported, a drop would be unattributable between "pooling lost the
layout" and "a linear map cannot use it", which are opposite conclusions.

**Read every number against the same measurement on the allocentric checkpoint**, never on its own:
the question is not how much pooling costs, it is whether it costs *more* egocentrically. Split is
by clip, in families, the same rule as `motion_decoder_ceiling.py`.
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
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def fit_mlp(X_tr, y_tr, X_te, y_te, device, hidden=512, epochs=400, wd=1e-2, seed=0):
    """`Student`'s own head: two GELU layers of 512, on whatever features it is given."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X_tr.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, y_tr.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=wd)
    Xtr, ytr = X_tr.to(device), y_tr.to(device)
    Xte, yte = X_te.to(device), y_te.to(device)
    best = -1e9
    for epoch in range(epochs):
        net.train()
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(idx), 64):
            sl = idx[i:i + 64]
            loss = torch.nn.functional.mse_loss(net(Xtr[sl]), ytr[sl])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                pred = net(Xte)
            ss = float(((pred - yte) ** 2).sum())
            best = max(best, 1 - ss / max(float(((yte - ytr.mean(0)) ** 2).sum()), 1e-9))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--projector", default="",
                    help="adds the **causal** rows: `proj(a_t-1)` is a latent the student could "
                         "actually be handed at run time, since `z = ITM(e_t, e_t+1)` needs the "
                         "future and deployment has none. **The file must be the projector fitted "
                         "against THIS checkpoint** -- `projector_ego.pt` for `beh12_ego`. A "
                         "projector fitted against an adapted checkpoint lives in a different "
                         "latent space, and comparing two `z` spaces reads noise as a result.")
    ap.add_argument("--reference", type=float, nargs=2, default=None, metavar=("RIDGE", "REFIT"),
                    help="the full-token numbers for this body: `0.608 0.847` insect egocentric, "
                         "`0.334 0.778` B1 egocentric, `0.938 0.982` and `0.789 0.910` allocentric. "
                         "Printed as the denominators so the pooled rows cannot be read alone.")
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

    projector = None
    if args.projector:
        saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu",
                           weights_only=False)
        projector = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
        projector.load_state_dict(saved.get("projector", saved))

    E, P, Z, A, Aprev, Zprev, clip_id = [], [], [], [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        if len(e) < 4:
            continue
        for t in range(1, len(e) - 2, args.stride):
            E.append(e[t].flatten().half())
            P.append(e[t].mean(-2).float())            # the student's pooled view
            with torch.no_grad():
                Z.append(itm(e[t:t + 1].to(device), e[t + 1:t + 2].to(device))[0].float().cpu())
            A.append(c["a"][t].flatten().float())
            # **strictly causal**: the command one step back, and its latent
            prev = c["a"][t - 1].flatten().float()
            Aprev.append(prev)
            if projector is not None:
                with torch.no_grad():
                    Zprev.append(projector(prev[None].to(device), args.embodiment)[0].cpu())
            clip_id.append(ci)
    E = torch.stack(E)
    P = torch.stack(P); Z = torch.stack(Z); A = torch.stack(A); Aprev = torch.stack(Aprev)
    Zprev = torch.stack(Zprev) if Zprev else None
    clip_id = np.array(clip_id)

    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id]); tr = ~te
    for M in ([P, Z, A, Aprev] + ([Zprev] if Zprev is not None else [])):
        M.sub_(M[tr].mean(0)).div_(M[tr].std(0) + 1e-6)
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    An = A.numpy()

    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}")
    print(f"{tr.sum()} train / {te.sum()} test transitions, split by clip, "
          f"pooled width {P.shape[1]}\n")

    K_e = gram(E, E, device).numpy(); K_e /= max(np.trace(K_e) / len(K_e), 1e-12)
    K_p = (P @ P.T).numpy(); K_p /= max(np.trace(K_p) / len(K_p), 1e-12)
    K_z = (Z @ Z.T).numpy(); K_z /= max(np.trace(K_z) / len(K_z), 1e-12)
    POOLED = "pooled(e_t)  (the student's input)"
    rows = {"e_t  (every token)": K_e, POOLED: K_p, "[pooled(e_t), z]": K_p + K_z}

    print(f"  {'features':>38}{'action R2':>11}{'alpha':>9}")
    got = {}
    for name, Kf in rows.items():
        r2, _, alpha = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], An[tr], An[te], folds)
        got[name] = r2
        print(f"  {name:>38}{r2:>11.3f}{alpha:>9.3g}")

    if Zprev is not None:
        K_zp = (Zprev @ Zprev.T).numpy(); K_zp /= max(np.trace(K_zp) / len(K_zp), 1e-12)
        K_ap = (Aprev @ Aprev.T).numpy(); K_ap /= max(np.trace(K_ap) / len(K_ap), 1e-12)
        # **The control, and without it the causal rows are unreadable.** In periodic locomotion
        # `a_t-1` nearly determines `a_t` on its own, so `[pooled, proj(a_t-1)]` beating `pooled`
        # could be the latent helping or it could be plain autoregression. Feeding the raw previous
        # command as a fourth row separates them: if raw does as well, the lift is not about `z`.
        for name, Kf in (("[pooled, proj(a_t-1)]  (causal z)", K_p + K_zp),
                         ("[pooled, a_t-1]  (the autoregression control)", K_p + K_ap)):
            r2, _, alpha = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], An[tr], An[te], folds)
            got[name] = r2
            print(f"  {name:>38}{r2:>11.3f}{alpha:>9.3g}")

    mlp_p = fit_mlp(P[tr], A[tr], P[te], A[te], device)
    mlp_pz = fit_mlp(torch.cat([P, Z], 1)[tr], A[tr], torch.cat([P, Z], 1)[te], A[te], device)
    print(f"\n  {'pooled, Student MLP':>38}{mlp_p:>11.3f}")
    print(f"  {'[pooled, z], Student MLP':>38}{mlp_pz:>11.3f}")

    if args.reference:
        r_ridge, r_refit = args.reference
        # **Like with like, or the ratio means nothing.** `pooled` alone is compared against `e_t`
        # alone, not against `[e_t, z]`; the student in the loop is handed a goal and not `z`, so
        # the pooled-alone row is the one that bounds it and it needs its own honest denominator.
        print(f"\n  {'like against like':>38}{'pooled':>11}{'full':>9}{'kept':>8}")
        print(f"  {'ridge, no z':>38}{got[POOLED]:>11.3f}{got['e_t  (every token)']:>9.3f}"
              f"{got[POOLED] / max(got['e_t  (every token)'], 1e-9):>8.0%}")
        print(f"  {'ridge, with z':>38}{got['[pooled(e_t), z]']:>11.3f}{r_ridge:>9.3f}"
              f"{got['[pooled(e_t), z]'] / max(r_ridge, 1e-9):>8.0%}")
        print(f"  {'nonlinear, with z':>38}{mlp_pz:>11.3f}{r_refit:>9.3f}"
              f"{mlp_pz / max(r_refit, 1e-9):>8.0%}")
        print(f"\n  {'what the student can actually be given':>38}{'':>11}")
        print(f"  {'pooled alone, Student MLP':>38}{mlp_p:>11.3f}")
        if Zprev is not None:
            mlp_zp = fit_mlp(torch.cat([P, Zprev], 1)[tr], A[tr],
                             torch.cat([P, Zprev], 1)[te], A[te], device)
            mlp_ap = fit_mlp(torch.cat([P, Aprev], 1)[tr], A[tr],
                             torch.cat([P, Aprev], 1)[te], A[te], device)
            print(f"  {'+ proj(a_t-1), Student MLP':>38}{mlp_zp:>11.3f}"
                  f"{mlp_zp - mlp_p:>+9.3f}")
            print(f"  {'+ a_t-1 raw, the control':>38}{mlp_ap:>11.3f}"
                  f"{mlp_ap - mlp_p:>+9.3f}")
            print("\n  **Read the two causal rows against each other, not against pooled alone.**")
            print("  A gait is periodic, so a_t-1 nearly states a_t by itself; if the raw control")
            print("  lifts as much as proj(a_t-1) does, the lift is autoregression and says nothing")
            print("  about the latent. Only proj(a_t-1) clearly beating raw a_t-1 is evidence that")
            print("  feeding the student a causal z is worth the architecture change.")

    print("\n  **The comparison is against the same rows on the allocentric checkpoint, never")
    print("  against these alone.** The question is not what pooling costs -- it is whether it")
    print("  costs MORE egocentrically, which is what F174's spatial hypothesis predicts and what")
    print("  would mean a pooling student trains toward the linear floor instead of the refit")
    print("  reference. **If it does, the student needs the token grid and the 20 Hz budget has to")
    print("  be revisited before anything is built; no teacher quality repairs it.**")


if __name__ == "__main__":
    main()
