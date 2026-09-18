"""Does `body_head`'s hidden layer cluster by BEHAVIOUR across embodiments, where raw `z` does not?

**Why this exists.** A same-night diagnostic found raw `z` (ITM's output) does not cluster by
behaviour across B1 and the hexapod: cross-embodiment same-behaviour cosine similarity 0.713 vs
different-behaviour 0.632, a gap smaller than either group's own spread (std ~0.09-0.10), and B1's
own clips barely separate from each other by behaviour at all (0.82-0.997 regardless of behaviour,
against the hexapod's own 0.24-0.99 split). That is not surprising: `z`'s only shaping during B1's
stage-1 adaptation is next-embedding MSE (`wm.adapt`), with no `lambda_body` or cross-embodiment
term anywhere in that loss -- nothing ever asked `z` to be a shared space.

`body_head` is a different object, and the architecture comment at `wm/models/motion_decoder.py`
is explicit about why: it is ONE set of weights for every embodiment (no embodiment key, unlike
the per-embodiment action-decoding heads, which must differ because 12-D and 18-D commands are
different spaces), fit "on the union" of every embodiment's data (`wm/fit_body_head.py`), and
DELIBERATELY blind to the frame -- letting it see the frame was tried (`body_sees_frame`, F57) and
collapsed cross-embodiment transfer from +0.544/+0.435 to -10.5/-57.2, because the frame reveals
which robot this is and the head then learns one silent mapping per robot instead of a shared one.
The comment's own words: "the head still reads vision -- through the bottleneck being shaped,
which is the point." That bottleneck is exactly this file's candidate: the 128-D hidden activation
between `body_head`'s two linear layers (`z_dim=64 -> body_hidden=128 -> body_dim=3`), extracted
BEFORE the final projection collapses it to three Froude scalars -- richer than the final output,
and unlike raw `z`, actually asked (by construction: one function, both bodies' data, blinded to
the appearance shortcut) to be shared.

**What this measures, and how to read it.** Same clips and same methodology as the raw-`z` check,
for a direct before/after: one hidden-activation vector per clip (mean over `body_head`'s own input
`z`s at every valid consecutive frame pair), cosine similarity, cross-embodiment same-behaviour vs
different-behaviour gap. A larger gap than raw `z`'s +0.0803 -- and ideally one that clears each
group's own std, unlike raw `z`'s -- would mean this bottleneck, not raw `z`, is where a
`z_goal`/`z_candidate` cross-embodiment matching mechanism should be built. A gap this small or
smaller means the frame-blinding alone isn't enough either, and the shared-representation question
stays open.

    .venv/bin/python3 scripts/diagnostics/cross_embodiment/body_head_hidden_share.py
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

CKPT = "wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt"
N_PER_BEHAVIOUR = 3  # matches the raw-z check exactly, for a like-for-like comparison

SOURCES = {
    "hexapod": "data/egocentric/beh12_c10f10t10_ego_flat",
    "b1": "data/egocentric/beh12_b1_ego_flat",
}


def hidden_of(body_head, z):
    """`body_head[0:3](z)` -- LayerNorm, Linear(z_dim, body_hidden), GELU. Stops one layer short
    of `body_head[3]`, the final Linear(body_hidden, body_dim) that collapses to 3 Froude scalars.
    Indices come straight from `MotionDecoder`'s own `nn.Sequential` (LayerNorm, Linear, GELU,
    Linear) -- not guessed."""
    return body_head[2](body_head[1](body_head[0](z)))


def clip_repr(itm, body_head, encoder, offset, path, device, chunk=4):
    with np.load(path, allow_pickle=True) as d:
        frames = d["frames"]
    e = encode_clip(encoder, frames, chunk).float().to(device)
    if offset is not None:
        e = e - offset.to(device)
    with torch.no_grad():
        z = itm(e[:-1], e[1:])
        h = hidden_of(body_head, z)
        f = body_head(z)   # the actual final 3-D Froude output -- never tested geometrically before
    return z.mean(0).cpu().numpy(), h.mean(0).cpu().numpy(), f.mean(0).cpu().numpy()


def cross_embodiment_gap(items, sim, label):
    same_beh_cross, diff_beh_cross, same_body_diff_beh, same_body_same_beh = [], [], [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            bi, behi, _ = items[i]
            bj, behj, _ = items[j]
            if bi == bj:
                (same_body_same_beh if behi == behj else same_body_diff_beh).append(sim[i, j])
                continue
            (same_beh_cross if behi == behj else diff_beh_cross).append(sim[i, j])

    def stat(name, vals):
        vals = np.asarray(vals)
        print(f"  {name:42s} n={len(vals):3d}  mean={vals.mean():+.4f}  std={vals.std():.4f}")

    print(f"\n=== {label} ===")
    stat("same-embodiment, SAME behaviour (tightest possible -- positive control)", same_body_same_beh)
    stat("cross-embodiment, SAME behaviour", same_beh_cross)
    stat("cross-embodiment, DIFFERENT behaviour", diff_beh_cross)
    stat("same-embodiment, DIFFERENT behaviour", same_body_diff_beh)
    gap = np.mean(same_beh_cross) - np.mean(diff_beh_cross)
    control_gap = np.mean(same_body_same_beh) - np.mean(same_body_diff_beh)
    print(f"  cross-embodiment gap (same-beh minus diff-beh, across bodies)      = {gap:+.4f}")
    print(f"  within-embodiment gap (same-beh minus diff-beh, positive control)  = {control_gap:+.4f}")
    return gap, control_gap, np.mean(same_body_same_beh)


def pca_and_identity_removal(items, X, label):
    """Two more targeted checks than a raw full-vector cosine gap, per the project's own standing
    caution that a projection (UMAP especially) is an illustration, never evidence on its own
    (`wm_umap.py`'s docstring, before this directory was cleared out): PCA to see how much of the
    variance is explained by a body-separating direction, and a mean-difference identity-removal
    (the same style of intervention `z_identity_ablation.py` used) to test directly whether
    behaviour clustering improves once the dominant body axis is projected out -- rather than
    inferring that from a single number.
    """
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()
    pcs = Xc @ Vt[:3].T
    print(f"\n--- {label}: PCA ---")
    print(f"  variance explained, top 3 PCs: {var_ratio[0]:.1%} {var_ratio[1]:.1%} {var_ratio[2]:.1%}")
    print(f"  {'body':8s}{'behaviour':10s}{'clip':16s}{'PC1':>9}{'PC2':>9}{'PC3':>9}")
    for (b, beh, name), (p1, p2, p3) in zip(items, pcs):
        print(f"  {b:8s}{beh:10s}{name[:14]:16s}{p1:>9.3f}{p2:>9.3f}{p3:>9.3f}")

    bodies = np.array([1.0 if b == "b1" else 0.0 for b, _, _ in items])
    identity_dir = Xc[bodies == 1].mean(0) - Xc[bodies == 0].mean(0)
    identity_dir /= np.linalg.norm(identity_dir) + 1e-9
    proj_len = Xc @ identity_dir
    residual = Xc - np.outer(proj_len, identity_dir)
    body_var = np.var(proj_len)
    total_var = np.var(Xc, axis=0).sum()
    print(f"  the body-mean-difference direction alone explains "
         f"{100 * body_var / total_var:.1f}% of total variance")

    Rn = residual / (np.linalg.norm(residual, axis=1, keepdims=True) + 1e-9)
    sim_r = Rn @ Rn.T
    print(f"\n--- {label}: same behaviour-gap test, AFTER removing the body-mean direction ---")
    return cross_embodiment_gap(items, sim_r, f"{label}, body-identity-direction removed")


def held_out_identity_removal(items, X, label, n_seeds=N_PER_BEHAVIOUR):
    """The rigorous version of `pca_and_identity_removal`'s residual test: the body-mean-difference
    direction is fit on a TRAIN fold and applied to a disjoint TEST fold, never both on the same 18
    clips. `items` is ordered body-major then behaviour-major then seed-index (0..n_seeds-1 within
    each (body, behaviour) cell, exactly how `main()` appends them) -- fold k holds out seed index k
    from every cell as test, fits the direction on the other n_seeds-1 seeds of every cell.
    """
    n_cells = len(items) // n_seeds
    fold_gaps = []
    for k in range(n_seeds):
        test_idx = [c * n_seeds + k for c in range(n_cells)]
        train_idx = [i for i in range(len(items)) if i not in test_idx]
        train_items = [items[i] for i in train_idx]
        test_items = [items[i] for i in test_idx]

        train_mean = X[train_idx].mean(0, keepdims=True)
        Xtr = X[train_idx] - train_mean
        bodies_tr = np.array([1.0 if b == "b1" else 0.0 for b, _, _ in train_items])
        direction = Xtr[bodies_tr == 1].mean(0) - Xtr[bodies_tr == 0].mean(0)
        direction /= np.linalg.norm(direction) + 1e-9

        Xte = X[test_idx] - train_mean               # centred by the TRAIN mean, never the test's own
        proj = Xte @ direction
        residual = Xte - np.outer(proj, direction)
        Rn = residual / (np.linalg.norm(residual, axis=1, keepdims=True) + 1e-9)
        sim = Rn @ Rn.T

        same_beh, diff_beh = [], []
        for i in range(len(test_items)):
            for j in range(i + 1, len(test_items)):
                bi, behi, _ = test_items[i]
                bj, behj, _ = test_items[j]
                if bi == bj:
                    continue
                (same_beh if behi == behj else diff_beh).append(sim[i, j])
        gap = (np.mean(same_beh) - np.mean(diff_beh)) if same_beh and diff_beh else float("nan")
        fold_gaps.append(gap)
        print(f"  fold {k} (test = seed {k} of every cell): held-out gap = {gap:+.4f}  "
             f"(n_same={len(same_beh)}, n_diff={len(diff_beh)})")

    fold_gaps = np.asarray(fold_gaps)
    print(f"  {label}: held-out gap, mean over {n_seeds} folds = {np.nanmean(fold_gaps):+.4f}  "
         f"(fold values: {np.round(fold_gaps, 4).tolist()})")
    return np.nanmean(fold_gaps)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, CKPT), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).eval().to(device)
    itm.load_state_dict(ck["itm"])
    md = MotionDecoder(cfg, {"b1": 12}).eval().to(device)
    md.load_state_dict(ck["md"], strict=False)
    if md.body_head is None:
        raise SystemExit(f"{CKPT} has no body_head (lambda_body was 0) -- nothing to test")
    print(f"body_head: z_dim={cfg.z_dim} -> hidden={cfg.body_hidden} -> body_dim={cfg.body_dim}")
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    items, Z, H, F = [], [], [], []
    for body, data_dir in SOURCES.items():
        offset = offset_for(ck, body)
        by_behaviour = {}
        for f in sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz"))):
            with np.load(f, allow_pickle=True) as d:
                b = str(d["behaviour"])
            by_behaviour.setdefault(b, []).append(f)
        for behaviour, paths in by_behaviour.items():
            for p in paths[:N_PER_BEHAVIOUR]:
                z, h, f = clip_repr(itm, md.body_head, encoder, offset, p, device)
                items.append((body, behaviour, os.path.basename(p)))
                Z.append(z); H.append(h); F.append(f)
                print(f"  encoded {body:8s} {behaviour:6s} {os.path.basename(p)}  "
                     f"froude={np.round(f, 3)}")

    Z, H, F = np.stack(Z), np.stack(H), np.stack(F)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    sim_z = Zn @ Zn.T
    sim_h = Hn @ Hn.T
    sim_f = Fn @ Fn.T

    gap_z, cgap_z, tight_z = cross_embodiment_gap(items, sim_z, "raw z (ITM output, 64-D)")
    gap_h, cgap_h, tight_h = cross_embodiment_gap(items, sim_h, f"body_head hidden layer ({cfg.body_hidden}-D)")
    gap_f, cgap_f, tight_f = cross_embodiment_gap(items, sim_f, "body_head FINAL output (Froude, 3-D)")

    print(f"\n{'':30}{'cross-embod gap':>18}{'within-embod gap':>18}{'tightest (control)':>20}")
    print(f"{'raw z':30}{gap_z:>18.4f}{cgap_z:>18.4f}{tight_z:>20.4f}")
    print(f"{'body_head hidden':30}{gap_h:>18.4f}{cgap_h:>18.4f}{tight_h:>20.4f}")
    print(f"{'body_head Froude output':30}{gap_f:>18.4f}{cgap_f:>18.4f}{tight_f:>20.4f}")

    for label, X in (("raw z", Z), ("body_head hidden", H), ("body_head Froude", F)):
        pca_and_identity_removal(items, X, label)

    print("\n" + "=" * 70)
    print("HELD-OUT TEST: identity direction fit on 2 seeds, applied to the 3rd, never seen together")
    print("=" * 70)
    for label, X in (("raw z", Z), ("body_head hidden", H), ("body_head Froude", F)):
        print(f"\n--- {label} ---")
        held_out_identity_removal(items, X, label)
    print("\nif the within-embodiment gap (positive control) is itself small, the representation "
         "doesn't organise by behaviour even WITHIN one body -- the cross-embodiment result would "
         "then not be a cross-embodiment-specific failure, just a general lack of behaviour "
         "structure at the clip-averaging granularity used here.")
    if gap_h > gap_z + 0.02:
        print("body_head's bottleneck clusters by behaviour across bodies MORE than raw z does --"
             " a real candidate for z_goal/z_candidate, worth the deeper geometric checks before "
             "building anything on it (this alone is not proof, just a screen).")
    else:
        print("no meaningful improvement over raw z -- frame-blinding alone did not make this "
             "bottleneck a shared cross-embodiment space either. The question stays open.")


if __name__ == "__main__":
    main()
