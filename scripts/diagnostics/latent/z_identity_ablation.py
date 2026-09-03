"""Is the embodiment identity in the latent load-bearing, or is it passive leakage?

F38 measured that 33.0% of the latent's variance is "which embodiment is this", and that a linear
probe recovers the embodiment from `z` at 1.000. Both say identity is *present*. Neither says
anything is *using* it, and that difference decides what the fix is:

  load-bearing   something downstream reads identity out of `z`. Give it a clean route -- an
                 embodiment side channel into the FTM -- so `z` no longer has to carry it.
  passive        identity leaked in from the frozen encoder, because a hexapod and a quadruped
                 look different and no loss ever penalised the difference surviving into `z`. A
                 side channel then changes nothing, since nothing was pulling on `z` to begin
                 with, and the fix is an adversary that removes the ability rather than the need.

The prior leans passive. The decoder's output head is *selected* by embodiment, so identity is
handed to it for free, and the FTM sees `x_t`, which is a picture of the robot. Neither has to ask
`z`.

The test needs no training. Identity is linearly decodable, so it occupies a low-dimensional
subspace of the 64-D latent; find that subspace, project it out, and re-score the motion decoder
with the crippled latent. Directions are removed one at a time, refitting the probe after each,
until the embodiment is no longer recoverable -- one direction rarely suffices, since the probe
can route around a single deleted axis.

Two controls, because "removing a direction costs accuracy" is true of *any* direction:

  random     project out the same number of random orthogonal directions, averaged over seeds.
             This is the floor: the cost of losing capacity, with nothing meaningful removed.
  zero_z     the whole latent zeroed, the ceiling from F26's ablation.

Read the identity row against the random row, not against zero. If identity costs no more than
random, it was passive and the side channel is the wrong intervention.

  .venv/bin/python3 scripts/z_identity_ablation.py --ckpt wm/runs/stage2_balanced/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for, upgrade_decoder_state  # noqa: E402
from wm.bodies import bodies_in  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

# The four bodies `stage2_clean` trains on. The earlier list included `c10f10t06` and
# `c06f10t06`, which that run holds out and which veer 0.35-0.40 m off course anyway -- scoring
# an ablation partly on bodies the model never saw reads 15.99 deg where the trained ones read
# under 4, and the ratio being measured is then a mix of two different questions.
INSECT_BODIES = bodies_in(os.path.join(ROOT, "data", "ik_walk_8body"))
INSECT_EPS = [6, 20, 22]


def clip_list(insect_dir, b1_dir):
    out = [("hexapod", f"{insect_dir}/{b}_ep{e}.npz")
           for b in INSECT_BODIES for e in INSECT_EPS]
    out += [("b1", p) for p in sorted(glob.glob(f"{b1_dir}/*.npz"))]
    return [(n, p) for n, p in out if os.path.exists(p)]


@torch.no_grad()
def embed(encoder, clips, chunk, cache_path):
    """Encoder embeddings per clip, cached: the V-JEPA2 pass dominates the runtime and does not
    depend on the checkpoint being scored, so it is paid once across reruns."""
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    missing = [(n, p) for n, p in clips if p not in cache]
    for i, (name, path) in enumerate(missing, 1):
        clip = load(path, REGISTRY[name])
        # .cpu() matters: the encoder may run on the GPU while the ITM and decoder stay on the
        # CPU, and a cached GPU tensor then fails at the first Linear rather than at the call site
        cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu()
        print(f"  encoded {i}/{len(missing)}  {os.path.basename(path)}", flush=True)
    if missing:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return cache


@torch.no_grad()
def gather(embeddings, itm, clips, stats, action_lag, checkpoint=None):
    """Per clip: embeddings, latents, standardised target commands, embodiment name."""
    out = []
    for name, path in clips:
        clip = load(path, REGISTRY[name])
        e = embeddings[path]
        # the cache holds raw encoder output, so a centred checkpoint's offset is applied here
        # rather than baked into the cache, which is shared across checkpoints
        offset = offset_for(checkpoint, name) if checkpoint else None
        if offset is not None:
            e = e - offset
        actions = clip["actions"]
        n = min(len(e) - 1, len(actions) - action_lag)
        z = torch.cat([itm(e[s:min(s + 8, n)], e[s + 1:min(s + 8, n) + 1])
                       for s in range(0, n, 8)])
        mean, std = (np.asarray(v, dtype=np.float32) for v in stats[name])
        target = (actions[action_lag:action_lag + n] - mean) / std
        out.append({"embodiment": name, "e": e[:n], "z": z,
                    "target": torch.tensor(target), "std": std})
    return out


def identity_basis(Z, label, max_dims, chance=0.5, tol=0.02):
    """Orthonormal directions carrying the embodiment, peeled off one at a time.

    A single logistic direction is not enough: with it deleted the probe refits onto correlated
    axes and often recovers most of its accuracy. Directions are added until the probe falls to
    chance, so what gets removed is the identity *subspace* rather than one convenient axis.
    """
    work, basis, trace = Z.copy(), [], []
    for _ in range(max_dims):
        acc = cross_val_score(LogisticRegression(max_iter=3000), work, label, cv=5).mean()
        trace.append(acc)
        if acc <= chance + tol:
            break
        w = LogisticRegression(max_iter=3000).fit(work, label).coef_[0]
        for b in basis:                                   # keep the basis orthonormal
            w -= (w @ b) * b
        norm = np.linalg.norm(w)
        if norm < 1e-8:
            break
        w /= norm
        basis.append(w)
        work = work - np.outer(work @ w, w)
    residual = cross_val_score(LogisticRegression(max_iter=3000), work, label, cv=5).mean()
    return np.stack(basis) if basis else np.zeros((0, Z.shape[1])), trace, residual


def project_out(z, basis):
    if len(basis) == 0:
        return z
    B = torch.tensor(basis, dtype=z.dtype)
    return z - (z @ B.T) @ B


@torch.no_grad()
def motion_error(md, records, transform, chunk=8):
    """RMSE per joint in degrees, per embodiment, with `transform` applied to every latent."""
    sq, count = {}, {}
    for rec in records:
        name, std = rec["embodiment"], torch.tensor(rec["std"])
        for s in range(0, len(rec["e"]), chunk):
            e = rec["e"][s:s + chunk]
            z = transform(rec["z"][s:s + chunk])
            pred = md(e, z, name)
            # de-standardise before converting: MSE in standardised units is not comparable
            # across embodiments, since each has its own per-joint spread
            err = (pred - rec["target"][s:s + chunk]) * std
            sq[name] = sq.get(name, 0.0) + float((err ** 2).sum())
            count[name] = count.get(name, 0) + err.numel()
    return {k: float(np.rad2deg(np.sqrt(sq[k] / count[k]))) for k in sq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--insect_dir", default=os.path.join(ROOT, "data", "ik_walk_8body"))
    ap.add_argument("--b1_dir", default=os.path.join(ROOT, "data", "b1_framed"))
    ap.add_argument("--max_dims", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5, help="random-control repeats")
    ap.add_argument("--encode_device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "wm", "cache",
                                                    "stage2_embeddings.pt"))
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    stats = checkpoint["action_stats"]

    itm = InverseTransitionModel(cfg).eval()
    itm.load_state_dict(checkpoint["itm"])
    md = MotionDecoder(cfg, heads={k: len(v[0]) for k, v in stats.items()}).eval()
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))

    clips = clip_list(args.insect_dir, args.b1_dir)
    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    embeddings = embed(encoder, clips, args.chunk, args.cache)
    del encoder
    records = gather(embeddings, itm, clips, stats, cfg.action_lag, checkpoint)

    Z = torch.cat([r["z"] for r in records]).numpy()
    label = np.concatenate([[r["embodiment"] == "b1"] * len(r["z"]) for r in records]).astype(int)
    print(f"{args.ckpt}  epoch {checkpoint.get('epoch', -1)}  action_lag {cfg.action_lag}")
    print(f"{len(Z)} latents of dim {Z.shape[1]}: "
          f"{int((label == 0).sum())} hexapod, {int((label == 1).sum())} b1\n")

    basis, trace, residual = identity_basis(Z, label, args.max_dims)
    print(f"embodiment probe as directions are peeled off: "
          f"{' -> '.join(f'{a:.3f}' for a in trace)}")
    print(f"{len(basis)} of {Z.shape[1]} directions carry the embodiment; with them removed the "
          f"probe reads {residual:.3f} against a chance level of 0.500\n")

    rows = [("intact", motion_error(md, records, lambda z: z))]
    rows.append((f"identity removed ({len(basis)}d)",
                 motion_error(md, records, lambda z: project_out(z, basis))))

    if len(basis):
        per_seed = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            R = np.linalg.qr(rng.standard_normal((Z.shape[1], len(basis))))[0].T
            per_seed.append(motion_error(md, records, lambda z: project_out(z, R)))
        rows.append((f"random {len(basis)}d removed",
                     {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0]}))

    rows.append(("z zeroed", motion_error(md, records, torch.zeros_like)))

    names = sorted(rows[0][1])
    width = max(len(r[0]) for r in rows) + 2
    print(f'{"latent":<{width}}' + "".join(f"{n:>12}" for n in names) + f'{"":>4}vs intact')
    intact = rows[0][1]
    for label_text, errors in rows:
        ratio = np.mean([errors[n] / intact[n] for n in names])
        print(f"{label_text:<{width}}" + "".join(f"{errors[n]:>11.2f}u" for n in names)
              + f"{ratio:>12.2f}x")
    print("\nErrors are RMSE per joint in degrees. Read the identity row against the random row:\n"
          "if deleting the embodiment subspace costs no more than deleting the same number of\n"
          "arbitrary directions, nothing downstream was using it, the 33% is leakage from the\n"
          "frozen encoder, and an FTM side channel has no pressure to relieve.")


if __name__ == "__main__":
    main()
