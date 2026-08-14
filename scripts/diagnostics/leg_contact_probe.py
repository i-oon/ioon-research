"""Does the frozen encoder describe *a leg being loaded* the same way on a hexapod and a quadruped?

This replaces the stance-fraction probe, which turned out to be badly conditioned. Stance fraction
is the proportion of feet on the ground, and on the B1 it barely moves -- spread 0.064 against the
insect's 0.143, because the B1 is an RL policy holding a commanded velocity while the insect is a
real animal's irregular wave. A readout cannot beat the mean when the target hardly leaves it, so
the B1 diagonal read 0.89 and the whole 2x2 was being judged against a ceiling that was not there.

Two other candidates fail for the same reason. Body height varies under 1% on the B1. A shared
gait-phase angle needs periodicity, and the insect's contact signal autocorrelates at only
0.13-0.24 at its own period against the B1's 0.39-0.56 -- it is not periodic enough to have one.

**Per-leg contact is balanced by construction.** Each leg is down about half the time on both
robots, so the target carries full variance and chance is 50%. And the four corner legs
correspond anatomically:

    left front   hexapod FL (duty 0.31)   b1 FL (0.57)
    left hind    hexapod HL (0.59)        b1 RL (0.50)
    right front  hexapod FR (0.51)        b1 FR (0.58)
    right hind   hexapod HR (0.52)        b1 RR (0.49)

The insect's middle legs have no counterpart. That is a real asymmetry between the bodies, not a
defect in the measurement, and it is why only four legs are scored.

Reported as balanced accuracy so an unbalanced duty cannot flatter the result, with the
within-embodiment diagonal as the ceiling each cross cell is read against. Features follow F41:
band-pooled patch tokens with each embodiment standardised by its own statistics, which removes
the colour and apparent-size difference without touching the target.

  .venv/bin/python3 scripts/leg_contact_probe.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import CONTACT_THRESHOLD  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.bodies import bodies_in  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

CACHE = os.path.join(ROOT, "results", "wm", "cache", "stage2_embeddings.pt")
# the four bodies stage2_clean trains on: every one walks straight
INSECT_BODIES = bodies_in(os.path.join(ROOT, "data", "ik_walk_8body"))
INSECT_EPS = [6, 20, 22]

# hexapod foot_order is FL ML HL FR MR HR; the B1's is FR FL RR RL
LEGS = [("left front", 0, 1), ("left hind", 2, 3), ("right front", 3, 0), ("right hind", 5, 2)]


def bands(tokens, grid=16, n=4):
    """Average within each of four horizontal bands of the patch grid.

    Averaging all 256 patches buries a quantity living in the few near the feet, and preserves a
    constant offset between the two datasets that a fitted readout absorbs and mis-applies (F41).
    """
    t = tokens.reshape(len(tokens), grid, grid, -1)
    return t.reshape(len(tokens), n, grid // n * grid, -1).mean(2).flatten(1).numpy()


def gather(encoder, chunk, itm=None, checkpoint=None):
    """Per embodiment: features, a binary label per corresponding leg, and clip ids."""
    cache = torch.load(CACHE, map_location="cpu") if os.path.exists(CACHE) else {}
    fresh = False
    out = {}
    for name in ("insect", "b1"):
        if name == "insect":
            paths = [f"{ROOT}/data/ik_walk_8body/{b}_ep{e}.npz"
                     for b in INSECT_BODIES for e in INSECT_EPS]
        else:
            paths = sorted(glob.glob(f"{ROOT}/data/b1_framed/*.npz"))
        feats, labels, clip_id = [], [], []
        for i, path in enumerate(paths):
            if not os.path.exists(path):
                continue
            clip = np.load(path, allow_pickle=True)
            if path not in cache:
                cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu()
                fresh = True
            contact = ((clip["forces"] > CONTACT_THRESHOLD) if name == "insect"
                       else clip["foot_contact"] > 0.5).astype(int)
            if itm is None:
                feats.append(bands(cache[path]))
            else:
                # the learned latent instead of the frozen encoder: does training make the two
                # embodiments *more* comparable than V-JEPA2 left them?
                e = cache[path]
                off = offset_for(checkpoint, "hexapod" if name == "insect" else "b1")
                if off is not None:
                    e = e - off
                n = len(e) - 1
                with torch.no_grad():
                    z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                                   for t in range(0, n, 8)]).numpy()
                feats.append(z)
                contact = contact[:len(z)]
            labels.append(contact)
            clip_id.append(np.full(len(contact), i))
        out[name] = (np.concatenate(feats), np.concatenate(labels), np.concatenate(clip_id))
    if fresh:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        torch.save(cache, CACHE)
    return out


def standardise(x):
    """Each embodiment centred and scaled by its own statistics: removes the colour and
    apparent-size difference using only which dataset a frame came from, never the label."""
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def split_by_clip(n_clips, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_clips)
    return set(order[:int(frac * n_clips)].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--raw", action="store_true", help="skip per-embodiment standardisation")
    ap.add_argument("--ckpt", default="",
                    help="probe the learned latent z from this checkpoint instead of e_t")
    args = ap.parse_args()

    itm = checkpoint = None
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        itm = InverseTransitionModel(from_checkpoint(checkpoint["config"])).eval()
        itm.load_state_dict(checkpoint["itm"])
        print(f"probing the learned latent z from {args.ckpt}")
    else:
        print("probing the frozen encoder e_t")
    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    data = gather(encoder, args.chunk, itm, checkpoint)
    del encoder

    for name, (x, y, _) in data.items():
        print(f"{name:8} {len(x)} frames, {x.shape[1]} features, "
              f"per-leg duty {np.round(y.mean(0), 3)}")
    if not args.raw:
        data = {k: (standardise(v[0]), v[1], v[2]) for k, v in data.items()}
        print("per-embodiment standardised")
    print()

    rows = {}
    for leg, hex_i, b1_i in LEGS:
        idx = {"insect": hex_i, "b1": b1_i}
        cell = {}
        for src in ("insect", "b1"):
            xs, ys, ids = data[src]
            ys = ys[:, idx[src]]
            # split by clip, so neighbouring frames of one clip cannot sit on both sides
            train_clips = split_by_clip(ids.max() + 1)
            tr = np.isin(ids, list(train_clips))
            # RidgeClassifier, not LogisticRegression: with 5,632 band-pooled features
            # against ~1,900 samples, lbfgs grinds for hours without converging. Ridge
            # has a closed form and picks its own penalty by cross-validation, which
            # matters when n_features exceeds n_samples.
            model = RidgeClassifierCV(alphas=np.logspace(-1, 4, 12)).fit(xs[tr], ys[tr])
            for dst in ("insect", "b1"):
                xd, yd, _ = data[dst]
                yd = yd[:, idx[dst]]
                xq, yq = (xs[~tr], ys[~tr]) if dst == src else (xd, yd)
                cell[(src, dst)] = balanced_accuracy_score(yq, model.predict(xq))
        rows[leg] = cell

    print(f'{"leg":13}{"in->in":>9}{"b1->b1":>9}{"in->b1":>9}{"b1->in":>9}')
    for leg, cell in rows.items():
        print(f'{leg:13}{cell[("insect","insect")]:9.3f}{cell[("b1","b1")]:9.3f}'
              f'{cell[("insect","b1")]:9.3f}{cell[("b1","insect")]:9.3f}')
    mean = {k: np.mean([c[k] for c in rows.values()]) for k in list(rows.values())[0]}
    print(f'{"mean":13}{mean[("insect","insect")]:9.3f}{mean[("b1","b1")]:9.3f}'
          f'{mean[("insect","b1")]:9.3f}{mean[("b1","insect")]:9.3f}')

    print("\nBalanced accuracy, chance 0.500. The diagonal is the ceiling: it says how well this\n"
          "quantity is readable at all. A cross cell near its own diagonal means the encoder\n"
          "describes a loaded leg the same way for both robots; a cross cell near 0.500 means it\n"
          "does not, and unlike the stance-fraction probe the target here carries full variance\n"
          "on both sides, so a weak result cannot be blamed on nothing being there to predict.")


if __name__ == "__main__":
    main()
