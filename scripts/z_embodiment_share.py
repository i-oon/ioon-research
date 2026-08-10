"""How much of the latent is "which embodiment is this" rather than "what movement is happening"?

The source method supports its shared-latent-space claim with overlapping UMAP clusters and one
qualitative rollout. Neither is a number. This produces one, for a hexapod and a quadruped trained
into a single model.

It is the measurement `scripts/z_body_share.py` makes across insect bodies, with two differences
forced by the setting:

  - Insect bodies all walk the same expert episodes, so that script builds a balanced body-by-phase
    grid and reads the between-body term straight off it. A hexapod and a B1 share no episodes and
    have different clip lengths, so no such grid exists.
  - Instead, phase is estimated from the data itself: stance fraction, the proportion of feet on
    the ground, which is defined in [0, 1] whether there are six feet or four, is recorded on both
    sides, and cycles within every clip. Binning it gives a phase label both embodiments share.

That makes the decomposition a two-way split of the latent's variance into an embodiment term, a
phase term, and the interaction, on a grid of embodiment by phase bin.

Also reported, because the decomposition alone can mislead: how well a linear probe recovers the
embodiment from the latent, and how far apart the two clusters sit relative to the spread within
one. A latent can hold embodiment identity at high probe accuracy while it explains almost none of
the variance -- presence and dominance are different questions, and F26 showed they come apart.

  .venv/bin/python3 scripts/z_embodiment_share.py --ckpt wm/runs/stage2_balanced/epoch012.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import CONTACT_THRESHOLD  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

INSECT_BODIES = ["c10f10t10", "c06f10t10", "c10f10t06", "c06f10t06", "c10f06t06"]
INSECT_EPS = [6, 20, 22]


def clips(insect_dir, b1_dir):
    """(embodiment, frames, stance fraction) per clip, for both sides."""
    out = []
    for body in INSECT_BODIES:
        for ep in INSECT_EPS:
            path = f"{insect_dir}/{body}_ep{ep}.npz"
            if not os.path.exists(path):
                continue
            clip = np.load(path)
            out.append(("hexapod", clip["frames"],
                        (clip["forces"] > CONTACT_THRESHOLD).mean(axis=1)))
    for path in sorted(glob.glob(f"{b1_dir}/*.npz")):
        clip = np.load(path, allow_pickle=True)
        out.append(("b1", clip["frames"], clip["foot_contact"].mean(axis=1)))
    return out


@torch.no_grad()
def latents(itm, embeddings, chunk=8):
    n = len(embeddings) - 1
    return torch.cat([itm(embeddings[s:min(s + chunk, n)],
                          embeddings[s + 1:min(s + chunk, n) + 1])
                      for s in range(0, n, chunk)]).numpy()


def two_way(values, row_label, col_label):
    """Variance of `values` split into a row term, a column term and the interaction.

    Cells are balanced by subsampling to the smallest occupied cell, because an unbalanced grid
    lets a term inherit variance from cell sizes rather than from the factor it names.
    """
    rows, cols = np.unique(row_label), np.unique(col_label)
    cells = {(r, c): np.where((row_label == r) & (col_label == c))[0] for r in rows for c in cols}
    smallest = min(len(v) for v in cells.values())
    if smallest == 0:
        raise SystemExit("some embodiment/phase cell is empty; use fewer phase bins")
    rng = np.random.default_rng(0)
    grid = np.stack([np.stack([values[rng.choice(cells[(r, c)], smallest, replace=False)].mean(0)
                               for c in cols]) for r in rows])          # rows x cols x dim
    centred = grid - grid.reshape(-1, grid.shape[-1]).mean(0)
    row_term = (centred.mean(1) ** 2).sum() * len(cols)
    col_term = (centred.mean(0) ** 2).sum() * len(rows)
    rest = ((centred - centred.mean(1)[:, None] - centred.mean(0)[None]) ** 2).sum()
    total = row_term + col_term + rest
    return 100 * col_term / total, 100 * row_term / total, 100 * rest / total, smallest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--insect_dir", default=os.path.join(ROOT, "data", "ik_walk_8body"))
    ap.add_argument("--b1_dir", default=os.path.join(ROOT, "data", "b1_framed"))
    ap.add_argument("--bins", type=int, default=6, help="phase bins from stance fraction")
    ap.add_argument("--encode_device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    itm = InverseTransitionModel(cfg).eval()
    itm.load_state_dict(checkpoint["itm"])

    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    Z, emb, stance = [], [], []
    for embodiment, frames, contact in clips(args.insect_dir, args.b1_dir):
        e = encode_clip(encoder, frames, args.chunk)
        z = latents(itm, e)
        Z.append(z); emb += [embodiment] * len(z); stance.append(contact[:len(z)])
    del encoder
    Z = np.concatenate(Z); emb = np.array(emb); stance = np.concatenate(stance)

    # a shared phase label: which band of stance fraction this frame falls in
    edges = np.quantile(stance, np.linspace(0, 1, args.bins + 1)[1:-1])
    phase = np.digitize(stance, edges)

    print(f"{args.ckpt}  epoch {checkpoint.get('epoch', -1)}")
    for name in np.unique(emb):
        print(f"  {name:<8} {int((emb == name).sum())} latents, stance fraction "
              f"{stance[emb == name].mean():.3f}")

    gait, embodiment_share, rest, per_cell = two_way(Z, emb, phase)
    print(f"\nvariance of the latent, {args.bins} phase bins, {per_cell} latents per cell")
    print(f"  gait phase        {gait:5.1f}%")
    print(f"  which embodiment  {embodiment_share:5.1f}%")
    print(f"  interaction       {rest:5.1f}%")

    acc = cross_val_score(LogisticRegression(max_iter=3000), Z,
                          (emb == "b1").astype(int), cv=5).mean()
    means = {n: Z[emb == n].mean(0) for n in np.unique(emb)}
    between = np.linalg.norm(means["hexapod"] - means["b1"])
    within = np.mean([np.linalg.norm(Z[emb == n] - means[n], axis=1).mean() for n in means])
    print(f"\n  embodiment decodable from the latent at {acc:.3f} (chance 0.500)")
    print(f"  clusters sit {between / within:.2f}x the within-cluster spread apart")
    print("\nCompare against the insect-only figures: with the cross-body loss the *body* share is "
          "0.8-1.2%\non training bodies. A large embodiment share here means the latent is a code "
          "for which robot\nthis is, which is what Stage 2 needs it not to be.")


if __name__ == "__main__":
    main()
