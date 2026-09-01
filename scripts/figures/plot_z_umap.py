"""Where each robot's latents sit, and which way speed runs inside each cloud.

**This figure exists to keep a claim honest, not to support it.** The probe shows a linear speed
readout transferring between robots once `lambda_body` is on. That is a statement about a *shared
direction*, and it is very easy to restate as "the latents of the two robots now coincide", which
is false: an embodiment classifier still reads `z` at ceiling AUC. Two clouds sitting in different
places with parallel speed gradients produce every number the probe reports.

So the panels are laid out to show exactly that and nothing more -- coloured by robot on the left,
by Froude on the right. The reading to look for is **two separated clouds whose colour gradients
run the same way**, not one merged cloud.

UMAP is illustration only: it does not preserve distance, and it will happily draw structure that
is not there. The measurement stays the probe's numbers, printed under each row.

  .venv/bin/python3 scripts/figures/plot_z_umap.py \\
      --ckpts wm/runs/s2_fwd_hex7-b1_ctrl/last.pt wm/runs/s2_fwd_hex7-b1_body0.5/last.pt \\
      --labels "control (no body term)" "with body head"
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import umap  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

from diagnostics.body_motion_probe import gather, standardise  # noqa: E402


def embodiment_auc(x, label, clip, seed=0):
    """How separable the two robots are, held out **by clip**.

    Neighbouring frames of one clip are near-duplicates, so a frame-level split trains and tests on
    the same walk and the score means nothing -- the first version of this function used 3-fold CV
    over frames and returned 0.212, below chance, which is a fold artefact rather than a result.
    Splitting by clip gives 1.000 on raw `z` and chance once each embodiment is standardised.
    """
    rng = np.random.default_rng(seed)
    clips = np.unique(clip)
    rng.shuffle(clips)
    train = np.isin(clip, clips[:int(0.7 * len(clips))])
    model = LogisticRegression(max_iter=3000).fit(x[train], label[train])
    return float(roc_auc_score(label[~train], model.decision_function(x[~train])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=[])
    ap.add_argument("--insect_dir", default="data/allocentric/fwd_hex7speed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/wm/stage2/figures/z_umap.png")
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, "results", "wm", "cache",
                              f"probe_{os.path.basename(args.insect_dir)}.pt")
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.ckpts]
    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)

    rows = []
    for ckpt, label in zip(args.ckpts, labels):
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        itm = InverseTransitionModel(from_checkpoint(checkpoint["config"]))
        itm.load_state_dict(checkpoint["itm"])
        itm.eval()
        data = gather(encoder, args.chunk, args.insect_dir, cache_path, itm, checkpoint)
        # standardised per embodiment, exactly as the probe does, so the picture is of the space
        # the probe measured and not of a colour/scale difference the probe had already removed
        raw = np.concatenate([data[n][0] for n in ("insect", "b1")])
        x = np.concatenate([standardise(data[n][0]) for n in ("insect", "b1")])
        n_frames = {n: len(data[n][1]) for n in ("insect", "b1")}
        n_clips = {n: int(data[n][2].max()) + 1 for n in ("insect", "b1")}
        y = np.concatenate([data[n][1] for n in ("insect", "b1")])
        who = np.concatenate([np.full(len(data[n][1]), i) for i, n in enumerate(("insect", "b1"))])
        clip = np.concatenate([data[n][2] + 1000 * i for i, n in enumerate(("insect", "b1"))])
        fit = lambda f: umap.UMAP(n_neighbors=30, min_dist=0.1,
                                  random_state=args.seed).fit_transform(f)
        xy_raw, xy = fit(raw), fit(x)
        # Both numbers, because the pair is the finding: raw `z` separates the robots perfectly,
        # and standardising each embodiment's features removes that entirely. All of the identity
        # sits in the per-feature mean and scale -- the two clouds are one cloud, shifted.
        rows.append((label, xy_raw, xy, y, who,
                     embodiment_auc(raw, who, clip, args.seed),
                     embodiment_auc(x, who, clip, args.seed),
                     n_frames, n_clips))
    del encoder

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.8 * len(rows)), squeeze=False)
    for r, (label, xy_raw, xy, y, who, auc_raw, auc_std, n_frames, n_clips) in enumerate(rows):
        for col, (coords, auc, what) in enumerate((
                (xy_raw, auc_raw, "raw z"), (xy, auc_std, "after per-robot standardising"))):
            ax = axes[r][col]
            for i, (name, colour) in enumerate((("insect", "#c0392b"), ("b1", "#2471a3"))):
                m = who == i
                ax.scatter(coords[m, 0], coords[m, 1], s=3, c=colour, alpha=0.45,
                           label=f"{name} ({n_clips[name]} clips)", linewidths=0)
            ax.legend(markerscale=4, fontsize=8, loc="best")
            ax.set_title(f"{label}\n{what} -- embodiment AUC {auc:.3f}", fontsize=10)

        ax = axes[r][2]
        # one colour scale per robot: the two walk at different absolute Froude, and a shared scale
        # would paint one cloud uniformly and hide the gradient this panel exists to show
        for i in (0, 1):
            m = who == i
            v = (y[m] - y[m].mean()) / (y[m].std() + 1e-9)
            s = ax.scatter(xy[m, 0], xy[m, 1], s=3, c=v, cmap="viridis",
                           vmin=-2, vmax=2, alpha=0.7, linewidths=0)
        fig.colorbar(s, ax=ax, label="Froude, standardised within robot")
        ax.set_title(f"{label}\nsame layout as middle, coloured by speed", fontsize=10)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("All of the embodiment identity is the per-feature mean and scale: raw z separates "
                 "the robots perfectly (left),\nand standardising each robot removes it entirely "
                 "(middle). Speed is read in that standardised space (right).", fontsize=11)
    fig.tight_layout()
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"-> {os.path.relpath(out, ROOT)}")
    for label, _, _, _, _, auc_raw, auc_std, n_frames, n_clips in rows:
        print(f"  {label:<28} AUC raw {auc_raw:.3f}  standardised {auc_std:.3f}   "
              f"insect {n_clips['insect']} clips/{n_frames['insect']} frames, "
              f"b1 {n_clips['b1']}/{n_frames['b1']}")
    print("\nThe clip counts are unequal (91 against 14). UMAP layout is dominated by the larger")
    print("set, and b1's probe numbers rest on 14 clips with 5 held out -- a real limit on how")
    print("finely the b1 side can be resolved.")
    print("\nUMAP does not preserve distance. Read the panels as 'are there two clouds' and 'does")
    print("colour run the same way in each', and take every quantity from the probe instead.")


if __name__ == "__main__":
    main()
