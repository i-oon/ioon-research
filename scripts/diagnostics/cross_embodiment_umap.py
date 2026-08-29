"""The picture the source method's shared-latent claim rests on, next to the number it hides.

LAC-WM supports "the latent action space is shared across embodiments" with a UMAP in which the
per-dataset clusters overlap, plus one qualitative rollout. No number is attached to either.

This reproduces that figure for our hexapod and quadruped, and prints underneath each panel the
two quantities that decide what the picture means, both computed in the **full** space:

  probe       how well a linear readout recovers the embodiment from the representation
  silhouette  how separated the two embodiments are, -1 to 1, near 0 meaning intermingled
  share       how much of the variance the embodiment explains (the F38 decomposition)

A probe near 1.0 with a silhouette near 0 is the case a picture cannot show: the label is fully
present and linearly recoverable while the clusters visually overlap. Overlapping is not shared.

UMAP itself is an illustration and never the evidence -- it has free parameters, it distorts
distance, and a cluster in it can be an artefact of the projection.

  .venv/bin/python3 scripts/cross_embodiment_umap.py --ckpt wm/runs/stage2_balanced/best.pt
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.dataset import CONTACT_THRESHOLD  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.bodies import bodies_in  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

# The four bodies `stage2_clean` trains on. The earlier list included `c10f10t06` and
# `c06f10t06`, which veer 0.35-0.40 m off course on a 94.6 mm dead zone, and the runs it was
# written for also trained on two bodies that collapse outright (FINDINGS.md F42). Plotting a
# latent over bodies the model never saw -- or saw only falling over -- describes neither.
INSECT_BODIES = bodies_in(os.path.join(ROOT, "data", "ik_walk_8body"))
INSECT_EPS = [6, 20, 22]
CACHE = f"{ROOT}/results/wm/cache/stage2_embeddings.pt"


def gather(encoder, itm, chunk, bodies, cache_path=CACHE):
    """Pooled encoder embeddings, latents, embodiment labels and stance fraction."""
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    fresh = False
    E, Z, emb, stance = [], [], [], []
    clips = [("hexapod", f"{ROOT}/data/fwd_hex8body/{b}_ep{e}.npz")
             for b in bodies for e in INSECT_EPS]
    clips += [("b1", p) for p in sorted(glob.glob(f"{ROOT}/data/fwd_b1_50hz/*.npz"))]
    for name, path in clips:
        clip = np.load(path, allow_pickle=True)
        if path not in cache:
            # the encoder may sit on the GPU; everything downstream is small and stays on the CPU
            cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu()
            fresh = True
        e = cache[path]
        n = len(e) - 1
        with torch.no_grad():
            z = torch.cat([itm(e[s:min(s + 8, n)], e[s + 1:min(s + 8, n) + 1])
                           for s in range(0, n, 8)]).numpy()
        contact = ((clip["forces"] > CONTACT_THRESHOLD).mean(axis=1) if name == "hexapod"
                   else clip["foot_contact"].mean(axis=1))
        E.append(e[:n].mean(1).numpy()); Z.append(z)
        emb += [name] * n; stance.append(contact[:n])
    if fresh:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return np.concatenate(E), np.concatenate(Z), np.array(emb), np.concatenate(stance)


def numbers(X, label):
    probe = cross_val_score(LogisticRegression(max_iter=3000), X,
                            (label == "b1").astype(int), cv=5).mean()
    sil = silhouette_score(X, (label == "b1").astype(int))
    means = {n: X[label == n].mean(0) for n in np.unique(label)}
    between = np.linalg.norm(means["hexapod"] - means["b1"])
    within = np.mean([np.linalg.norm(X[label == n] - means[n], axis=1).mean() for n in means])
    return probe, sil, between / within


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--encode_device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bodies", nargs="+", default=INSECT_BODIES,
                    help="hexapod bodies to plot; default is what stage2_clean trains on")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "wm", "figures",
                                                  "cross_embodiment_umap.png"))
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    itm = InverseTransitionModel(from_checkpoint(checkpoint["config"])).eval()
    itm.load_state_dict(checkpoint["itm"])
    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    E, Z, emb, stance = gather(encoder, itm, args.chunk, args.bodies)
    del encoder
    print(f"{len(E)} frames: {(emb=='hexapod').sum()} hexapod, {(emb=='b1').sum()} b1", flush=True)

    import umap
    fig, ax = plt.subplots(2, 2, figsize=(11, 9.5))
    phase = np.digitize(stance, np.quantile(stance, [.2, .4, .6, .8]))

    for col, (X, name) in enumerate(((E, "frozen encoder  e_t"), (Z, "learned latent  z"))):
        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=args.seed).fit_transform(X)
        probe, sil, sep = numbers(X, emb)
        for row, (colour, title) in enumerate(((emb, "coloured by embodiment"),
                                               (phase, "coloured by gait phase"))):
            p = ax[row, col]
            if row == 0:
                for n, c in (("hexapod", "#2471a3"), ("b1", "#c0392b")):
                    m = emb == n
                    p.scatter(xy[m, 0], xy[m, 1], s=5, alpha=.55, c=c, label=n)
                p.legend(fontsize=9, markerscale=2.5)
            else:
                p.scatter(xy[:, 0], xy[:, 1], s=5, alpha=.55, c=colour, cmap="viridis")
            p.set_xticks([]); p.set_yticks([])
            p.set_title(f"{name}\n{title}", fontsize=11)
            if row == 1:
                p.set_xlabel(f"in the FULL space:  embodiment probe {probe:.3f}   "
                             f"silhouette {sil:+.3f}\nclusters {sep:.2f}x the within-cluster spread apart",
                             fontsize=9)
        print(f"{name:<22} probe {probe:.3f}  silhouette {sil:+.3f}  separation {sep:.2f}x")

    fig.suptitle("Overlapping is not shared: the picture, and the numbers underneath it",
                 fontsize=12.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
