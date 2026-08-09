"""UMAP of the frozen encoder embedding e_t next to the learned latent action z.

Two panels per colouring: e_t is what V-JEPA2 already gives us, z is what the world model
built on top of it. Colouring by body asks whether morphology still organises the space;
colouring by foot-contact pattern asks whether behaviour does.

UMAP is an illustration, never the evidence: it has free parameters, it distorts distance,
and a cluster in it can be an artefact of the projection. The numbers printed under each
panel -- linear-probe accuracy and silhouette -- are computed in the full space and are what
should be quoted. A probe near 1.0 with a silhouette near 0 means the label is present but
not dominant, which a picture cannot show.

Run from the repository root:
  .venv/bin/python3 scripts/wm_umap.py --ckpt wm/runs/<run>/epoch020.pt --clips 8
"""
import argparse
import os
import sys
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import umap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config, from_checkpoint  # noqa: E402
from wm.data.dataset import clip_paths  # noqa: E402
from wm.evaluate import (behaviour_labels, collect, decode_accuracy,  # noqa: E402
                         structure, upgrade_decoder_state)
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

COLOURS = {"long": "#c1121f", "medium": "#f77f00", "short": "#0466c8"}


def project(features, seed):
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=seed)
    return reducer.fit_transform(features)


def panel(ax, points, labels, order, colours, title, subtitle):
    for name in order:
        mask = labels == name
        ax.scatter(points[mask, 0], points[mask, 1], s=6, alpha=0.6,
                   c=colours[name] if isinstance(colours, dict) else colours(name), label=str(name))
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2.2, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", type=int, default=8, help="clips per body")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    known = {f.name for f in fields(Config)}
    cfg = from_checkpoint(checkpoint["config"])
    cfg.train_morphs = tuple(cfg.train_morphs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg).to(device).eval()
    itm.load_state_dict(checkpoint["itm"])
    md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    morphs = tuple(cfg.train_morphs) + (cfg.heldout_morph,)
    paths = [p for body in morphs for p in clip_paths(data_dir, (body,))[:args.clips]]

    encoder = VJEPA2FrameEncoder(device=str(device))
    data = collect(encoder, itm, md, paths, checkpoint["action_mean"], checkpoint["action_std"],
                   device, chunk=args.chunk, seed=args.seed,
                   frame_range=(cfg.frame_start, cfg.frame_stop))
    del encoder
    torch.cuda.empty_cache()

    codes, keep = behaviour_labels(data["contact"])
    stats = {"e": structure(data["e"], data["morph"]), "z": structure(data["z"], data["morph"])}
    gait = {name: decode_accuracy(data[name][keep], codes[keep]) for name in ("e", "z")}
    order = list(morphs)
    epoch = int(checkpoint.get("epoch", -1))
    held = cfg.heldout_morph

    print(f"trained on {list(cfg.train_morphs)}, held out '{held}', epoch {epoch}, "
          f"{len(data['z'])} transitions")
    for name in ("e", "z"):
        print(f"  {name}: body probe {stats[name]['decode']:.4f} (chance {1/len(morphs):.3f}), "
              f"silhouette {stats[name]['silhouette']:.4f}, gait probe {gait[name]:.4f}")

    points = {name: project(data[name], args.seed) for name in ("e", "z")}
    gait_names = sorted(set(codes[keep]))
    palette = plt.get_cmap("tab10")
    gait_colours = {name: palette(i % 10) for i, name in enumerate(gait_names)}

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for col, name in enumerate(("e", "z")):
        label = ("frozen V-JEPA2 embedding e_t" if name == "e"
                 else f"learned latent action z ({cfg.z_dim}-d)")
        panel(axes[0, col], points[name], data["morph"], order, COLOURS, label,
              f"by body -- probe {stats[name]['decode']:.3f}, "
              f"silhouette {stats[name]['silhouette']:.3f}")
        panel(axes[1, col], points[name][keep], codes[keep], gait_names, gait_colours, label,
              f"by foot-contact pattern -- probe {gait[name]:.3f}")

    fig.suptitle(
        f"UMAP before and after the world model -- trained on {', '.join(cfg.train_morphs)}, "
        f"'{held}' held out, epoch {epoch}\n"
        "Illustration only; the printed probe and silhouette are measured in the full space.",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.945))

    out = args.out or os.path.join(
        ROOT, "results", "wm",
        f"umap_{os.path.basename(os.path.dirname(args.ckpt))}_epoch{epoch:03d}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
