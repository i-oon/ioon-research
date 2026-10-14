"""Every world-model loss term of trained checkpoints on the SAME evaluation sets (Ajan Blink week 20: show losses before /
after adding the branch data, not only correlations). Uses the training code's own validation pass (`wm.train.run_epoch`
without an optimiser), each checkpoint's own config and training sources (so normalisation statistics are the ones it
was trained with); only the evaluated pairs differ:
  clips_val          c10 + B1 `*_clips_val` (the common checkpoint-selection set; branch-free)
  branches_heldout   a fixed 10 % sample of c10 + B1 `*_branches_heldout` (never trained on; other actions from one state)
Per-epoch training curves exist only for later runs (step logging); this is the before / after comparison by measurement.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/loss_before_after.py \\
        --ckpt "no branches=wm/runs/round1_clips_s0/best.pt" --ckpt "with branches=wm/runs/round1_branches_s0/best.pt"
    -> results/eval/loss_before_after.txt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
CW = "data/counterfactual_walks"
SAMPLE = os.path.join(ROOT, "results/eval/_loss_eval_sample")


def branch_sample(frac=0.10):
    """fixed, sorted every-10th sample of the held-out branches as symlinks (identical for every checkpoint)"""
    out = {}
    for body in ("c10", "b1"):
        d = os.path.join(SAMPLE, f"{body}_branches_heldout")
        if not os.path.isdir(d):
            os.makedirs(d)
            fs = sorted(glob.glob(os.path.join(ROOT, CW, f"{body}_branches_heldout", "*.npz")))
            for f in fs[::int(round(1 / frac))]:
                os.symlink(f, os.path.join(d, os.path.basename(f)))
        out[body] = os.path.relpath(d, ROOT)
    return out


@torch.no_grad()
def losses(ckpt, val_sources, device):
    from torch.utils.data import DataLoader
    from wm.config import from_checkpoint
    from wm.train import build_cross_embodiment, build_models, run_epoch
    from wm.data.dataset import EmbodimentBatchSampler
    from wm.data import augment
    from vjepa2_encoder import VJEPA2FrameEncoder
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])                  # sets VJEPA_GRAY for a grayscale checkpoint
    cfg.val_sources = list(val_sources)
    augment.configure(cfg)
    train_set, val_set, heads = build_cross_embodiment(cfg, ROOT)
    loader = DataLoader(val_set, batch_sampler=EmbodimentBatchSampler(val_set, cfg.batch_size, False, cfg.seed),
                        num_workers=cfg.num_workers)
    enc = VJEPA2FrameEncoder(device=str(device))
    models = build_models(cfg, device, heads=heads)
    for name, m in models.items():
        if name in ck:
            m.load_state_dict(ck[name], strict=False)
    out = run_epoch(models, enc, loader, cfg, device)
    del enc, models
    torch.cuda.empty_cache()
    return out, len(val_set)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", default="results/eval/loss_before_after.txt")
    a = ap.parse_args()
    dev = torch.device("cuda")
    bs = branch_sample()
    sets = {"clips_val": [f"hexapod={CW}/c10_clips_val", f"b1={CW}/b1_clips_val"],
            "branches_heldout (10 %)": [f"hexapod={bs['c10']}", f"b1={bs['b1']}"]}
    res = {}
    for spec in a.ckpt:
        lab, path = spec.split("=", 1)
        for sn, vs in sets.items():
            res[(lab, sn)] = losses(path, vs, dev)
            print(lab, sn, {k: round(v, 4) for k, v in res[(lab, sn)][0].items()}, flush=True)
    keys = sorted({k for (r, _) in res.values() for k in r})
    L = ["World-model loss terms on the same evaluation sets (training code's validation pass; lower is better).",
         "Each checkpoint with its own training sources' normalisation; " + "; ".join(f"{s.split('=', 1)[0]}" for s in a.ckpt)]
    for sn in sets:
        n = res[(a.ckpt[0].split("=", 1)[0], sn)][1]
        L.append(f"=== {sn} ({n} pairs)")
        L.append("term".ljust(22) + "".join(s.split("=", 1)[0][:22].rjust(24) for s in a.ckpt))
        for k in keys:
            L.append(k.ljust(22) + "".join(f"{res[(s.split('=', 1)[0], sn)][0].get(k, float('nan')):24.4f}" for s in a.ckpt))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    open(os.path.join(ROOT, a.out), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
