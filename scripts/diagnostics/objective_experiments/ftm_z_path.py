"""Where does `z` enter the FTM's output -- through the dynamics, or by a short additive path?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/ftm_z_path.py

**The question the multi-step null raised.** `multistep_derisk.py` showed the FTM can be made better
at multi-step prediction while stamping `z` into its output at an unchanged rate, so the stamp is not
single-step myopia. This asks whether it is architectural instead.

**At `z_tokens = 1` the FTM's cross-attention is not attention.** One key/value means the softmax is
over a single element, so its weight is 1 whatever the query is: the attended term is a per-sample
constant, identical across all 256 visual tokens and independent of the visual content. The block is
therefore `visual <- visual + c(z)`, an additive conditioning bias, applied at **every** block
including the last -- and the head reads the last block's output.

    z -> latent_proj -> [block_k.cross] -> + visual -> norm -> head -> prediction

**The ablation.** Feed the real `z` at some blocks and the batch-mean `z` at others, then measure how
much of the action survives in the output. If replacing `z` at the final block alone collapses the
stamp, the action is written in late and bypasses the dynamics. If the stamp holds until early
blocks are also replaced, `z` is routed through the transition computation and the cause is
elsewhere.

Reproduces the real forward exactly when nothing is ablated, so the `none` row is the
`where_action_lives.py` number and the comparison is against a check, not an assumption.

**Diagnosis only; trains nothing and writes no checkpoint.**
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

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402


def ftm_forward(ftm, x_t, z, z_alt, ablate):
    """The FTM's own forward, with `z_alt` substituted at the blocks in `ablate`."""
    visual = ftm.visual_proj(x_t)
    lat = ftm.latent_proj(z).view(-1, ftm.z_tokens, ftm.hidden)
    lat_alt = ftm.latent_proj(z_alt).view(-1, ftm.z_tokens, ftm.hidden)
    for i, block in enumerate(ftm.blocks):
        visual = block.visual(visual)
        lat = block.latent(lat)
        lat_alt = block.latent(lat_alt)
        visual = block.cross(visual, lat_alt if i in ablate else lat)
    return ftm.head(ftm.norm(visual))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for m in (itm, ftm):
        for p in m.parameters():
            p.requires_grad_(False)

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, None, ck, cache, 2,
                   max(1, cfg.action_lag), device)

    n_blocks = len(ftm.blocks)
    schemes = {
        "none  (the real forward)": set(),
        f"last block only  ({n_blocks - 1})": {n_blocks - 1},
        f"last two  ({n_blocks - 2},{n_blocks - 1})": {n_blocks - 2, n_blocks - 1},
        f"second half  ({n_blocks // 2}..{n_blocks - 1})": set(range(n_blocks // 2, n_blocks)),
        f"first half  (0..{n_blocks // 2 - 1})": set(range(n_blocks // 2)),
        "every block  (mean z throughout)": set(range(n_blocks)),
    }

    # z is replaced by the mean z of the whole set: the same "no particular action" substitute the
    # `/mean-z` diagnostics use, so a collapse here means the same thing it means there
    zs = []
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            for t in range(1, len(e) - 2, args.stride):
                zs.append(itm(e[t:t + 1].to(device), e[t + 1:t + 2].to(device))[0].cpu())
    z_mean = torch.stack(zs).mean(0, keepdim=True).to(device)

    cols, A, clip_id = collections.defaultdict(list), [], []
    with torch.no_grad():
        for ci, c in enumerate(clips):
            e = c["e"].float()
            for t in range(1, len(e) - 2, args.stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                z = itm(e_t, e1)
                for name, ab in schemes.items():
                    out = ftm_forward(ftm, e_t, z, z_mean.expand(len(z), -1), ab)
                    cols[name].append(out[0].flatten().half().cpu())
                cols["e_t+1 ground truth"].append(e[t + 1].flatten().half())
                A.append(c["a"][t].flatten().float())
                clip_id.append(ci)

    A = torch.stack(A).numpy()
    clip_id = np.array(clip_id)
    order = collections.defaultdict(list)
    for ci in sorted(set(clip_id.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test_clips = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test_clips for c in clip_id]); tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in clip_id[tr]])
    y = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)

    print(f"{args.ckpt}\n{len(clips)} clips from {args.data}, {n_blocks} FTM blocks, "
          f"z_tokens {cfg.z_tokens}")
    print(f"{tr.sum()} train / {te.sum()} test transitions\n")
    print(f"  {'z fed as the batch mean at':>34}{'action R2':>11}{'vs none':>10}")
    base = None
    for name in list(schemes) + ["e_t+1 ground truth"]:
        g = gram(torch.stack(cols[name]), torch.stack(cols[name]), device).numpy()
        g = g / max(np.mean(np.diag(g)), 1e-12)
        r2, _p, _a = ridge_r2(g[np.ix_(tr, tr)], g[np.ix_(te, tr)], y[tr], y[te], folds)
        if base is None:
            base = r2
        tag = "" if name == "e_t+1 ground truth" else f"{r2 - base:>+10.3f}"
        print(f"  {name:>34}{r2:>11.3f}{tag}")


if __name__ == "__main__":
    main()
