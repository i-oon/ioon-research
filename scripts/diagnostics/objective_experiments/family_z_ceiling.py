"""Is the ACTUAL data separated across a behaviour family, or is `/mean-z family_mean`'s near-1.0
reading (0.886-0.938, `beh12_hinge_multistep_anchor`) a ceiling the data itself imposes rather than
a model weakness?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/family_z_ceiling.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor/best.pt

**Why this check, and why MSE not cosine.** `/mean-z`'s ratio is `MSE(real z) / MSE(mean z)` --
squared-error, not direction. A ceiling check that used cosine distance would measure something
the ratio under question does not, so it cannot answer whether 0.89-0.94 is close to the best this
data allows. This computes the same quantity MSE would, on REAL observed transitions only, with no
model prediction involved at all: how far apart are two conditions' real z's, in the same units
`/mean-z` is read in.

**The one honest caveat this cannot resolve on its own**: if `z` itself is dominated by large,
action-irrelevant components (the same 97%-static-content story F142 measured), an MSE-based
ceiling could ALSO be swamped by that irrelevant variance, reading close to 1 for a different
reason than genuine task redundancy. A ceiling far from 1 is unambiguous (real separation exists,
the model is leaving it on the table); a ceiling near 1 is ambiguous between "the task really is
redundant here" and "the same big-signal-small-signal problem lives in z itself," and must be
reported as such, not resolved by this script alone.

**Method.** For each condition pair within the same family (e.g. `speed_c5.8` vs `speed_c7.1`),
at matching time indices, compute real `z = ITM(e_t, e_t+1)` from REAL frames of each condition
(no FTM, no prediction). `within` = MSE between two DIFFERENT clips of the SAME condition (the
noise floor -- irreducible even for a perfect model). `across` = MSE between different conditions
of the SAME family. `ceiling ratio = within / across`: close to 1 means the family's conditions
are barely more separated than two repeats of the same one -- the data itself does not support a
much lower `/mean-z` than what is already measured. Far from 1 (across >> within) means real,
usable separation exists that the model is not using.
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda cond: cond.rsplit("_", 1)[0] if "_" in cond else cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--pairs_per_family", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    reg = REGISTRY[args.embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))

    by_cond = defaultdict(list)
    for p in paths:
        with np.load(p, allow_pickle=True) as raw:
            cond = str(raw["condition"] if "condition" in raw.files else
                       raw["behavior"] if "behavior" in raw.files else "walk")
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        with torch.no_grad():
            z = itm(e[:-1], e[1:])   # one z per real transition, whole clip
        by_cond[cond].append(z.cpu())
    del encoder
    torch.cuda.empty_cache()

    by_family = defaultdict(list)
    for cond in by_cond:
        by_family[FAMILY(cond)].append(cond)

    rng = np.random.default_rng(args.seed)
    print(f"{len(by_cond)} conditions, {len(by_family)} families\n")
    print(f"{'family':>12}  {'within (noise floor)':>22}  {'across (family)':>18}  "
         f"{'ceiling ratio':>14}  n_within  n_across")
    overall_within, overall_across = [], []
    for fam, conds in sorted(by_family.items()):
        if len(conds) < 2:
            continue  # a family of one condition has no "across" comparison
        within_d, across_d = [], []
        for _ in range(args.pairs_per_family):
            c1 = conds[rng.integers(0, len(conds))]
            z1 = by_cond[c1]
            i = rng.integers(0, len(z1))
            t = rng.integers(0, z1[i].shape[0])
            a = z1[i][t]
            # within: same condition, a different clip if one exists, else a different timestep
            if len(z1) > 1:
                j = rng.integers(0, len(z1) - 1)
                j = j + 1 if j >= i else j
                zb = z1[j]
                tb = rng.integers(0, zb.shape[0])
                b_within = zb[tb]
            else:
                tb = rng.integers(0, z1[i].shape[0])
                b_within = z1[i][tb]
            within_d.append(((a - b_within) ** 2).mean().item())
            # across: a different condition in the SAME family
            others = [c for c in conds if c != c1]
            c2 = others[rng.integers(0, len(others))]
            z2 = by_cond[c2]
            k = rng.integers(0, len(z2))
            tk = rng.integers(0, z2[k].shape[0])
            b_across = z2[k][tk]
            across_d.append(((a - b_across) ** 2).mean().item())
        w, a_ = float(np.mean(within_d)), float(np.mean(across_d))
        ratio = w / max(a_, 1e-9)
        overall_within.extend(within_d); overall_across.extend(across_d)
        print(f"{fam:>12}  {w:>22.5f}  {a_:>18.5f}  {ratio:>14.3f}  "
             f"{len(within_d):>8}  {len(across_d):>8}")

    w, a_ = float(np.mean(overall_within)), float(np.mean(overall_across))
    print(f"\noverall: within {w:.5f}  across {a_:.5f}  ceiling ratio {w / max(a_, 1e-9):.3f}")
    print("\nceiling ratio near 1.0 -> the family's conditions are barely more separated in z than "
         "two repeats of the same condition; the model's /mean-z (0.886-0.938) is then close to "
         "what the data allows, not evidence of a weak model. Far from 1.0 (across >> within, "
         "ratio << 1) -> real separation exists in z that the model is not turning into a lower "
         "/mean-z -- genuine unused signal.\n"
         "Caveat this cannot resolve alone: a ratio near 1.0 could also mean z's large "
         "action-irrelevant content is swamping a real but small action-relevant difference -- "
         "the same big-signal/small-signal story as F142, one level upstream. Report both "
         "readings, do not silently pick one.")


if __name__ == "__main__":
    main()
