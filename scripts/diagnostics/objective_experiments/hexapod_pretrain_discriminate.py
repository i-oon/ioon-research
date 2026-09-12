"""Does `beh12_hinge_multistep_anchor`'s modest `/mean-z` gain (0.886-0.938, F199) translate into
actually picking the right condition, or is it too small to matter for real candidate selection?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/hexapod_pretrain_discriminate.py \\
        --ckpt wm/runs/beh12_hinge_multistep_anchor/best.pt

**Why this can't reuse `wm/adapt3.py`'s own `discriminate()` directly.** That function scores
`proj(action)` -- but a raw pretrain checkpoint has no `ActionProjector` at all (pretraining feeds
the FTM `z = ITM(e_t, e_next)` directly; the projector is only fitted afterward, against a frozen
ITM, in stage 2). So candidates here are **real z's from other conditions' own real transitions**
at a matching time index, not actions run through a projector -- the substitute this checkpoint's
own machinery actually supports, same spirit as `adapt3.py`'s discrimination test.

**Method, matching `adapt3.py`'s own conventions exactly.** At each held-out transition `(e_t,
e_{t+1})` of condition `c`, gather one candidate real z per OTHER available condition (drawn from
a different clip, same time index where possible), plus the transition's own true z. Roll
`FTM(e_t, z)` for every candidate, pick the argmin MSE to the true `e_{t+1}`. A hit is the argmin
landing on the true condition's own z; a family hit is the argmin landing on any condition in the
true family. **Chance is NOT `1/n_candidates`** -- families hold unequal numbers of conditions, so
chance is accumulated per pick over the candidates actually offered, exactly as `adapt3.py` does.
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
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda cond: cond.rsplit("_", 1)[0] if "_" in cond else cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--held_out_frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    embodiment = args.embodiment if getattr(ftm, "embodiments", None) else None

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    reg = REGISTRY[args.embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    n_held = max(1, int(len(paths) * args.held_out_frac))
    held_paths = {paths[i] for i in order[:n_held]}

    clips = []  # each: {"cond": str, "e": [T,256,1408], "held_out": bool}
    for p in paths:
        with np.load(p, allow_pickle=True) as raw:
            cond = str(raw["condition"] if "condition" in raw.files else
                       raw["behavior"] if "behavior" in raw.files else "walk")
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        clips.append({"cond": cond, "e": e, "held_out": p in held_paths})
    del encoder
    torch.cuda.empty_cache()

    # one representative clip per condition, taken from the TRAIN set, used as each candidate's
    # source of a real z -- held-out clips are only ever the query side, never the candidate pool
    cand_source = {}
    for i, c in enumerate(clips):
        if not c["held_out"]:
            cand_source.setdefault(c["cond"], i)
    conds = sorted(cand_source)
    print(f"{len(clips)} clips, {n_held} held out, {len(conds)} candidate conditions\n")

    held = [i for i, c in enumerate(clips) if c["held_out"]]
    picks = [(i, t) for i in held for t in range(clips[i]["e"].shape[0] - 1)]
    if len(picks) > args.limit:
        picks = [picks[j] for j in rng.permutation(len(picks))[:args.limit].tolist()]

    hit = fam_hit = fam_chance_sum = 0
    with torch.no_grad():
        for i, t in picks:
            e_t = clips[i]["e"][t:t + 1]
            truth = clips[i]["e"][t + 1:t + 2]
            true_cond = clips[i]["cond"]
            errs, keep = [], []
            for cond in conds:
                src = clips[cand_source[cond]]
                tt = min(t, src["e"].shape[0] - 2)
                z = itm(src["e"][tt:tt + 1], src["e"][tt + 1:tt + 2])
                pred = ftm(e_t, z, embodiment) if embodiment else ftm(e_t, z)
                errs.append(((pred - truth) ** 2).mean().item())
                keep.append(cond)
            best = keep[int(np.argmin(errs))]
            true_fam = FAMILY(true_cond)
            hit += best == true_cond
            fam_hit += FAMILY(best) == true_fam
            fam_chance_sum += sum(FAMILY(k) == true_fam for k in keep) / len(keep)

    n = max(len(picks), 1)
    print(f"n_picks={n}  n_candidate_conditions={len(conds)}")
    print(f"exact condition top-1: {hit / n:.1%}  (chance {1 / len(conds):.1%})")
    print(f"family top-1:          {fam_hit / n:.1%}  (chance {fam_chance_sum / n:.1%})")
    fam_acc, fam_chance = fam_hit / n, fam_chance_sum / n
    print(f"\n{'SELECTION' if fam_acc > fam_chance + 0.15 else 'NOT selection'} "
         f"(clears chance by >15 points: {fam_acc - fam_chance:+.1%})")


if __name__ == "__main__":
    main()
