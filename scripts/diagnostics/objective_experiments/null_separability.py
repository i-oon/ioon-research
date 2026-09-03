"""Is `ITM(e_t, e_t)` still a *different thing* from a real transition, on this checkpoint's own data?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/null_separability.py \\
        --ckpt wm/runs/beh12_ego/best.pt --data data/egocentric/beh12_c10f10t10_ego_flat \\
        --embodiment hexapod

**A gate, not a report.** `null/real` contrasts a prediction under the real action with one under the
null. **If the two latents are nearly the same vector, that contrast has nothing to measure** and a
ratio near 1.0 would mean "these two identical things predict identically" rather than "the action is
worthless" -- an F160-shaped confound, where the number cannot distinguish the hypothesis from an
artefact of the setup.

**It cannot be checked before training.** The only reading available beforehand feeds egocentric
embeddings to an ITM trained on allocentric ones: measured that way `cos(null, real)` reads 0.952 on
the insect and 0.982 on the B1, against the reference checkpoint's **0.903** on its own held-out
data. But the egocentric transition also moves the embedding **1.76x further** (1561 against 887),
which is what an out-of-distribution artefact looks like rather than a property of the view. **So
this runs on the egocentric-trained ITM, and nothing downstream is read until it passes.**

Exits non-zero on failure.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda c: "side" if c.startswith("side") else c.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--cache", default="")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_cos", type=float, default=0.94,
                    help="**the gate, calibrated on the reference rather than guessed.** The "
                         "allocentric checkpoint -- on which `null/real` = 1.03 was a meaningful "
                         "measurement -- reads **0.903** over the full held-out set, and 0.922 on "
                         "its worst family. The out-of-distribution readings that raised the "
                         "concern were 0.952 and 0.982. 0.94 sits between them with margin over "
                         "the reference; a tighter 0.92 would fail the reference's own turn family")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])

    cache_path = os.path.join(ROOT, args.cache or f"results/wm/cache/nullsep_{args.embodiment}.pt")
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), args.embodiment, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    per, moved = {}, []
    with torch.no_grad():
        for c in clips:
            e = c["e"].float()
            fam = FAMILY(c["cond"])
            for t in range(1, len(e) - 2, args.stride):
                a, b = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                zn, zr = itm(a, a), itm(a, b)
                per.setdefault(fam, []).append(
                    float(torch.nn.functional.cosine_similarity(zn.flatten(1), zr.flatten(1))))
                moved.append(float(torch.norm(b - a)))

    allv = [v for vs in per.values() for v in vs]
    print(f"{args.ckpt}\n{len(clips)} clips of {args.embodiment} from {args.data}\n")
    print(f"  {'family':>10}{'cos(null z, real z)':>22}{'n':>8}")
    for fam in sorted(per):
        print(f"  {fam:>10}{np.mean(per[fam]):>22.3f}{len(per[fam]):>8}")
    print(f"  {'ALL':>10}{np.mean(allv):>22.3f}{len(allv):>8}")
    print(f"\n  mean ||e_t+1 - e_t|| = {np.mean(moved):.1f}   "
          f"(allocentric insect reads 887, egocentric 1561)")
    print(f"  reference: the allocentric checkpoint, on which `null/real` = 1.03 was a meaningful\n"
          f"  measurement, reads **0.903** overall and 0.922 on its worst family (turn).\n")

    if np.mean(allv) > args.max_cos:
        print(f"**GATE FAILED: {np.mean(allv):.3f} > {args.max_cos}.** The null latent and a real "
              f"transition's latent are\nnearly the same vector, so `null/real` would be comparing "
              f"a thing with itself.\n**Stop. Do not read null/real.** Report this instead: it is a "
              f"result about the null, not about\nthe action.")
        raise SystemExit(1)
    print(f"gate passed: {np.mean(allv):.3f} <= {args.max_cos} — the null is a distinct latent, so "
          f"`null/real` has\nsomething to measure")


if __name__ == "__main__":
    main()
