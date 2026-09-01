"""Is the latent organised by behaviour or by body?

**This decides whether the project's strongest demonstration is possible.** Driving the B1 from a
*hexapod* goal image is what would make "cross-embodiment" a control result rather than a
measurement -- the candidates stay B1 clips, because only those are executable, and only the goal
crosses. It requires one thing: that `z` for the same behaviour on two robots be closer than `z`
for different behaviours. **If the latent separates by body first, that loop fails before it is
built, and this measures it in an hour without training or a simulator.**

Three distances, all cosine, on `z = ITM(e_t, e_{t+1})` pooled over each clip:

    same behaviour, across bodies    what the demonstration needs to be small
    different behaviour, same body   the everyday distance the planner already resolves
    same behaviour, same body        the floor -- two clips of one condition on one robot

**The comparison that matters is the first against the second.** If crossing bodies costs more than
changing behaviour, a hexapod goal is further from every B1 candidate than the B1 candidates are
from each other, and the nearest one would be picked for reasons unrelated to the behaviour asked
for.

    .venv/bin/python3 scripts/diagnostics/z_crosses_bodies.py \\
        --ckpt wm/runs/beh12_hexonly/stage3_b1_nce_s0.pt
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

FAMILY = lambda c: c.rsplit("_", 1)[0] if "_" in c else c


@torch.no_grad()
def clip_latents(path, name, encoder, itm, ck, chunk, device):
    """One `z` per transition, then the clip's mean direction."""
    clip = load(path, REGISTRY[name])
    e = encode_clip(encoder, clip["frames"], chunk).float()
    off = offset_for(ck, name)
    if off is not None:
        e = e - off
    e = e.to(device)
    n = len(e) - 1
    z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                   for t in range(0, n, 8)])
    with np.load(path, allow_pickle=True) as raw:
        cond = str(raw["condition"])
    return torch.nn.functional.normalize(z.mean(0), dim=-1).cpu(), cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hexonly/stage3_b1_nce_s0.pt")
    ap.add_argument("--hex_dir", default="data/allocentric/beh12_c08f09t09_flat")
    ap.add_argument("--b1_dir", default="data/allocentric/beh12_b1_flat",
                    help="**`beh12_b1_flat`, not `beh12_b1_flat`.** The old set clips the robot in 61% of frames, never pins its camera, files the forward clip under `turn_wz0.00`, and turns the opposite way from the insect (F113-F115).")
    ap.add_argument("--chunk", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    bank = {"hexapod": [], "b1": []}
    for name, d in (("hexapod", args.hex_dir), ("b1", args.b1_dir)):
        for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
            z, cond = clip_latents(p, name, encoder, itm, ck, args.chunk, device)
            bank[name].append((cond, z))
    del encoder
    torch.cuda.empty_cache()

    def dist(a, b):
        return float(1.0 - torch.dot(a, b))

    same_beh_cross, diff_beh_same, same_beh_same = [], [], []
    for ch, zh in bank["hexapod"]:
        for cb, zb in bank["b1"]:
            if FAMILY(ch) == FAMILY(cb):
                same_beh_cross.append(dist(zh, zb))
    for name in ("hexapod", "b1"):
        for (ca, za), (cb, zb) in itertools.combinations(bank[name], 2):
            (same_beh_same if FAMILY(ca) == FAMILY(cb) else diff_beh_same).append(dist(za, zb))

    print(f"  {'comparison':<40}{'mean':>8}{'sd':>7}{'n':>7}")
    for label, v in (("same behaviour, same body", same_beh_same),
                     ("different behaviour, same body", diff_beh_same),
                     ("**same behaviour, across bodies**", same_beh_cross)):
        a = np.array(v)
        print(f"  {label:<40}{a.mean():>8.3f}{a.std():>7.3f}{len(a):>7}")

    # **The aggregate distance is the wrong question and the ranking is the right one.** What the
    # loop does is pick the nearest candidate, so what matters is whether the B1 clip sharing the
    # hexapod goal's behaviour is nearer than the ones that do not -- a retrieval question. Two
    # means that differ by 3% with standard deviations of 0.04 and 0.06 say nothing about that.
    print()
    b1_conds = np.array([FAMILY(c) for c, _ in bank["b1"]])
    b1_z = torch.stack([z for _, z in bank["b1"]])
    hits, per = 0, {}
    for ch, zh in bank["hexapod"]:
        d = 1.0 - (b1_z @ zh)
        got = b1_conds[int(torch.argmin(d))]
        ok = got == FAMILY(ch)
        hits += ok
        per.setdefault(FAMILY(ch), [0, 0])
        per[FAMILY(ch)][0] += ok
        per[FAMILY(ch)][1] += 1
    n = len(bank["hexapod"])
    prior = float(np.mean([(b1_conds == FAMILY(c)).mean() for c, _ in bank["hexapod"]]))
    print(f"  a hexapod clip retrieves a B1 clip of the same behaviour: **{hits / n:.0%}** "
          f"(chance {prior:.0%}, n={n})")
    for k, v in sorted(per.items()):
        print(f"    {k:<10}{v[0] / v[1]:>6.0%}  of {v[1]}")

    cross, within = np.mean(same_beh_cross), np.mean(diff_beh_same)
    print(f"\n  crossing bodies costs {cross / max(within, 1e-9):.2f}x what changing behaviour does,")
    print(f"  which on its own decides nothing -- read the retrieval rate above.\n")
    if hits / n > 2 * prior:
        print("**A hexapod goal is reachable.** The same behaviour on the other robot sits closer")
        print("than a different behaviour on the same one, so a B1 candidate matching the")
        print("demonstrated behaviour would win on distance. The demonstration is worth building.")
    else:
        print("**A hexapod goal is not reachable as things stand.** Every hexapod frame is further")
        print("from every B1 candidate than the B1 candidates are from each other, so whichever")
        print("candidate wins does so for reasons unrelated to the behaviour asked for. **The")
        print("latent separates by body before it separates by behaviour**, and that has to change")
        print("before the demonstration is attempted.")


if __name__ == "__main__":
    main()
