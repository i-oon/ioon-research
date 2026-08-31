"""De-risking Direction B: can a motion-organised representation be action-necessary *and* body-shared?

    .venv/bin/python3 scripts/diagnostics/motion_rep_check.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt

**Two properties Direction B needs, and they may trade off against each other.** F159 showed the
action is readable from a single V-JEPA2 frame because the pose encodes the command; F136 showed the
three-channel body coordinate transfers across robots. An encoder rebuild is only worth weeks if a
motion-organised representation **breaks the first without breaking the second**.

The candidate is the cheapest possible one: **the temporal difference in V-JEPA2 space**,
`m_t = e_t+1 - e_t`. No training, no new encoder -- if even this cannot break the redundancy, a
learned motion encoder is a much longer bet on the same hope.

**Part 1, redundancy.** Action R2 from one snapshot against a pair, appearance against motion. A
motion representation should *not* reveal the command from a single snapshot the way a pose does.

**Part 2, sharing.** Fit forward, lateral and yaw on the insect and test on the B1 **without
refitting**, on both representations. If motion fixes redundancy by destroying transfer, Direction B
is dead by the trade-off rather than by either property alone.

**Two of the three outcomes kill B**, and they were fixed before the run. Ridge in the dual on the
full 360,448 dimensions, split by clip, no world model and no pretraining anywhere in this file.
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_structure import FAMILY, gram, ridge_r2  # noqa: E402

CH = ("forward", "lateral", "yaw")


def collect(data, embodiment, ck, cache_path, chunk, stride, device):
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, data), embodiment, encoder, ck, cache, chunk, 1, device)
    if len(cache) > before:
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    App, Mot, Mot2, A, B, fam, cid = [], [], [], [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float()
        bm = load(os.path.join(ROOT, data, c["path"]), REGISTRY[embodiment])["body_motion"]
        bm = np.asarray(bm, dtype=np.float64)
        for t in range(1, min(len(e) - 3, len(bm) - 1), stride):
            App.append(e[t].flatten().half())
            Mot.append((e[t + 1] - e[t]).flatten().half())
            Mot2.append((e[t + 2] - e[t + 1]).flatten().half())
            A.append(c["a"][t].flatten().float())
            B.append(torch.tensor(bm[t][:3], dtype=torch.float64))
            fam.append(FAMILY(c["cond"])); cid.append(ci)
    return (torch.stack(App), torch.stack(Mot), torch.stack(Mot2),
            torch.stack(A).numpy(), torch.stack(B).numpy(), np.array(fam), np.array(cid), clips)


def split(cid, clips):
    order = collections.defaultdict(list)
    for ci in sorted(set(cid.tolist())):
        order[FAMILY(clips[ci]["cond"])].append(ci)
    test = {ci for v in order.values() for ci in v[1::2]}
    te = np.array([c in test for c in cid])
    return ~te, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hex_data", default="data/beh12_c08f09t09_flat")
    ap.add_argument("--b1_data", default="data/beh12_b1_flat")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    from_checkpoint(ck["config"])

    data = {}
    for name, d, cachefile in (("hexapod", args.hex_data, "fid_hexapod.pt"),
                               ("b1", args.b1_data, "fid_b1.pt")):
        data[name] = collect(d, name, ck, os.path.join(ROOT, "results/wm/cache", cachefile),
                             args.chunk, args.stride, device)

    print(f"{args.ckpt}\ntemporal difference in V-JEPA2 space, m_t = e_t+1 - e_t\n")

    # ---------------- Part 1: is the action still readable from one snapshot?
    print("PART 1  action R2, per body, split by clip")
    print(f"  {'body':>8}{'family':>10}{'e_t appearance':>17}{'m_t motion':>13}"
          f"{'[m_t,m_t+1] pair':>19}{'n':>7}")
    part1 = {}
    for name in ("hexapod", "b1"):
        App, Mot, Mot2, A, B, fam, cid, clips = data[name]
        tr, te = split(cid, clips)
        folds = np.array([hash(int(c)) % 4 for c in cid[tr]])
        y = (A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)
        Ka, Km, Km2 = (gram(X, X, device).numpy() for X in (App, Mot, Mot2))
        feats = {"app": Ka, "mot": Km, "pair": Km + Km2}
        preds = {}
        for k, Kf in feats.items():
            _, preds[k], _ = ridge_r2(Kf[np.ix_(tr, tr)], Kf[np.ix_(te, tr)], y[tr], y[te], folds)
        def r2(p, m):
            return 1 - ((p[m] - y[te][m]) ** 2).sum() / max(((y[te][m] - y[tr].mean(0)) ** 2).sum(), 1e-9)
        allm = np.ones(te.sum(), bool)
        part1[name] = {k: r2(preds[k], allm) for k in feats}
        print(f"  {name:>8}{'all':>10}" + "".join(f"{part1[name][k]:>17.3f}" if k == "app" else
              (f"{part1[name][k]:>13.3f}" if k == "mot" else f"{part1[name][k]:>19.3f}")
              for k in feats) + f"{te.sum():>7}")
        for f in sorted(set(fam[te])):
            m = fam[te] == f
            print(f"  {'':>8}{f:>10}" + f"{r2(preds['app'], m):>17.3f}"
                  f"{r2(preds['mot'], m):>13.3f}{r2(preds['pair'], m):>19.3f}{m.sum():>7}")

    # ---------------- Part 2: does the shared coordinate survive the transform?
    print("\nPART 2  body-motion coordinate, fitted on the insect, tested on the B1 unrefitted")
    Ah, Mh, _, _, Bh, _, cidh, cliph = data["hexapod"]
    Ab, Mb, _, _, Bb, _, cidb, clipb = data["b1"]
    trh, teh = split(cidh, cliph)
    folds = np.array([hash(int(c)) % 4 for c in cidh[trh]])
    mu, sd = Bh[trh].mean(0), Bh[trh].std(0) + 1e-9
    yh, yb = (Bh - mu) / sd, (Bb - mu) / sd

    print(f"  {'representation':>16}{'test':>20}" + "".join(f"{c:>10}" for c in CH))
    for label, Xh, Xb in (("e_t appearance", Ah, Ab), ("m_t motion", Mh, Mb)):
        centre = Xh[trh].float().mean(0, keepdim=True).half()
        xh, xb = Xh - centre, Xb - centre
        Kh = gram(xh, xh, device).numpy()
        Kb = gram(xb, xh, device).numpy()
        _, ph, _ = ridge_r2(Kh[np.ix_(trh, trh)], Kh[np.ix_(teh, trh)], yh[trh], yh[teh], folds)
        _, pb, _ = ridge_r2(Kh[np.ix_(trh, trh)], Kb[:, trh], yh[trh], yb, folds)
        for tag, pred, truth in (("insect held-out", ph, yh[teh]), ("**B1 unrefitted**", pb, yb)):
            row = ""
            for j in range(3):
                r = np.corrcoef(pred[:, j], truth[:, j])[0, 1]
                row += f"{r:>10.2f}"
            print(f"  {label:>16}{tag:>20}{row}")

    print("\n  reading, fixed before the run")
    drop = min(part1[n]["app"] - part1[n]["mot"] for n in part1)
    print(f"  single-snapshot action R2 falls by at least {drop:+.3f} across the two bodies")
    print("  " + ("redundancy is NOT broken -> **Direction B is dead**, stop"
                  if drop < 0.10 else
                  "redundancy is broken -> Part 2 decides; a coordinate that fails to transfer "
                  "kills B by the trade-off"))


if __name__ == "__main__":
    main()
