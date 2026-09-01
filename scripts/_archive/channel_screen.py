"""Which channels of body motion carry meaning both robots share, and at which timescale.

`lambda_body` supervises one number, forward Froude, and reaches agreement 0.85-0.92 across the two
robots (F66). The obvious next move is more channels -- the source method's shared target is a full
6-DOF pose delta where ours is one axis of it (F67). This asks whether the other channels are
actually there to be used.

**The answer turns on a split that is easy to miss.** A body-velocity channel is the sum of

    slow   what the robot is doing        -- walking speed, turn rate, drift
    fast   where it is in its gait cycle  -- the rocking, bobbing and yawing of each stride

The slow part is behaviour and can be shared: 0.16 Froude means the same thing to a hexapod and a
quadruped. The fast part is the gait's signature, and a six-leg wave and a four-leg trot have
nothing in common there -- F41b measured a per-leg quantity transferring at 0.373, below chance.

So each channel is scored twice, raw and smoothed over about one stride. A channel is usable only
if its **slow** component both varies and transfers.

Three gates, following F69:

    varies        a constant carries no signal, whatever else is true of it
    hides robot   AUC near 0.5 from the target alone; lateral speed failed this at 0.788
    transfers     a readout fitted on one robot, applied to the other

  .venv/bin/python3 scripts/diagnostics/channel_screen.py --ckpt wm/runs/s2_fwd_hex7-b1_body0.5/last.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import BODY_WINDOW_S, G  # noqa: E402
from wm.evaluate import offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

from diagnostics.body_motion_probe import DT, insect_paths  # noqa: E402

AXES = ("forward", "lateral", "vertical")


def velocity(position, dt, height, smooth):
    """Dimensionless body velocity per axis, optionally averaged over one stride.

    Smoothing is what separates the two timescales. `wm/data/embodiment.py` applies the same
    one-second window to build the training target, and the contrast between the two rows here is
    the reason it has to.
    """
    scale = np.sqrt(G * max(height, 1e-6))
    out = []
    for axis in range(3):
        v = np.gradient(position[:, axis].astype(np.float64), dt) / scale
        if smooth:
            window = max(3, int(round(BODY_WINDOW_S / dt)))
            v = np.convolve(v, np.ones(window) / window, mode="same")
        out.append(v)
    return np.stack(out, axis=1)


def by_clip(clip, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    ids = np.unique(clip)
    rng.shuffle(ids)
    return np.isin(clip, ids[:int(frac * len(ids))])


def cell(x_tr, y_tr, x_te, y_te):
    y_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-9)
    y_te = (y_te - y_te.mean()) / (y_te.std() + 1e-9)
    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x_tr, y_tr)
    return float(r2_score(y_te, model.predict(x_te)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/s2_fwd_hex7-b1_body0.5/last.pt")
    ap.add_argument("--insect_dir", default="data/allocentric/fwd_hex7speed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, "results", "wm", "cache",
                              f"probe_{os.path.basename(args.insect_dir)}.pt")
    cache = torch.load(cache_path, map_location="cpu")
    checkpoint = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    itm = InverseTransitionModel(from_checkpoint(checkpoint["config"]))
    itm.load_state_dict(checkpoint["itm"])
    itm.eval()

    sources = (("insect", insect_paths(os.path.join(ROOT, args.insect_dir)), "head"),
               ("b1", sorted(glob.glob(os.path.join(ROOT, "data/allocentric/fwd_b1_50hz/*.npz"))), "base_pos"))
    data = {}
    for name, paths, key in sources:
        Z, Y, C = [], {False: [], True: []}, []
        for i, path in enumerate(paths):
            if path not in cache:
                continue
            with np.load(path, allow_pickle=True) as clip:
                position = clip[key] if key in clip.files else clip["base_pos"]
            e = cache[path].float()
            off = offset_for(checkpoint, "hexapod" if name == "insect" else "b1")
            if off is not None:
                e = e - off
            n = len(e) - 1
            with torch.no_grad():
                z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                               for t in range(0, n, 8)]).numpy()
            Z.append(z)
            C.append(np.full(len(z), i))
            for smooth in (False, True):
                Y[smooth].append(velocity(position, DT[name], float(np.median(position[:, 2])),
                                          smooth)[:len(z)])
        data[name] = (np.concatenate(Z), {s: np.concatenate(v) for s, v in Y.items()},
                      np.concatenate(C))

    zi, yi, ci = data["insect"]
    zb, yb, cb = data["b1"]
    xi = (zi - zi.mean(0)) / (zi.std(0) + 1e-6)
    xb = (zb - zb.mean(0)) / (zb.std(0) + 1e-6)
    ti, tb = by_clip(ci, args.seed), by_clip(cb, args.seed)

    print(f"insect {len(np.unique(ci))} clips, b1 {len(np.unique(cb))} clips\n")
    print(f"{'channel':<12}{'timescale':<12}{'varies':>9}{'robot AUC':>11}"
          f"{'insect->b1':>12}{'b1->insect':>12}")
    for a, axis in enumerate(AXES):
        for smooth in (False, True):
            ai, ab = yi[smooth][:, a], yb[smooth][:, a]
            pooled = np.concatenate([ai, ab])
            who = np.concatenate([np.zeros(len(ai)), np.ones(len(ab))])
            clip = np.concatenate([ci, cb + 1000])
            tr = by_clip(clip, args.seed)
            auc = roc_auc_score(who[~tr], LogisticRegression(max_iter=2000)
                                .fit(pooled[tr, None], who[tr]).decision_function(pooled[~tr, None]))
            # against the forward channel at the same timescale, so "varies" is comparable down a
            # column rather than against a mixture of timescales
            ref = np.concatenate([yi[smooth][:, 0], yb[smooth][:, 0]]).std()
            print(f"{axis:<12}{'smoothed' if smooth else 'per frame':<12}"
                  f"{pooled.std() / ref:>9.2f}{max(auc, 1 - auc):>11.3f}"
                  f"{cell(xi[ti], ai[ti], xb[~tb], ab[~tb]):>12.3f}"
                  f"{cell(xb[tb], ab[tb], xi[~ti], ai[~ti]):>12.3f}")

    print("\nRead the smoothed rows: that is the behaviour, and it is the only part that can be")
    print("shared. A channel whose smoothed row barely varies has nothing in it to align, however")
    print("large its per-frame row is -- that size is the gait's signature, which the two robots do")
    print("not share (F41b, F69).")


if __name__ == "__main__":
    main()
