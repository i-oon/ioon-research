"""Which body-motion channels carry shared meaning, now that both robots actually vary in them.

F70 screened six channels and only forward speed passed, and its stated cause was that the other
five were **constants in our data** -- both robots only ever walked forwards. `data/beh12_*` removes
that: twelve matched conditions per robot spanning speed (Froude 0.12-0.21), turn (w_hat
0.007-0.076) and sideways travel (Froude 0.07-0.19), balanced 4/4/4. So this is a direct re-test of
a negative result whose stated failure mode has been deliberately removed.

Four channels, not F70's three: **yaw is the one the collection was built for** and the old screen
had no rotational channel at all.

Each is scored at two timescales, because a body-velocity channel is the sum of what the robot is
doing (slow, shareable) and where it is in its gait cycle (fast, a six-leg wave and a four-leg trot
have nothing in common there). And against F69's three gates:

    varies        a constant carries no signal whatever else is true of it
    hides robot   AUC near 0.5 from the target alone; lateral failed this at 0.788 in F58
    transfers     a readout fitted on one robot, applied to the other, R^2 on held-out clips

**`dt` is read from each clip, never assumed.** F74: the B1 was rendered at 50 Hz against the
insect's 20 until 2026-08-22, and `B1_DT = 0.02` is still the constant in `wm/data/embodiment.py`.
Hard-coding a rate is what let that go unnoticed for so long.

  .venv/bin/python3 scripts/diagnostics/screen_behaviour_channels.py \\
      --ckpt wm/runs/stage2speed7body/last.pt
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
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import (BODY_WINDOW_S, G, HEXAPOD_DT,  # noqa: E402
                                forward_axis)
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

CHANNELS = ("forward", "lateral", "vertical", "yaw")
_WINDOW = [0.0]

# Mean distance of the standing feet from the stance centroid: the moment arm a turn acts through,
# measured from each robot's own model. Height and stance radius disagree about which robot is
# "bigger" -- B1 is 3.19x taller and 0.72x narrower -- so they are genuinely different choices of
# length scale for yaw, and YAW_SCALE=stance swaps to the second.
STANCE = {"hexapod": 0.576, "b1": 0.414}


def heading(quat, embodiment):
    """Yaw of the body, unwrapped, in radians.

    The two robots store orientation differently and neither convention is guessable:

        hexapod   `body_quat` off /abdomen as (x, y, z, w), and the abdomen's **z axis points
                  aft** -- F71 read left and right swapped by taking it as forward
        B1        `base_quat` from MuJoCo as (w, x, y, z), base frame x forward, world z up

    Only differences of this are used, so a constant offset (the aft-pointing axis) cancels.
    """
    q = np.asarray(quat, dtype=np.float64)
    if embodiment == "hexapod":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx = 2 * (x * z + w * y)          # world x,y of the body's z axis
        fy = 2 * (y * z - w * x)
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx = 1 - 2 * (y * y + z * z)
        fy = 2 * (x * y + w * z)
    return np.unwrap(np.arctan2(fy, fx))


def targets(position, quat, dt, embodiment, smooth):
    """The four channels, dimensionless, optionally averaged over about one stride.

    **Forward and lateral come from `wm.data.embodiment`, not from a second implementation here.**
    F79: this file differenced world x and y, so its "forward" was walking speed times how much the
    robot still pointed along world x -- which for a turning robot is mostly a rotation measurement.
    Re-deriving a target in a diagnostic is how a probe ends up scoring a quantity the loss was
    never taught, which F70's docstring already warns about for the smoothing window.
    """
    height = float(np.median(position[:, 2]))
    scale = np.sqrt(G * max(height, 1e-6))
    # **Unsmoothed here, smoothed once below.** `body_velocity` applies its own stride window, so
    # calling it and then smoothing again gave forward and lateral roughly two strides of averaging
    # against yaw's one -- a quiet advantage to forward in every channel comparison, introduced with
    # the F79 frame fix. The body frame comes from the same `forward_axis`, so nothing about F79 is
    # undone; only the double smoothing is.
    f = forward_axis(quat, embodiment)
    left = np.stack([-f[:, 1], f[:, 0]], axis=1)
    v = np.gradient(position[:, :2].astype(np.float64), dt, axis=0)
    out = [(v * f).sum(1) / scale, (v * left).sum(1) / scale,
           np.gradient(position[:, 2].astype(np.float64), dt) / scale]
    # yaw rate is made dimensionless by sqrt(h/g), not by sqrt(g h): it is a rate, not a speed.
    # This is the w_hat that F72's matched-turn table is built on, so the two agree by construction.
    yaw_scale = (np.sqrt(STANCE[embodiment] / G) if os.environ.get("YAW_SCALE") == "stance"
                 else np.sqrt(max(height, 1e-6) / G))
    out.append(np.gradient(heading(quat, embodiment), dt) * yaw_scale)
    out = np.stack(out, axis=1)
    if smooth:
        window = max(3, int(round((_WINDOW[0] or BODY_WINDOW_S) / dt)))
        k = np.ones(window) / window
        out = np.stack([np.convolve(out[:, c], k, mode="same") for c in range(out.shape[1])], 1)
    return out


def by_clip(clip_id, seed, frac=0.7):
    rng = np.random.default_rng(seed)
    ids = np.unique(clip_id)
    rng.shuffle(ids)
    return np.isin(clip_id, ids[:int(frac * len(ids))])


def transfer(x_tr, y_tr, x_te, y_te):
    y_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-9)
    y_te = (y_te - y_te.mean()) / (y_te.std() + 1e-9)
    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x_tr, y_tr)
    return float(r2_score(y_te, model.predict(x_te)))


def bands(tokens, n=4):
    """Average within four horizontal bands of the patch grid, as body_motion_probe does.

    Mean-pooling all 256 patches buries a quantity that lives in a few of them; the bands keep
    coarse vertical structure, which is where body height and pitch show up.
    """
    grid = int(round(tokens.shape[1] ** 0.5))
    t = tokens.reshape(len(tokens), grid, grid, -1)
    rows = np.array_split(np.arange(grid), n)
    return np.concatenate([t[:, r].mean((1, 2)) for r in rows], axis=1)


def load(name, directory, encoder, itm, checkpoint, cache, chunk, features="z", behaviours=()):
    Z, Y, C, cond = [], {False: [], True: []}, [], []
    for i, path in enumerate(sorted(glob.glob(os.path.join(directory, "*.npz")))):
        with np.load(path, allow_pickle=True) as clip:
            if behaviours and str(clip["behaviour"]) not in behaviours:
                continue
            frames = clip["frames"]
            position = clip["head"] if "head" in clip.files else clip["base_pos"]
            quat = clip["body_quat"] if "body_quat" in clip.files else clip["base_quat"]
            # never assume the rate -- see F74
            dt = float(clip["dt"]) if "dt" in clip.files else HEXAPOD_DT
            condition = str(clip["condition"])
        if path not in cache:
            cache[path] = encode_clip(encoder, frames, chunk).cpu().half()
        e = cache[path].float()
        off = offset_for(checkpoint, name)
        if off is not None:
            e = e - off
        n = len(e) - 1
        if features == "frozen":
            # the control: V-JEPA2 untouched. If a channel transfers here and not through `z`,
            # the barrier is in the modules we train, not in the data (F43/F46)
            z = bands(e.numpy())[:n]
        else:
            with torch.no_grad():
                z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                               for t in range(0, n, 8)]).numpy()
        Z.append(z)
        C.append(np.full(len(z), i))
        cond += [condition] * len(z)
        for smooth in (False, True):
            Y[smooth].append(targets(position, quat, dt, name, smooth)[:len(z)])
    return (np.concatenate(Z), {s: np.concatenate(v) for s, v in Y.items()},
            np.concatenate(C), np.array(cond))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/stage2speed7body/last.pt")
    ap.add_argument("--hex_dir", default="data/beh12_hex_flat")
    ap.add_argument("--b1_dir", default="data/beh12_b1_flat",
                    help="**`beh12_b1_flat`, not `beh12_b1_flat`.** The old set clips the robot in 61% of frames, never pins its camera, files the forward clip under `turn_wz0.00`, and turns the opposite way from the insect (F113-F115).")
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--window", type=float, default=0.0,
                    help="smoothing window in seconds, overriding BODY_WINDOW_S. F70 established "
                         "these channels only cross robots at stride scale; whether every channel "
                         "needs the *same* scale was never tested, and yaw's noise floor is 2.6x "
                         "forward's (F85), so it may need a longer one")
    ap.add_argument("--behaviours", default="",
                    help="comma-separated behaviour axes to keep, e.g. 'speed'. **Needed to compare "
                         "against any pre-2026-08-22 number**: the old datasets were forward "
                         "walking only, so 'predict forward speed' there meant predicting the one "
                         "thing that varied. Here eight of twelve conditions are turning or "
                         "strafing, and a harder question is not the same as a worse "
                         "representation")
    ap.add_argument("--split", default="clip", choices=("clip", "condition"),
                    help="what a held-out unit is. 'clip' leaves near-duplicates of a training "
                         "behaviour in the test set -- within-condition spread is 2-10%% of "
                         "between-condition spread, so four clips of one condition say one thing "
                         "four times. 'condition' holds out whole behaviours and is the honest "
                         "measure of whether a readout generalises to a behaviour it never saw")
    ap.add_argument("--features", default="z", choices=("z", "frozen"),
                    help="'z' = the trained latent; 'frozen' = V-JEPA2 bands, which needs no "
                         "checkpoint and so separates a data problem from a model problem")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)

    checkpoint = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    itm = InverseTransitionModel(from_checkpoint(checkpoint["config"]))
    itm.load_state_dict(checkpoint["itm"])
    itm.eval()
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    _WINDOW[0] = args.window
    keep = tuple(b for b in args.behaviours.split(",") if b)
    data = {n: load(n, os.path.join(ROOT, d), encoder, itm, checkpoint, cache, args.chunk,
                    args.features, keep)
            for n, d in (("hexapod", args.hex_dir), ("b1", args.b1_dir))}
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)

    zh, yh, ch, condh = data["hexapod"]
    zb, yb, cb, condb = data["b1"]
    if args.split == "condition":
        # group id = the behaviour, so every clip of a held-out condition leaves together
        uh = {c: i for i, c in enumerate(sorted(set(condh)))}
        ub = {c: i for i, c in enumerate(sorted(set(condb)))}
        ch = np.array([uh[c] for c in condh])
        cb = np.array([ub[c] for c in condb])
    xh = (zh - zh.mean(0)) / (zh.std(0) + 1e-6)
    xb = (zb - zb.mean(0)) / (zb.std(0) + 1e-6)
    th, tb = by_clip(ch, args.seed), by_clip(cb, args.seed)

    print(f"hexapod {len(np.unique(ch))} clips, b1 {len(np.unique(cb))} clips\n")
    print(f"{'channel':<10}{'timescale':<11}{'varies':>8}{'robot AUC':>11}"
          f"{'hex->b1':>9}{'b1->hex':>9}   gates")
    for c, channel in enumerate(CHANNELS):
        for smooth in (False, True):
            ah, ab = yh[smooth][:, c], yb[smooth][:, c]
            pooled = np.concatenate([ah, ab])
            who = np.concatenate([np.zeros(len(ah)), np.ones(len(ab))])
            split = by_clip(np.concatenate([ch, cb + 1000]), args.seed)
            auc = roc_auc_score(
                who[~split], LogisticRegression(max_iter=2000)
                .fit(pooled[split, None], who[split]).decision_function(pooled[~split, None]))
            auc = max(auc, 1 - auc)
            # "varies" is relative to the forward channel at the same timescale, so a column is
            # comparable rather than mixing timescales
            ref = np.concatenate([yh[smooth][:, 0], yb[smooth][:, 0]]).std()
            spread = pooled.std() / ref
            f1 = transfer(xh[th], ah[th], xb[~tb], ab[~tb])
            f2 = transfer(xb[tb], ab[tb], xh[~th], ah[~th])
            gates = "".join(["V" if spread > 0.25 else ".",
                             "H" if auc < 0.65 else ".",
                             "T" if max(f1, f2) > 0.0 else "."])
            print(f"{channel:<10}{'smoothed' if smooth else 'per frame':<11}"
                  f"{spread:>8.2f}{auc:>11.3f}{f1:>9.3f}{f2:>9.3f}   {gates}")
    print("\nVHT = varies / hides which robot / transfers. Read the smoothed rows: the per-frame")
    print("rows are dominated by gait phase, which the two bodies do not share (F70).")


if __name__ == "__main__":
    main()
