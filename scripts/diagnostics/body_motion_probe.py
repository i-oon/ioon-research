"""Does a body-level motion readout transfer between the two robots, where a leg-level one does not?

`leg_contact_probe.py` asks whether "is this leg loaded" moves across embodiments. It does not:
0.986 within the B1 and **0.373 across**, below the frozen encoder's 0.531 and below chance, so
training leaves the two robots' codes pointing opposite ways.

F56 measured why that is not going to be fixed by a better leg-level label. One leg's phase fixes
all four of the B1's legs (concentration 0.99-1.00) and almost nothing about the insect's other
five (0.07-0.24), so the insect's gait carries roughly six loosely coupled degrees of freedom where
the B1's carries one. **Any leg-level quantity is asking for a correspondence that does not exist.**

Body-level motion is the level where a correspondence *does* exist, and F56 measured that too: both
robots walk at a Froude number of 0.155 and 0.159 despite hip heights of 0.13 m and 0.56 m. Every
legged robot has a body, a forward speed and a height, whatever its leg count.

**This is the metric the body-motion decoding term should be judged on, and the leg probe is not.**
LAC-WM's shared latent comes from requiring `z` to decode to a hand-unified motion label -- an
end-effector pose every arm has -- and our `L_motion` targets joint commands instead, 18-D against
12-D, which is why nothing pushes `z` toward shared meaning. The locomotion equivalent of their
label is body motion. Adding that term would not be expected to move the leg probe at all, so
measuring the wrong one would read as failure while the intervention worked.

Same protocol as the leg probe so the two are directly comparable: per-embodiment standardisation,
a clip-level train/test split, and the frozen encoder reported beside the learned latent to say
whether training helped or hurt. The metric is R^2 against **the target embodiment's own mean**, so
**0.0 is the no-learning line** the way 0.500 is for the binary leg probe.

  .venv/bin/python3 scripts/diagnostics/body_motion_probe.py
  .venv/bin/python3 scripts/diagnostics/body_motion_probe.py --ckpt wm/runs/stage2_clean/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.bodies import bodies_in  # noqa: E402
from wm.data.embodiment import B1_DT, HEXAPOD_DT, body_motion  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

DEFAULT_CACHE = os.path.join(ROOT, "results", "wm", "cache", "stage2_embeddings.pt")
INSECT_EPS = [6, 20, 22]
DT = {"insect": HEXAPOD_DT, "b1": B1_DT}


def bands(tokens, grid=16, n=4):
    """Average within four horizontal bands of the patch grid, as the leg probe does.

    Mean-pooling all 256 patches buries a quantity living in a few of them and preserves a constant
    offset between the datasets that a fitted readout absorbs and mis-applies (F41).
    """
    t = tokens.reshape(len(tokens), grid, grid, -1)
    return t.reshape(len(tokens), n, grid // n * grid, -1).mean(2).flatten(1).numpy()


def insect_paths(directory):
    """Every clip in the directory, or the three fixed episodes when it is the 8-body set.

    `ik_walk_8body` has 270 clips and the probe only ever used three episodes of it, so keeping
    that behaviour makes an old measurement reproducible. A speed-varied directory has to be taken
    whole: dropping to three episodes would drop most of the speed range, which is the only thing
    this probe is trying to read.
    """
    full = os.path.join(ROOT, directory)
    if os.path.basename(directory.rstrip("/")) == "ik_walk_8body":
        return [f"{full}/{b}_ep{e}.npz" for b in bodies_in(full) for e in INSECT_EPS]
    return sorted(p for p in glob.glob(f"{full}/*.npz") if "manifest" not in os.path.basename(p))


def gather(encoder, chunk, insect_dir, cache_path, itm=None, checkpoint=None):
    """Per embodiment: features, the Froude number per frame, and clip ids."""
    # A cache keyed by path grows without bound as directories are added, and these are full-width
    # patch tokens -- about 1.4 MB a frame. Half precision halves that; the readout is fitted in
    # float64 by RidgeCV either way.
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    fresh = False
    out = {}
    for name in ("insect", "b1"):
        if name == "insect":
            paths = insect_paths(insect_dir)
        else:
            paths = sorted(glob.glob(f"{ROOT}/data/fwd_b1_50hz/*.npz"))
        feats, labels, clip_id = [], [], []
        for i, path in enumerate(paths):
            if not os.path.exists(path):
                continue
            clip = np.load(path, allow_pickle=True)
            if path not in cache:
                cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu().half()
                fresh = True
            position = clip["head"] if "head" in clip.files else clip["base_pos"]
            # **The same function the training target uses**, not a second implementation. An
            # earlier version of this file smoothed over five frames while `lambda_body` trains on
            # a one-second window, which would have had the probe scoring a quantity the loss was
            # never taught -- mostly within-stride rocking rather than travel speed.
            label = body_motion(position.astype(np.float64), DT[name])[:, 0]
            if itm is None:
                feats.append(bands(cache[path].float()))
            else:
                # the learned latent instead of the frozen encoder: does training make the two
                # embodiments more comparable at body level than V-JEPA2 left them?
                e = cache[path].float()
                off = offset_for(checkpoint, "hexapod" if name == "insect" else "b1")
                if off is not None:
                    e = e - off
                n = len(e) - 1
                with torch.no_grad():
                    z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                                   for t in range(0, n, 8)]).numpy()
                feats.append(z)
                label = label[:len(z)]
            labels.append(label)
            clip_id.append(np.full(len(label), i))
        out[name] = (np.concatenate(feats), np.concatenate(labels), np.concatenate(clip_id))
    if fresh:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    return out


def standardise(x):
    """Each embodiment centred and scaled by its own statistics: removes the colour and
    apparent-size difference using only which dataset a frame came from, never the label."""
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def split_by_clip(n_clips, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_clips)
    return set(order[:int(frac * n_clips)].tolist())


def cell(train, test):
    """Fit on one embodiment's train clips, score R^2 on the other's test clips.

    The target's Froude is standardised by **its own** statistics before scoring, so a readout is
    judged on whether it tracks the target's variation rather than on whether the two robots happen
    to walk at the same absolute speed -- which they nearly do, and which would otherwise flatter
    a cross cell that had learned nothing.
    """
    (x_tr, y_tr), (x_te, y_te) = train, test
    y_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-9)
    y_te = (y_te - y_te.mean()) / (y_te.std() + 1e-9)
    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x_tr, y_tr)
    pred = model.predict(x_te)
    r2 = float(r2_score(y_te, pred))
    # Pearson beside R^2 because the two differ by exactly the scale and offset calibration:
    # r^2 is the R^2 this readout would reach if its gain and intercept were refitted on the
    # target. A cell with low `r` has the direction wrong; a cell with high `r` and very negative
    # R^2 has the direction right and only the calibration wrong. R^2 alone cannot tell them
    # apart, and -7.083 is where that mattered (F66).
    r = 0.0 if pred.std() < 1e-9 else float(np.corrcoef(pred, y_te)[0, 1])
    return r2, r


def readout(train):
    """The ridge fitted on one embodiment, as a callable direction in feature space."""
    x, y = train
    y = (y - y.mean()) / (y.std() + 1e-9)
    return RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x, y)


def correlation(a_train, b_train, tests):
    """Correlation between the two robots' readouts applied to the *same* frames.

    **The stable companion to the R^2 cells, and it exists because they are not.** Two seeds of an
    identical configuration move validation total by 0.7 percent and the cross-embodiment R^2 by
    **27** (F65). R^2 scores a readout fitted on one robot and applied to the other, so it charges
    for scale and offset on top of direction, and it is unbounded below -- a slightly wrong readout
    reads -7 where a slightly right one reads +0.6.

    This asks only the question the claim is about: do the two readouts **order the same frames the
    same way**. Fit one per robot, run both over one common set of frames, correlate the outputs.
    Bounded, symmetric, and blind to the scale and offset that R^2 punishes.

    Comparing the fitted weight vectors directly does *not* work and was the first thing tried
    here. `z` is 64-D and heavily correlated, so a ridge's coefficients are not identified -- most
    of their norm lies in low-variance directions that hardly move a prediction. Measured that way
    every run sat at chance (0.014 to 0.085) including the one with the **best** transfer of all,
    R^2 +0.749, which is the contradiction that exposed the error. Correlating predictions weights
    each direction by how much the data actually varies along it, so the unidentified part drops
    out.

    Averaged over both robots' test frames so neither embodiment's feature distribution decides
    the answer alone.

    Pearson rather than Spearman even though the sentence above says "order". Spearman is the
    literal form of that claim, and it was measured: it tracked Pearson to within 0.013 on every
    run (0.845/0.834, 0.898/0.885, 0.313/0.286). The straight-line assumption costs nothing here,
    so the extra column was dropped rather than carried.
    """
    wa, wb = readout(a_train), readout(b_train)
    scores = []
    for x, _ in tests:
        pa, pb = wa.predict(x), wb.predict(x)
        if pa.std() < 1e-9 or pb.std() < 1e-9:
            continue
        scores.append(np.corrcoef(pa, pb)[0, 1])
    return float(np.mean(scores)) if scores else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--raw", action="store_true", help="skip per-embodiment standardisation")
    ap.add_argument("--ckpt", default="",
                    help="score the learned latent from this checkpoint's ITM as well")
    ap.add_argument("--insect_dir", default="data/fwd_hex8body",
                    help="the hexapod clips. The default reproduces the original measurement; "
                         "point it at a speed-varied set to ask whether body motion is readable "
                         "when there is any body-motion variation to read")
    ap.add_argument("--cache", default="",
                    help="defaults to a file named after --insect_dir, so two directories never "
                         "share one cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cache_path = args.cache or (DEFAULT_CACHE if args.insect_dir.endswith("ik_walk_8body")
                                else os.path.join(ROOT, "results", "wm", "cache",
                                                  f"probe_{os.path.basename(args.insect_dir)}.pt"))

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    rows = [("frozen encoder e_t", gather(encoder, args.chunk, args.insect_dir, cache_path))]
    if args.ckpt:
        path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        itm = InverseTransitionModel(from_checkpoint(checkpoint["config"]))
        itm.load_state_dict(checkpoint["itm"])
        itm.eval()
        rows.append((f"z from {os.path.basename(os.path.dirname(path))}",
                     gather(encoder, args.chunk, args.insect_dir, cache_path, itm, checkpoint)))
    del encoder

    print(f"insect clips from {args.insect_dir}\n")
    print("Body-level motion (Froude number) read out of each representation.")
    print("R^2 against the target embodiment's own mean -- **0.0 is the no-learning line**.\n")
    print(f"{'representation':<26}{'insect->insect':>13}{'b1->b1':>13}"
          f"{'insect->b1':>13}{'b1->insect':>13}{'correlation':>13}")
    for label, data in rows:
        prepared = {}
        for name, (x, y, clip) in data.items():
            x = x if args.raw else standardise(x)
            train_clips = split_by_clip(int(clip.max()) + 1, args.seed)
            in_train = np.isin(clip, list(train_clips))
            prepared[name] = ((x[in_train], y[in_train]), (x[~in_train], y[~in_train]))
        i, b = prepared["insect"], prepared["b1"]
        cells = [cell(i[0], i[1]), cell(b[0], b[1]), cell(i[0], b[1]), cell(b[0], i[1])]
        corr = correlation(i[0], b[0], (i[1], b[1]))
        print(f"{label:<26}" + "".join(f"{r2:>13.3f}" for r2, _ in cells)
              + f"{corr:>13.3f}")
        print(f"{'  same, as Pearson r':<26}" + "".join(f"{r:>13.3f}" for _, r in cells))

    print("\nThe second line of each pair is Pearson r for the same cell. r^2 is the R^2 that")
    print("readout would reach with its gain and offset refitted, so the gap between them is")
    print("calibration and the gap to zero is direction.")
    print("\n`correlation` fits a readout on each robot separately, runs both over the same frames,")
    print("and correlates the outputs: 1.0 means the two robots order speed identically, 0.0 means")
    print("unrelated. Bounded and symmetric where the R^2 cells are neither, and blind to the scale")
    print("and offset R^2 charges for, so it is the number to compare across seeds (F65).")
    print("\nRead the two cross columns against the frozen-encoder row above them, not against")
    print("zero alone: the question is whether training made the robots more comparable at body")
    print("level than V-JEPA2 already had them, which is the comparison the leg probe fails.")
    print("\nFor reference, the same 2x2 for the leg-level 'is this leg loaded' readout (F38):")
    print("  frozen encoder    0.806  0.941  0.531  0.547   (chance 0.500)")
    print("  z, Stage 2        0.811  0.986  0.373  0.401")


if __name__ == "__main__":
    main()
