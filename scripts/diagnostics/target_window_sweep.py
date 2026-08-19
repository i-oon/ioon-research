"""Does shortening the window turn the motion target from a state into a change?

**This checks the premise of the planned port before we spend a retrain on it (F67, step 2l).**
LAC-WM's motion decoder is allowed to see the current frame because its target is a *delta* between
two frames, which no single still can supply. Ours is body speed averaged over one second, which the
frozen encoder reads from one frame at R^2 0.676 -- a *state*. The plan assumes that chunking to
five steps, as the source method does, moves our target toward theirs.

**That assumption is not obviously true for locomotion.** A still of an arm says little about where
the gripper goes next. A still of a walking robot shows the leg configuration, which is most of the
gait phase, and gait phase largely determines the instantaneous velocity. Shortening the window
could therefore make the target *more* readable from one frame, not less -- the opposite of what the
plan needs.

So measure it: fit a readout from **one frame's** embedding to the average speed over W steps, for a
range of W, and watch what R^2 does. Falling with shorter W supports the plan. Flat or rising means
chunking does not buy the delta property here, the head has to stay blind, and slide 21 must say so.

Body displacement is a proxy for the foot displacement the port actually proposes -- feet move
faster and more phase-dependently, so this is the optimistic case. If the proxy already fails, the
real target will not do better.

  .venv/bin/python3 scripts/diagnostics/target_window_sweep.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from wm.data.embodiment import G  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402

from diagnostics.body_motion_probe import DT, bands, insect_paths  # noqa: E402

WINDOWS = (1, 2, 5, 10, 20)


def speed_over(position, dt, window):
    """Average forward speed across `window` steps, made dimensionless by hip height.

    Not smoothed: this is exactly `(x[t+w] - x[t]) / (w*dt)`, the quantity a decoder would be asked
    to reconstruct if actions were chunked into `w` steps. The one-second smoothed version the loss
    uses today is close to the w=20 column.
    """
    height = float(np.median(position[:, 2]))
    x = position[:, 0].astype(np.float64)
    delta = (x[window:] - x[:-window]) / (window * dt)
    return (delta / np.sqrt(G * max(height, 1e-6))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--insect_dir", default="data/ik_walk_speed7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="",
                    help="also read each target out of this checkpoint's `z`. Both halves of the "
                         "F64 condition have to hold: the target must be unreadable from the frame "
                         "AND readable from `z`. A target nothing can supply is noise, not a "
                         "constraint.")
    args = ap.parse_args()

    cache_path = os.path.join(ROOT, "results", "wm", "cache",
                              f"probe_{os.path.basename(args.insect_dir)}.pt")
    if not os.path.exists(cache_path):
        raise SystemExit(f"no cached embeddings at {cache_path}; run body_motion_probe.py first")
    cache = torch.load(cache_path, map_location="cpu")

    import glob
    groups = {"insect": insect_paths(os.path.join(ROOT, args.insect_dir)),
              "b1": sorted(glob.glob(f"{ROOT}/data/b1_framed/*.npz"))}

    print("Forward speed averaged over W steps, read from a SINGLE frame's frozen embedding.")
    print("R^2 near 0 means one frame cannot supply it -- the property that makes a target safe")
    print("to decode with a frame-conditioned head (F64).\n")
    itm, checkpoint = None, None
    if args.ckpt:
        path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        itm = InverseTransitionModel(from_checkpoint(checkpoint["config"]))
        itm.load_state_dict(checkpoint["itm"])
        itm.eval()

    print(f"{'embodiment':<12}{'source':<10}" + "".join(f"{'W=' + str(w):>10}" for w in WINDOWS))

    def latent(embeddings, name):
        e = embeddings.float()
        off = offset_for(checkpoint, "hexapod" if name == "insect" else "b1")
        if off is not None:
            e = e - off
        n = len(e) - 1
        with torch.no_grad():
            return torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                              for t in range(0, n, 8)]).numpy()

    for name, paths in groups.items():
      for source in (("frame",) if itm is None else ("frame", "z", "z+frame")):
        row = []
        for window in WINDOWS:
            feats, labels, clips = [], [], []
            for i, path in enumerate(paths):
                if path not in cache:
                    continue
                with np.load(path, allow_pickle=True) as clip:
                    position = clip["head"] if "head" in clip.files else clip["base_pos"]
                y = speed_over(position.astype(np.float64), DT[name], window)
                f = bands(cache[path].float())
                if source == "frame":
                    x = f
                elif source == "z":
                    x = latent(cache[path], name)
                else:
                    # **The measurement F64 actually calls for.** `z` beating the frame is close to
                    # tautological -- `z` is built from e_t AND e_{t+1}, so on any forward-looking
                    # target it holds information a single frame cannot have. The head, though, gets
                    # both inputs at once, so what decides whether it shortcuts is whether the frame
                    # adds anything *on top of* `z`. No gain here means nothing to shortcut to.
                    zz = latent(cache[path], name)
                    x = np.concatenate([zz, f[:len(zz)]], axis=1)
                x = x[:len(y)]
                feats.append(x)
                labels.append(y[:len(x)])
                clips.append(np.full(len(labels[-1]), i))
            x = np.concatenate(feats)
            y = np.concatenate(labels)
            c = np.concatenate(clips)
            x = (x - x.mean(0)) / (x.std(0) + 1e-6)
            y = (y - y.mean()) / (y.std() + 1e-9)
            rng = np.random.default_rng(args.seed)
            ids = np.unique(c)
            rng.shuffle(ids)
            train = np.isin(c, ids[:int(0.7 * len(ids))])
            model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(x[train], y[train])
            row.append(r2_score(y[~train], model.predict(x[~train])))
        print(f"{name:<12}{source:<10}" + "".join(f"{v:>10.3f}" for v in row))

    print("\n`frame` low and `z` high is what a usable target looks like: the head cannot shortcut")
    print("it, and `z` can still supply it. Both low means the target is unlearnable noise.")
    print("\nW=20 at the hexapod's 0.05 s step is the one-second window the loss uses today.")
    print("W=5 is the source method's chunk length. Compare those two columns: a large drop means")
    print("chunking buys the delta property, and no drop means it does not.")


if __name__ == "__main__":
    main()
