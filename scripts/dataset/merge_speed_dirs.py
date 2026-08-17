"""Merge the per-speed collections into one dataset, keeping cross-body pairing inside a speed.

`collect_ik.py --speed` writes one directory per speed and every one of them reuses the source
episode numbers, so `c10f10t10_ep6.npz` exists five times over with five different lengths. Copying
them into one directory would collide on the filename; renaming only the file would be worse.

**`lambda_cross` pairs on `expert_episode`, the field inside the npz, not on the filename**
(`wm/data/dataset.py`). Left untouched, a 92-frame clip at speed 0.72 and a 60-frame clip at 1.10
both claim to be episode 6, and the loss would decode one body's latent against another body's
frame from a different point in the stride at a different speed. F45 measured what that costs: a
mis-paired frame supplies a **wrong** partner command rather than a noisy one, and wrong targets do
not average out with more data.

So the episode identity has to carry the speed. Each speed gets its own block of episode numbers,
`speed_index * 1000 + original`, written into both the filename and the field. Pairing then happens
only between bodies at the same speed and the same source episode, which is the only place the
frames are at the same point of the same stride.

The original episode is still recoverable as `episode % 1000` and the speed is stored outright, so
nothing about provenance is lost.

  .venv/bin/python3 scripts/dataset/merge_speed_dirs.py \\
      --dirs data/ik_speed_0.72 data/ik_speed_0.82 data/ik_speed_0.91 \\
             data/ik_speed_1.00 data/ik_speed_1.10 \\
      --out data/ik_walk_speed5
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.bodies import walk_check  # noqa: E402


def speed_of(directory):
    """`ik_speed_1.10` -> (1.10, 1.10); `ik_ramp_0.72_1.10` -> (0.72, 1.10).

    Returned as a pair so a ramp and a constant collection never collide on one key. Taking only
    the trailing number would map `ik_ramp_1.10_0.72` and `ik_speed_0.72` to the same value, and
    they would then share a block of episode numbers -- putting a 72-frame ramped clip and a
    92-frame constant clip in the same "episode", which is precisely the mis-pairing this whole
    module exists to prevent.
    """
    name = os.path.basename(directory.rstrip("/"))
    parts = name.split("_")
    nums = []
    for token in parts:
        try:
            nums.append(float(token))
        except ValueError:
            continue
    if not nums:
        raise SystemExit(f"cannot read a speed from {directory!r}; expected ik_speed_<float> "
                         f"or ik_ramp_<float>_<float>")
    return (nums[0], nums[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=1000,
                    help="episode numbers are speed_index * block + original")
    ap.add_argument("--keep_failing", action="store_true",
                    help="copy clips that do not pass wm.bodies.walk_check as well")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{args.out} already has files; remove it first rather than mixing runs")
    os.makedirs(out, exist_ok=True)

    kept, dropped, per_speed = 0, [], {}
    for index, directory in enumerate(sorted(args.dirs, key=speed_of)):
        speed_pair = speed_of(directory)
        speed, speed_end = speed_pair
        source = directory if os.path.isabs(directory) else os.path.join(ROOT, directory)
        paths = sorted(p for p in glob.glob(os.path.join(source, "*.npz"))
                       if "manifest" not in os.path.basename(p))
        n_here = 0
        for path in paths:
            with np.load(path, allow_pickle=True) as clip:
                data = {k: clip[k] for k in clip.files}
            body = str(data["morph"])
            head = data["head"]
            forward = float(head[-1, 0] - head[0, 0])
            lateral = abs(float(head[-1, 1] - head[0, 1]))
            if not args.keep_failing and not (forward >= 0.30 and lateral < 0.20):
                dropped.append(f"{os.path.basename(directory)}/{os.path.basename(path)} "
                               f"(fwd {forward:.2f}, lat {lateral:.2f})")
                continue
            episode = index * args.block + int(data["expert_episode"])
            data["expert_episode"] = np.array(episode)
            data["speed"] = np.array(speed)
            data["speed_end"] = np.array(speed_end)
            np.savez_compressed(os.path.join(out, f"{body}_ep{episode}.npz"), **data)
            kept += 1
            n_here += 1
        per_speed[speed_pair] = n_here

    for source in args.dirs:
        manifest = os.path.join(ROOT, source, "manifest.npy")
        if os.path.exists(manifest):
            shutil.copy(manifest, os.path.join(out, f"manifest_{speed_of(source)}.npy"))

    print(f"{kept} clips -> {args.out}")
    for i, d in enumerate(sorted(args.dirs, key=speed_of)):
        lo, hi = speed_of(d)
        tag = f"{lo}" if lo == hi else f"{lo}->{hi}"
        print(f"  {tag:<12} {per_speed[speed_of(d)]} clips, episode block {i * args.block}")
    if dropped:
        print(f"\ndropped {len(dropped)} clips that fail walk_check:")
        for line in dropped:
            print(f"  {line}")

    # A body missing from a speed block would leave its partners unpairable at that speed, which
    # is the failure `build_stage1_dirs.shared_episodes` exists to prevent one level up.
    bodies, by_episode = set(), {}
    for path in glob.glob(os.path.join(out, "*.npz")):
        body, episode = os.path.basename(path).split("_ep")
        bodies.add(body)
        by_episode.setdefault(episode[:-4], set()).add(body)
    orphans = {e: sorted(bodies - b) for e, b in by_episode.items() if b != bodies}
    print(f"\n{len(bodies)} bodies, {len(by_episode)} episodes")
    if orphans:
        print("episodes missing a body -- these cannot be paired and will be dropped by the "
              "dataset guard:")
        for episode, missing in sorted(orphans.items()):
            print(f"  ep{episode}: missing {', '.join(missing)}")
    else:
        print("every episode has every body, so cross-body pairing is defined throughout")


if __name__ == "__main__":
    main()
