"""Watch collected clips before training on them, straight from the frames in the npz.

`collect_ik.py` already renders and stores every frame, so nothing has to be re-simulated to look
at a clip -- which matters because the project's own rule is that a generated body is not a valid
body until you have watched it walk, and a walk check that passes on numbers can still be a robot
sliding its feet or folding a leg.

Two outputs, because they answer different questions:

    video   one mp4 per clip, played back at the real capture rate so the gait's tempo is right
    strip   a single png of evenly spaced stills, for the questions a still answers better --
            is the foot planted, is the leg folding, is the body height sane

The **speed comparison** case is what this was written for: pass several directories and the strip
puts them in rows, same episode and body across the rows, so a retimed clip can be read against its
original frame by frame rather than from memory.

  .venv/bin/python3 scripts/dataset/preview_clips.py data/ik_speed_test_1.35
  .venv/bin/python3 scripts/dataset/preview_clips.py \\
      data/ik_speed_test_0.75 data/ik_speed_test_1.0 data/ik_speed_test_1.35 --match medium_ep6
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FPS = 20.0            # the insect expert runs at 20 Hz; see sim_time in the expert CSV


def clips_in(directory, match):
    paths = sorted(glob.glob(os.path.join(ROOT, directory, "*.npz")))
    if match:
        paths = [p for p in paths if match in os.path.basename(p)]
    return paths


def write_video(path, out_dir, fps, label):
    with np.load(path, allow_pickle=True) as clip:
        frames = clip["frames"]
    # The directory has to be in the name. Clips keep the same basename across collections -- the
    # same body and episode retimed is still `medium_ep6.npz` -- so writing by basename alone
    # silently leaves one file where three were asked for.
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, f"{label}__{name}.mp4")
    # A retimed clip has fewer frames for the same ground covered, so holding fps fixed is what
    # makes the playback actually look faster instead of just being shorter.
    imageio.mimwrite(out, frames.astype(np.uint8), fps=fps, macro_block_size=1)
    return out, len(frames)


def write_strip(groups, out, n_stills):
    """One row per directory, evenly spaced stills across each clip's full length."""
    rows = len(groups)
    fig, axes = plt.subplots(rows, n_stills, figsize=(2.0 * n_stills, 2.2 * rows), squeeze=False)
    for r, (label, path) in enumerate(groups):
        with np.load(path, allow_pickle=True) as clip:
            frames = clip["frames"]
        # Fractions of the clip, not fixed indices: the whole point is that the clips differ in
        # length, so index 40 is a different part of the stride in each of them.
        idx = np.linspace(0, len(frames) - 1, n_stills).round().astype(int)
        for c, t in enumerate(idx):
            ax = axes[r][c]
            ax.imshow(frames[t].astype(np.uint8))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{label}\n{len(frames)} frames", fontsize=8)
            ax.set_title(f"{t / (len(frames) - 1):.0%}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="one or more directories of collected clips")
    ap.add_argument("--match", default="", help="only clips whose filename contains this")
    ap.add_argument("--out", default="results/wm/dataset/preview")
    ap.add_argument("--stills", type=int, default=8)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--no_video", action="store_true", help="strip only, skip the mp4s")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)

    groups = []
    for directory in args.dirs:
        paths = clips_in(directory, args.match)
        if not paths:
            print(f"no clips matching {args.match!r} in {directory}")
            continue
        label = os.path.basename(directory.rstrip("/"))
        groups.append((label, paths[0]))
        if args.no_video:
            continue
        for path in paths:
            written, n = write_video(path, out_dir, args.fps, label)
            print(f"  {n:>3} frames -> {os.path.relpath(written, ROOT)}")

    if len(groups) >= 1:
        name = f"strip_{args.match or 'first'}.png"
        strip = os.path.join(out_dir, name)
        write_strip(groups, strip, args.stills)
        print(f"\nstills -> {os.path.relpath(strip, ROOT)}")
        print("Rows are the directories in the order given, columns are percentages through each")
        print("clip. Check the planted foot: it should stay put against the floor between stills.")


if __name__ == "__main__":
    main()
