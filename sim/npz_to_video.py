"""Export the .npz episodes as watchable H.264 .mp4 videos (+ a grid overview).

Why the data is stored as .npz, not video:
  - .npz keeps frames + a_t + head position + step_idx together, lossless.
  - Video codecs alter pixels, which would corrupt what V-JEPA2 sees.
So .npz is the source of truth; these .mp4s are only for human inspection.

Uses imageio's bundled ffmpeg (H.264) so the files play in VS Code / browsers.
(OpenCV on this machine can only write mp4v, which those players reject.)

Usage:
  python sim/npz_to_video.py --data data/step0          # one mp4 per episode + grid
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

PERIOD = 64
MORPHS = ("long", "medium", "short")


def label(arr, txt):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 15], fill=(0, 0, 0))
    d.text((3, 3), txt, fill=(255, 255, 255))
    return np.asarray(im)


def write_mp4(path, frames_iter, fps):
    w = imageio.get_writer(path, fps=fps, codec="libx264",
                           quality=8, macro_block_size=1,
                           ffmpeg_params=["-pix_fmt", "yuv420p"])  # yuv420p = plays everywhere
    for fr in frames_iter:
        w.append_data(fr)
    w.close()


def morph_from_tag(tag):
    for morph in MORPHS:
        if tag == morph or tag.startswith(morph + "_"):
            return morph
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/step0")
    ap.add_argument("--fps", type=int, default=20)  # sim is 20 Hz -> real-time
    ap.add_argument("--pattern", type=str, default="*_ep*.npz")
    args = ap.parse_args()

    out = os.path.join(args.data, "videos")
    os.makedirs(out, exist_ok=True)
    # clear stale mp4v/gif files from earlier attempts
    for old in glob.glob(os.path.join(out, "*.mp4")) + glob.glob(os.path.join(out, "*.gif")):
        os.remove(old)

    files = sorted(glob.glob(os.path.join(args.data, args.pattern)))
    for f in files:
        tag = os.path.basename(f).replace(".npz", "")
        d = np.load(f)
        fr, si = d["frames"], d["step_idx"]
        gen = (label(fr[i], f"{tag}  t={int(si[i])}  phase={int(si[i]) % PERIOD}/{PERIOD}")
               for i in range(len(fr)))
        p = os.path.join(out, tag + ".mp4")
        write_mp4(p, gen, args.fps)
        print(f"  {tag:12s} {len(fr)} frames -> {p}  ({os.path.getsize(p)//1024} KB)")

    # grid: one episode per morphology, side by side
    picks = {}
    for f in files:
        tag = os.path.basename(f).replace(".npz", "")
        m = morph_from_tag(tag)
        if m is not None:
            picks.setdefault(m, f)
    order = [picks[m] for m in ["long", "medium", "short"] if m in picks]
    labs = ["long 1.0x", "medium 0.75x", "short 0.5x"]
    arrs = [np.load(p)["frames"] for p in order]
    si = np.load(order[0])["step_idx"]
    n = min(len(a) for a in arrs)

    def grid_gen():
        for i in range(n):
            tiles = [label(a[i], f"{l}  ph={int(si[i]) % PERIOD}") for a, l in zip(arrs, labs)]
            yield np.hstack(tiles)

    gp = os.path.join(out, "grid_overview.mp4")
    write_mp4(gp, grid_gen(), args.fps)
    print(f"  grid (long|medium|short) -> {gp}  ({os.path.getsize(gp)//1024} KB)")
    print(f"\nH.264 videos in {out}/")


if __name__ == "__main__":
    main()
