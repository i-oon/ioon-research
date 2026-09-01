"""One video per counterfactual: the shared prefix in sync, then the split, side by side.

    .venv/bin/python3 scripts/render/merge_counterfactual.py \\
        --a data/cf_insect/straight.npz --b data/cf_insect/turn.npz --branch 33 \\
        --label_a "forward" --label_b "turn" --out results/cf/insect_fwd_vs_turn.mp4

**The render is the check on whether the counterfactual is real.** Both panels play the *identical*
commanded prefix, so up to the branch frame they should look the same; after it one takes action A
and the other action B. What the viewer is asked to confirm is exactly the thing the numbers claim:
**same start, a visible moment of splitting, futures that keep moving apart.**

The branch frame is marked three ways because one is easy to miss: the border turns from grey to
colour, a caption changes from `shared prefix` to the two action names, and a bar under the frame
counts the frames since the split. **Before the branch the two panels are the same experiment; after
it they are not, and the video should never leave that ambiguous.**

Frames come from the `.npz`, which is the source of truth -- the mp4 is for humans only and the
codec alters pixels (`npz_to_video.py`).
"""
import argparse
import os

import os as _os
import sys as _sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__)))))
from wm.data.embodiment import heading as _heading  # noqa: E402

PAD, BAR = 34, 22
QUAT = {"body_quat": "hexapod", "base_quat": "b1"}


def frames_of(path):
    with np.load(path, allow_pickle=True) as z:
        return np.asarray(z["frames"])


def heading_of(path):
    """Heading per frame, in radians, or None if the clip carries no orientation.

    **Turning lives here and not in the position channel.** The B1's turn branch moves its base
    10.2 mm by h=25 -- under a pixel -- while its heading has moved 13.6 degrees, so a video that
    shows only where the robot *is* makes the strongest turn counterfactual look like nothing.
    """
    with np.load(path, allow_pickle=True) as z:
        for field, emb in QUAT.items():
            if field in z.files:
                return _heading(np.asarray(z[field], float), emb)
    return None


def compass(d, cx, cy, r, angle, colour, ref=True):
    """One body-axis arrow, drawn **relative to the branch frame**, so both start level."""
    if ref:
        d.line([cx - r, cy, cx + r, cy], fill=(110, 110, 110), width=1)
    x, y = cx + r * np.cos(-angle), cy + r * np.sin(-angle)
    d.line([cx, cy, x, y], fill=colour, width=3)
    d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=colour)


def panel(img, label, colour, since, dpsi=None):
    h, w = img.shape[:2]
    out = Image.new("RGB", (w + 2 * 4, h + PAD + BAR), colour)
    out.paste(Image.fromarray(img), (4, PAD))
    d = ImageDraw.Draw(out)
    d.text((8, 8), label, fill=(255, 255, 255))
    if dpsi is not None:
        compass(d, 4 + 40, PAD + h - 40, 26, dpsi, (255, 255, 255))
        d.text((4 + 68, PAD + h - 46), f"{np.degrees(dpsi):+6.1f} deg", fill=(255, 255, 255))
    if since is not None:
        d.rectangle([4, h + PAD + 4, 4 + int((w - 8) * min(since / 30.0, 1.0)), h + PAD + BAR - 6],
                    fill=(255, 255, 255))
        d.text((w - 74, h + PAD + 4), f"+{since}", fill=(255, 255, 255))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--branch", type=int, required=True,
                    help="frame at which the two commands stop being identical")
    ap.add_argument("--label_a", default="A"); ap.add_argument("--label_b", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10, help="half real time; the split is easy to miss")
    ap.add_argument("--hold", type=int, default=8, help="frames to freeze on the branch")
    args = ap.parse_args()

    A, B = frames_of(args.a), frames_of(args.b)
    HA, HB = heading_of(args.a), heading_of(args.b)
    n = min(len(A), len(B))
    if HA is None or HB is None:
        print("  no orientation field in one of the clips; the compass is omitted")
    grey, ca, cb = (70, 70, 70), (30, 110, 200), (200, 90, 30)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    out = []
    for t in range(n):
        pre = t < args.branch
        since = None if pre else t - args.branch
        la = "shared prefix" if pre else args.label_a
        lb = "shared prefix" if pre else args.label_b
        da = db = None
        ref = max(args.branch - 1, 0)                    # the last SHARED frame, not the first split one
        if HA is not None and HB is not None and ref < len(HA) and ref < len(HB):
            wrap = lambda x: float(np.arctan2(np.sin(x), np.cos(x)))
            da, db = wrap(HA[t] - HA[ref]), wrap(HB[t] - HB[ref])
        row = np.concatenate([panel(A[t], la, grey if pre else ca, since, da),
                              panel(B[t], lb, grey if pre else cb, since, db)], axis=1)
        if da is not None and not pre:
            # **Bottom of the image, on the seam.** At the top it lands on the right panel's own
            # label, and inside the bottom strip it lands under the progress bar -- both checked by
            # looking at a rendered frame rather than by reasoning about coordinates.
            im = Image.fromarray(row); dr = ImageDraw.Draw(im)
            gap = np.degrees(np.arctan2(np.sin(da - db), np.cos(da - db)))
            txt = f"heading apart {abs(gap):5.1f} deg"
            dr.text((row.shape[1] // 2 - 4 * len(txt), row.shape[0] - BAR - 16), txt,
                    fill=(255, 255, 255))
            row = np.asarray(im)
        out.append(row)
        if t == args.branch - 1:
            out += [row] * args.hold                      # hold on the last shared frame
    imageio.mimsave(args.out, out, fps=args.fps, quality=8, macro_block_size=1)

    d = np.abs(A[:args.branch].astype(np.int16) - B[:args.branch].astype(np.int16)).mean()
    print(f"{args.out}  {n} frames, branch at {args.branch}")
    print(f"  mean pixel difference over the SHARED prefix: {d:.3f} / 255")
    print("  **this should be near zero.** If the prefix visibly differs, the two runs did not "
          "share a start\n  and every divergence number after the branch is measuring that instead.")
    if HA is not None and HB is not None and args.branch < min(len(HA), len(HB)):
        w = lambda x: np.arctan2(np.sin(x), np.cos(x))
        r = max(args.branch - 1, 0)
        gap = np.degrees(w((HA[n - 1] - HA[r]) - (HB[n - 1] - HB[r])))
        print(f"  heading apart at the last frame: {abs(gap):.1f} deg")


if __name__ == "__main__":
    main()
