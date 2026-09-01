"""See the edge artefact in the `lambda_body` target, next to the video it was computed from.

`wm/data/embodiment.py:body_motion` smooths per-frame speed with

    np.convolve(speed, np.ones(w) / w, mode="same")

`mode="same"` returns the middle of the *full* convolution, which is the same as zero-padding the
signal. At frame 0 only half the kernel's taps land on real data and the rest multiply zeros, yet
the sum is still divided by the full window, so the value comes out at roughly **half** the truth --
0.0164 against 0.0328 on the clip below. That is **28 percent of a 72-frame clip** trained on a
number the robot never had.

**Only the gap between the two lines is the bug.** Both dip at the ends because the robot really
does start from rest and slow at the end, and that transient is *useful* -- speed variation is the
one intervention that moved the forward model (F63), and accelerating from a stop is a behaviour
both robots share, which is exactly what an alignment target wants. The fix keeps the transient and
corrects its magnitude; it is not an argument for trimming the data.

It matters twice. The probe reads it as within-clip variation and scores it easily, because the
robot's position in frame says how near the clip's edge it is -- that inflated the measured
"physics" from 0.61 to 0.86. And `body_motion` is the function that builds the **training** target,
so `lambda_body` has been fitting a dip that is an artefact of the padding.

The fix is to divide by how many taps actually landed, which `corrected()` below does.

  .venv/bin/python3 scripts/diagnostics/show_body_motion_edges.py
"""
import argparse
import glob
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import BODY_WINDOW_S, G, HEXAPOD_DT  # noqa: E402


def as_shipped(speed, window):
    """Exactly what `body_motion` does today."""
    return np.convolve(speed, np.ones(window) / window, mode="same")


def corrected(speed, window):
    """The same average over however many samples actually exist at each position.

    Dividing by the number of taps that landed on real data turns the zero-padded sum into a true
    mean, so the ends stay unbiased instead of decaying toward zero.
    """
    kernel = np.ones(window)
    total = np.convolve(speed, kernel, mode="same")
    count = np.convolve(np.ones_like(speed), kernel, mode="same")
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="", help="defaults to a constant-speed insect clip")
    ap.add_argument("--out", default="results/wm/stage2/figures")
    args = ap.parse_args()

    path = args.clip or sorted(glob.glob(f"{ROOT}/data/allocentric/fwd_hex7speed/c10f10t10_ep*.npz"))[0]
    with np.load(path, allow_pickle=True) as clip:
        frames, position = clip["frames"], clip["head"]
    height = float(np.median(position[:, 2]))
    window = max(3, int(round(BODY_WINDOW_S / HEXAPOD_DT)))
    speed = np.gradient(position[:, 0].astype(np.float64), HEXAPOD_DT)
    scale = np.sqrt(G * max(height, 1e-6))
    shipped, fixed = as_shipped(speed, window) / scale, corrected(speed, window) / scale
    raw = speed / scale
    half = window // 2

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(0, half, color="#c0392b", alpha=0.12)
    ax.axvspan(len(raw) - half, len(raw), color="#c0392b", alpha=0.12,
               label=f"padded region, {2 * half}/{len(raw)} frames = {2 * half / len(raw):.0%}")
    ax.plot(raw, color="#bbbbbb", lw=1, label="per-frame speed (unsmoothed)")
    ax.plot(shipped, color="#c0392b", lw=2, label="as shipped — mode='same'")
    ax.plot(fixed, color="#2471a3", lw=2, ls="--", label="corrected — divide by taps that landed")
    ax.set_xlabel("frame"); ax.set_ylabel("forward Froude")
    ax.set_title(f"{name}: the red line is half the blue one at both ends, and only the gap is a bug")
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(out_dir, "body_motion_edge_artefact.png")
    fig.savefig(png, dpi=140); plt.close(fig)

    # the video, because a curve alone does not show that the robot is plainly still walking
    # while the red line is falling
    writer = imageio.get_writer(os.path.join(out_dir, "body_motion_edge_artefact.mp4"),
                                fps=10, macro_block_size=1)
    for t in range(len(frames)):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        a1.imshow(frames[t].astype(np.uint8)); a1.set_xticks([]); a1.set_yticks([])
        a1.set_title(f"frame {t}", fontsize=10)
        a2.axvspan(0, half, color="#c0392b", alpha=0.12)
        a2.axvspan(len(raw) - half, len(raw), color="#c0392b", alpha=0.12)
        a2.plot(shipped, color="#c0392b", lw=2, label="target as shipped")
        a2.plot(fixed, color="#2471a3", lw=2, ls="--", label="corrected")
        a2.axvline(t, color="k", lw=1)
        a2.plot([t], [shipped[t]], "o", color="#c0392b", ms=7)
        a2.set_ylim(min(fixed.min(), shipped.min()) - 0.02, max(fixed.max(), shipped.max()) + 0.02)
        a2.set_xlabel("frame"); a2.set_ylabel("forward Froude"); a2.legend(fontsize=8, loc="lower center")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        writer.append_data(np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3])
        plt.close(fig)
    writer.close()

    print(f"clip {name}, {len(raw)} frames, window {window}")
    print(f"  first frame   as shipped {shipped[0]:.4f}   corrected {fixed[0]:.4f}   "
          f"ratio {shipped[0] / fixed[0]:.2f}")
    print(f"  last frame    as shipped {shipped[-1]:.4f}   corrected {fixed[-1]:.4f}   "
          f"ratio {shipped[-1] / fixed[-1]:.2f}")
    print(f"  middle        as shipped {shipped[len(raw)//2]:.4f}   "
          f"corrected {fixed[len(raw)//2]:.4f}")
    print(f"-> {os.path.relpath(png, ROOT)}")
    print(f"-> {os.path.relpath(os.path.join(out_dir, 'body_motion_edge_artefact.mp4'), ROOT)}")


if __name__ == "__main__":
    main()
