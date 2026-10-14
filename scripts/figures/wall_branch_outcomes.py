"""Slide figure (weekly 2026-10-15, "Why counterfactual wall data"): ONE saved start state (B1 wall test state
`ep08805_b1`, 0.88 m from the boundary), three of its 28 branches -- every branch starts from the identical physics state,
then holds one command: one stays CLEAR, one comes NEAR (< 0.13 body lengths), one CROSSES the virtual boundary. Frames at
0 / 1.25 s / the branch's last frame (the crossing frame for a crossing branch), grayscale as the model sees them.

    .venv/bin/python3 scripts/figures/wall_branch_outcomes.py           # -> results/deck/weekly_1015/wall_branch_outcomes.png
    .venv/bin/python3 scripts/figures/wall_branch_outcomes.py --video   # -> results/deck/weekly_1015/wall_branch_outcomes.mp4
       (video: the three branches side by side, ego view with time / clearance / status per frame, and the clearance of
        all three over time with the NEAR margin; real time, 20 fps; frames held after a branch ends at its crossing)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from wall_question_pair import gray

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = "data/wall_b1/test/ep08805_b1"      # kept in the balanced test set (ep08801_b4 was moved out by the balance step)
PICK = (("bwd1", "stays CLEAR", (0, 140, 0)), ("turn_left0", "comes NEAR", (200, 140, 0)),
        ("fwd1", "CROSSES", (200, 0, 0)))
OUT = "results/deck/weekly_1015/wall_branch_outcomes.png"
SZ, LAB, HEAD = 220, 22, 28


def main():
    files = {str(np.load(f, allow_pickle=True)["cf_command_name"]): f
             for f in glob.glob(os.path.join(ROOT, STATE + "_c*.npz"))}
    im = Image.new("RGB", (3 * SZ + 20, len(PICK) * (SZ + LAB + HEAD) + 10), (255, 255, 255)); d = ImageDraw.Draw(im)
    y = 5
    for cmd, what, col in PICK:
        x = np.load(files[cmd], allow_pickle=True)
        T = int(x["first_pair"]) - 1; cl = np.asarray(x["clearance"], float); end = len(cl) - 1
        d.text((8, y + 8), f"command {cmd} held from the same state: {what}", fill=col); y += HEAD
        for i, (k, lab) in enumerate(((T, "0 s"), (T + 25, "+1.25 s"), (end, f"+{(end - T) * 0.05:.2f} s"))):
            xx = 5 + i * (SZ + 5)
            im.paste(Image.fromarray(gray(x["frames"][k])).resize((SZ, SZ)), (xx, y))
            d.text((xx + 4, y + SZ + 5), f"{lab}, clearance {cl[k]:+.2f} m", fill=(0, 0, 0))
        y += SZ + LAB
    im.save(os.path.join(ROOT, OUT)); print("->", OUT)


def video():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from weekly_visuals import write_mp4
    files = {str(np.load(f, allow_pickle=True)["cf_command_name"]): f
             for f in glob.glob(os.path.join(ROOT, STATE + "_c*.npz"))}
    runs = []
    for cmd, what, col in PICK:
        x = np.load(files[cmd], allow_pickle=True); T = int(x["first_pair"]) - 1
        runs.append((cmd, what, col, x["frames"][T:], np.asarray(x["clearance"], float)[T:], float(x["margin"])))
    n = max(len(r[3]) for r in runs); W_ = 256
    out = []
    for t in range(n):
        tiles = []
        for cmd, what, col, fr, cl, m in runs:
            k = min(t, len(fr) - 1)
            st = "CROSSED" if cl[k] < 0 else ("NEAR" if cl[k] < m else "SAFE")
            sc = {"SAFE": (0, 140, 0), "NEAR": (200, 140, 0), "CROSSED": (200, 0, 0)}[st]
            im = Image.fromarray(gray(fr[k])).resize((W_, W_)); can = Image.new("RGB", (W_, W_ + 44), (255, 255, 255))
            can.paste(im, (0, 44)); d = ImageDraw.Draw(can)
            d.text((4, 2), f"{cmd}: {what}", fill=col)
            d.text((4, 22), f"t {k * 0.05:4.2f} s  clearance {cl[k]:+.2f} m  {st}" + ("  (ended)" if t >= len(fr) else ""), fill=sc)
            tiles.append(np.asarray(can))
        top = np.hstack(tiles)
        fig, ax = plt.subplots(figsize=(top.shape[1] / 100, 1.9), dpi=100)
        for cmd, what, col, fr, cl, m in runs:
            k = min(t, len(cl) - 1)
            ax.plot(np.arange(k + 1) * 0.05, cl[:k + 1], color=tuple(c / 255 for c in col), lw=2, label=cmd)
        ax.axhline(runs[0][5], color="0.5", ls="--", lw=1); ax.axhline(0, color="k", lw=1)
        ax.text(0.02, runs[0][5] + 0.02, "NEAR margin", fontsize=7, color="0.4"); ax.text(0.02, 0.02, "boundary", fontsize=7)
        ax.set_xlim(0, n * 0.05); ax.set_ylim(-0.1, max(r[4].max() for r in runs) + 0.1)
        ax.set_xlabel("seconds after the command switch", fontsize=8); ax.set_ylabel("clearance (m)", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper right"); ax.tick_params(labelsize=7)
        fig.tight_layout(); fig.canvas.draw(); plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]; plt.close(fig)
        plot = np.asarray(Image.fromarray(plot).resize((top.shape[1], int(plot.shape[0] * top.shape[1] / plot.shape[1]))))
        out.append(np.vstack([top, plot]))
    path = os.path.join(ROOT, OUT[:-4] + ".mp4"); write_mp4(path, out, fps=20); print("->", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    video() if "--video" in sys.argv else main()
