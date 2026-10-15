"""Slide figure (weekly 2026-10-15, "A scene-dependent research question"): the SAME forward command from two scenes.
Top: B1 on open floor (held-out clip `b1_ep50012`, speed vx 0.38). Bottom: B1 facing a wall (wall test branch
`ep08805_b1_c01`, command fwd1, the same forward command held from a state ~0.9 m from the wall) until it crosses the
boundary. Frames at 0 / 1.25 / 2.5 s after the decision, grayscale (what the current model sees), clearance printed.

    .venv/bin/python3 scripts/figures/wall_question_pair.py           # -> results/deck/weekly_1015/wall_question_pair.png
    .venv/bin/python3 scripts/figures/wall_question_pair.py --paths   # -> results/deck/weekly_1015/wall_question_paths.png
       (slide 2, 2026-10-15: the two START views large, side by side, same goal / same command, each with a top-down inset
        of the footprint's path over the next ~3 s only with --inset -- open floor: the path goes on; wall: it crosses the red boundary)
"""
import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPEN = "data/counterfactual_walks/b1_clips_heldout/b1_ep50012.npz"
WALL = "data/wall_b1/test/ep08805_b1_c01.npz"
OUT = "results/deck/weekly_1015/wall_question_pair.png"
SZ, LAB, HEAD = 256, 22, 30


def gray(fr):
    g = (fr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).clip(0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def main():
    o = np.load(os.path.join(ROOT, OPEN), allow_pickle=True)
    w = np.load(os.path.join(ROOT, WALL), allow_pickle=True)
    T = int(w["first_pair"]) - 1                                      # last shared frame = the decision state
    cw = np.asarray(w["clearance"], float); cross = int(np.argmax(cw < 0))
    rows = [("open floor, command: walk forward", [gray(o["frames"][k]) for k in (0, 25, 50)],
             ["0 s", "+1.25 s", "+2.5 s"]),
            ("wall ahead, the same command", [gray(w["frames"][k]) for k in (T, T + 25, cross)],
             [f"0 s, clearance {cw[T]:+.2f} m", f"+1.25 s, {cw[T + 25]:+.2f} m",
              f"+{(cross - T) * 0.05:.2f} s, {cw[cross]:+.2f} m: CROSSED"])]
    im = Image.new("RGB", (3 * SZ + 20, 2 * (SZ + LAB + HEAD) + 10), (255, 255, 255)); d = ImageDraw.Draw(im)
    y = 5
    for title, frs, labs in rows:
        d.text((8, y + 8), title, fill=(0, 0, 0)); y += HEAD
        for i, (fr, lab) in enumerate(zip(frs, labs)):
            x = 5 + i * (SZ + 5)
            im.paste(Image.fromarray(fr).resize((SZ, SZ)), (x, y))
            d.text((x + 4, y + SZ + 5), lab, fill=(200, 0, 0) if "CROSSED" in lab else (0, 0, 0))
        y += SZ + LAB
    os.makedirs(os.path.dirname(os.path.join(ROOT, OUT)), exist_ok=True)
    im.save(os.path.join(ROOT, OUT)); print("->", OUT)


INSET = False                                                       # --inset: the top-down footprint-path inset


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def paths():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    o = np.load(os.path.join(ROOT, OPEN), allow_pickle=True)
    w = np.load(os.path.join(ROOT, WALL), allow_pickle=True)
    T = int(w["first_pair"]) - 1
    L_, W_ = np.asarray(w["foot_lw"], float); H = float(w["boundary_half"])
    cases = [("Open floor", gray(o["frames"][0]), np.asarray(o["base_pos"], float)[:60, :2],
              np.asarray(o["base_quat"], float)[:60], None),
             ("Approaching a wall", gray(w["frames"][T]), np.asarray(w["base_pos"], float)[T:, :2],
              np.asarray(w["base_quat"], float)[T:], H)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax, (title, img, xy, q, half) in zip(axes, cases):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=15)
        crossed = half is not None and bool((np.asarray(w["clearance"])[T:] < 0).any())
        if not INSET:                                               # 2026-10-15: inset off on the slide (user)
            ax.text(0.5, -0.06, "command: forward" + ("  ->  crosses the boundary" if crossed else "  ->  keeps walking"),
                    transform=ax.transAxes, ha="center", fontsize=13, color="red" if crossed else "green")
            continue
        ins = ax.inset_axes([0.62, 0.02, 0.36, 0.36]); ins.set_facecolor("white")
        y0 = _yaw(q[0]); R = np.array([[np.cos(-y0 + np.pi / 2), -np.sin(-y0 + np.pi / 2)],
                                       [np.sin(-y0 + np.pi / 2), np.cos(-y0 + np.pi / 2)]])
        loc = (xy - xy[0]) @ R.T                                   # robot frame at the start, heading = up
        crossed = False
        if half is not None:                                       # the wall the robot heads for, in the same frame
            p0 = xy[0]; dw = [half - p0[0], half + p0[0], half - p0[1], half + p0[1]]; k = int(np.argmin(dw))
            a, b = {0: ((half, -20), (half, 20)), 1: ((-half, -20), (-half, 20)),
                    2: ((-20, half), (20, half)), 3: ((-20, -half), (20, -half))}[k]
            seg = (np.array([a, b]) - p0) @ R.T
            ins.plot(seg[:, 0], seg[:, 1], color="red", lw=3, label="boundary")
            crossed = bool((np.asarray(w["clearance"])[T:] < 0).any())
        ins.plot(loc[:, 0], loc[:, 1], color="red" if crossed else "green", lw=2.5)
        for j, c_ in ((0, "0.5"), (len(loc) - 1, "red" if crossed else "green")):
            yj = _yaw(q[j]) - y0 + np.pi / 2
            corners = np.array([[L_ / 2, W_ / 2], [L_ / 2, -W_ / 2], [-L_ / 2, -W_ / 2], [-L_ / 2, W_ / 2]])
            rr = np.array([[np.cos(yj), -np.sin(yj)], [np.sin(yj), np.cos(yj)]])
            ins.add_patch(Polygon(corners @ rr.T + loc[j], closed=True, fill=False, ec=c_, lw=1.5))
        ins.set_xlim(-1.6, 1.6); ins.set_ylim(-0.8, 2.4); ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
        ins.set_title("footprint path, next %.1f s" % ((len(loc) - 1) * 0.05), fontsize=8)
        ax.text(0.5, -0.06, "command: forward" + ("  ->  crosses the boundary" if crossed else "  ->  keeps walking"),
                transform=ax.transAxes, ha="center", fontsize=13, color="red" if crossed else "green")
    fig.suptitle("Goal: walk forward  (same goal, same command, different consequence)", fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = os.path.join(ROOT, "results/deck/weekly_1015/wall_question_paths.png"); fig.savefig(out, dpi=130)
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    import sys
    INSET = "--inset" in sys.argv
    paths() if "--paths" in sys.argv else main()
