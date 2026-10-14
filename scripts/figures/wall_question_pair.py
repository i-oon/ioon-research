"""Slide figure (weekly 2026-10-15, "A scene-dependent research question"): the SAME forward command from two scenes.
Top: B1 on open floor (held-out clip `b1_ep50012`, speed vx 0.38). Bottom: B1 facing a wall (wall test branch
`ep08805_b1_c01`, command fwd1, the same forward command held from a state ~0.9 m from the wall) until it crosses the
boundary. Frames at 0 / 1.25 / 2.5 s after the decision, grayscale (what the current model sees), clearance printed.

    .venv/bin/python3 scripts/figures/wall_question_pair.py   # -> results/deck/weekly_1015/wall_question_pair.png
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


if __name__ == "__main__":
    main()
