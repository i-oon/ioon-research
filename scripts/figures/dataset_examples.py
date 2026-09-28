"""Example frames for the deck's dataset declaration.

    .venv/bin/python3 scripts/figures/dataset_examples.py

    results/deck/dataset_stage1_hex_bodies.png   a few of the nine hexapod bodies, allocentric, one clip each
    results/deck/dataset_stage2_allocentric.png  hexapod and B1, same condition, third-person camera
    results/deck/dataset_stage2_egocentric.png   hexapod and B1, same condition, head camera
"""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
CELL = 200
GAP = 4
LABEL_W = 230
HEAD_H = 26


def first_clip(directory, key, value):
    for p in sorted(glob.glob(os.path.join(ROOT, directory, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            if str(z[key]) == value:
                return p
    raise FileNotFoundError(f"{directory}: no clip with {key}={value}")


def montage(rows, frame_ids, out, col_title):
    """rows: [(label_lines, npz_path)]; frame_ids: frame indices shown as columns.
    Compact layout for a one-third-width slide column: no label column, the row label is a dark strip on the
    first frame of the row and the time is stamped on every frame."""
    f_big, f_small = ImageFont.truetype(BOLD, 13), ImageFont.truetype(FONT, 11)
    W = len(frame_ids) * (CELL + GAP) - GAP
    H = len(rows) * (CELL + GAP) - GAP
    img = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(img)
    for r, (lines, path) in enumerate(rows):
        frames = np.load(path, allow_pickle=True)["frames"]
        y = r * (CELL + GAP)
        for c, f in enumerate(frame_ids):
            x = c * (CELL + GAP)
            fr = Image.fromarray(frames[min(f, len(frames) - 1)]).resize((CELL, CELL), Image.LANCZOS)
            img.paste(fr, (x, y))
            d.rectangle([x, y + CELL - 18, x + 62, y + CELL], fill=(20, 20, 20))
            d.text((x + 4, y + CELL - 16), f"{f * 0.05:.1f} s", font=f_small, fill=(210, 210, 210))
        d.rectangle([0, y, CELL, y + 18 * len(lines) + 6], fill=(20, 20, 20))
        for k, line in enumerate(lines):
            d.text((5, y + 3 + 18 * k), line, font=f_big if k == 0 else f_small,
                   fill=(255, 255, 255) if k == 0 else (175, 175, 175))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print(out, img.size)


def main():
    deck = os.path.join(ROOT, "results/deck")
    ids = [0, 30, 60]
    title = lambda f: f"frame {f}  ({f * 0.05:.1f} s)"

    # Stage 1: hexapod bodies (leg segment scale = number / 10), forward walking, allocentric
    bodies = [("c06f06t06", "Smallest body"), ("c06f10t10", "Short coxa"), ("c08f09t09", "Held-out body"),
              ("c10f10t10", "Largest body")]
    rows = []
    for m, note in bodies:
        rows.append(([note, "coxa / femur / tibia", "scales " + " / ".join(f"{int(m[i:i + 2]) / 10:.1f}" for i in (1, 4, 7))],
                     first_clip("data/allocentric/fwd_hex8body", "morph", m)))
    montage(rows, ids, os.path.join(deck, "dataset_stage1_hex_bodies.png"), title)

    # Stage 2: hexapod vs. B1, same condition, both cameras
    # the two bodies name their twelve conditions differently (hexapod: gait-parameter s, B1: command w);
    # the third turn level of each is shown
    cond_hex, cond_b1 = "turn_s0.29", "turn_w0.037"
    for view, hex_dir, b1_dir, name in (
            ("third-person", "data/allocentric/beh12_c10f10t10_flat", "data/allocentric/beh12_b1_flat",
             "dataset_stage2_allocentric.png"),
            ("head camera", "data/egocentric/beh12_c10f10t10_ego_flat", "data/egocentric/beh12_b1_ego_flat",
             "dataset_stage2_egocentric.png")):
        rows = [(["Hexapod", "nominal body", f"third turn level, {view}"], first_clip(hex_dir, "condition", cond_hex)),
                (["Unitree B1", "quadruped", f"third turn level, {view}"], first_clip(b1_dir, "condition", cond_b1))]
        montage(rows, ids, os.path.join(deck, name), title)


if __name__ == "__main__":
    main()
