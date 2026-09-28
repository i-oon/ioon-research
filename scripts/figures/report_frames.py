"""Single frames for the report's dataset tables (Section 3.5): one PNG per cell, cropped around the robot for the
third-person view, resized for the head camera.

    .venv/bin/python3 scripts/figures/report_frames.py
"""
import glob
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "report/image/frames")
TIMES = (0, 30, 60)          # frame index; 20 Hz -> 0.0 s, 1.5 s, 3.0 s


def first_clip(directory, key, value):
    for p in sorted(glob.glob(os.path.join(ROOT, directory, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            if str(z[key]) == value:
                return p
    raise FileNotFoundError(f"{directory}: {key}={value}")


def save(path, name, view):
    frames = np.load(path, allow_pickle=True)["frames"]
    for t in TIMES:
        im = Image.fromarray(frames[min(t, len(frames) - 1)])
        if view == "allo":
            im = im.crop((38, 77, 250, 218))          # around the hexapod, aspect 3:2
        elif view == "allo_b1":
            im = im.crop((38, 45, 250, 186))          # the B1 sits higher in the frame
        im = im.resize((360, 240) if view.startswith("allo") else (256, 256), Image.LANCZOS)
        im.save(os.path.join(OUT, f"{name}_{t}.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    for tag, m in (("s1_small", "c06f06t06"), ("s1_shortcoxa", "c06f10t10"), ("s1_heldout", "c08f09t09"),
                   ("s1_large", "c10f10t10")):
        save(first_clip("data/allocentric/fwd_hex8body", "morph", m), tag, "allo")
    hx, b1 = "turn_s0.29", "turn_w0.037"
    save(first_clip("data/allocentric/beh12_c10f10t10_flat", "condition", hx), "s2_hex_allo", "allo")
    save(first_clip("data/allocentric/beh12_b1_flat", "condition", b1), "s2_b1_allo", "allo_b1")
    save(first_clip("data/egocentric/beh12_c10f10t10_ego_flat", "condition", hx), "s2_hex_ego", "ego")
    save(first_clip("data/egocentric/beh12_b1_ego_flat", "condition", b1), "s2_b1_ego", "ego")
    print(len(os.listdir(OUT)), "frames ->", OUT)


if __name__ == "__main__":
    main()
