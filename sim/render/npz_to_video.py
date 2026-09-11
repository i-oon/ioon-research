"""Convert one or more closed-loop / render npz files to mp4, no re-running anything.

    .venv/bin/python3 sim/render/npz_to_video.py results/wm/closed_loop/b1_babble_v2/*/*.npz
    .venv/bin/python3 sim/render/npz_to_video.py some_dir/*.npz --out combined_grid.mp4
    .venv/bin/python3 sim/render/npz_to_video.py one_run.npz --fps 20

**One npz per video by default** -- each `<path>.npz` becomes `<path>.mp4` next to it (or under
--out_dir if given). Pass --out to instead merge ALL given files into ONE grid video (a row per
allo/ego, a column per input file, in the order given) -- this is what every merged showcase video
this session used, e.g. `b1_babble_pipeline_data_3behaviours` and `closed_loop_3behaviours_grid`.

**Frame keys read, in this order of preference**: `frames` (allocentric) and `ego_frames`
(egocentric) if both exist -> two rows, labeled. Only one of them -> one row. Neither -> skipped
with a warning, not a crash (some npz files here are scores/traces, not renders).
"""
import argparse
import os

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def label(arr, txt):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((3, 2), txt, fill=(255, 255, 255))
    return np.asarray(im)


def rows_for(path, col_label):
    with np.load(path, allow_pickle=True) as d:
        has_allo = "frames" in d.files
        has_ego = "ego_frames" in d.files
        if not has_allo and not has_ego:
            print(f"  skip {path}: no 'frames' or 'ego_frames' key ({list(d.files)})")
            return None
        allo = d["frames"] if has_allo else None
        ego = d["ego_frames"] if has_ego else None
    rows = []
    if allo is not None:
        rows.append(("allo", allo))
    if ego is not None:
        rows.append(("ego", ego))
    return [(f"{col_label} ({tag})", frames) for tag, frames in rows]


def write_video(out_path, grid_rows, fps):
    """`grid_rows`: list of rows, each a list of (label, frames) column entries."""
    n_frames = min(frames.shape[0] for row in grid_rows for _, frames in row)
    w = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for i in range(n_frames):
        full_rows = [np.concatenate([label(frames[i], txt) for txt, frames in row], axis=1)
                     for row in grid_rows]
        w.append_data(np.concatenate(full_rows, axis=0))
    w.close()
    print(f"saved -> {out_path}  ({n_frames} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="one or more .npz files (shell-glob them yourself)")
    ap.add_argument("--out", default=None, help="merge all inputs into ONE grid video at this "
                    "path instead of one mp4 per input")
    ap.add_argument("--out_dir", default=None, help="one-mp4-per-input mode only: write there "
                    "instead of next to each input")
    ap.add_argument("--labels", default=None, help="--out mode only: SEMICOLON-separated column "
                    "labels (not comma -- a label may itself contain a comma), same order and "
                    "count as the npz args -- e.g. 'expert (locked, sim goal);babble (locked, "
                    "sim goal)'. Defaults to each file's own basename, which is usually an "
                    "unreadable run-command string, not a real label.")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    if args.out:
        labels = [s.strip() for s in args.labels.split(";")] if args.labels else None
        if labels and len(labels) != len(args.npz):
            raise SystemExit(f"--labels has {len(labels)} entries but {len(args.npz)} npz files given")
        per_file_rows = []
        for idx, p in enumerate(args.npz):
            col_label = labels[idx] if labels else os.path.splitext(os.path.basename(p))[0]
            r = rows_for(p, col_label)
            if r:
                per_file_rows.append(r)
        if not per_file_rows:
            raise SystemExit("no input had a renderable frame key")
        n_rows = max(len(r) for r in per_file_rows)
        grid_rows = [[per_file_rows[c][r] for c in range(len(per_file_rows)) if r < len(per_file_rows[c])]
                     for r in range(n_rows)]
        write_video(args.out, grid_rows, args.fps)
    else:
        for p in args.npz:
            col_label = os.path.splitext(os.path.basename(p))[0]
            r = rows_for(p, col_label)
            if not r:
                continue
            out_name = os.path.splitext(os.path.basename(p))[0] + ".mp4"
            out_path = os.path.join(args.out_dir, out_name) if args.out_dir else \
                os.path.splitext(p)[0] + ".mp4"
            write_video(out_path, [[row] for row in r], args.fps)


if __name__ == "__main__":
    main()
