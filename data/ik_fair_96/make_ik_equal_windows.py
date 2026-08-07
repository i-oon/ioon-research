"""Build equal-length IK clips from longer source clips.

Walk clips use the earliest window because the fixed camera is aimed ahead of the
start pose. Turn clips are selected by a simple body-path score that favours
clear lateral displacement and net heading change inside the window.
"""
import argparse
import glob
import os
import shutil

import numpy as np


ARRAY_KEYS = ("frames", "actions", "forces", "head", "step_idx")


def heading(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def wrap_deg(x):
    return float((x + 180.0) % 360.0 - 180.0)


def window_metrics(head):
    xy = np.asarray(head[:, :2], dtype=float)
    delta = xy[-1] - xy[0]
    end_heading = heading(delta)
    n = len(xy)
    a = xy[min(12, n - 1)] - xy[0]
    b = xy[-1] - xy[max(0, n - 13)]
    tangent_change = wrap_deg(heading(b) - heading(a))
    return {
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "dist": float(np.linalg.norm(delta)),
        "end_heading_deg": end_heading,
        "tangent_change_deg": tangent_change,
    }


def score_turn(head):
    m = window_metrics(head)
    return abs(m["dy"]) * 10.0 + abs(m["end_heading_deg"]) + 0.25 * abs(m["tangent_change_deg"])


def copy_window(src, dst, start, length, behavior):
    d = np.load(src)
    payload = {}
    for key in d.files:
        value = d[key]
        if key in ARRAY_KEYS and len(value) >= start + length:
            payload[key] = value[start:start + length]
        else:
            payload[key] = value
    payload["source_file"] = np.array(os.path.basename(src))
    payload["source_start"] = np.array(start, dtype=np.int32)
    payload["source_length"] = np.array(length, dtype=np.int32)
    payload["behavior"] = np.array(behavior)
    np.savez_compressed(dst, **payload)


def best_turn_start(path, length, stride):
    d = np.load(path)
    head = d["head"]
    best = None
    for start in range(0, len(head) - length + 1, stride):
        score = score_turn(head[start:start + length])
        if best is None or score > best[0]:
            best = (score, start)
    if best is None:
        raise ValueError(f"{path} has fewer than {length} frames")
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default="data/ik_walk_132")
    ap.add_argument("--turn", default="data/ik_turn")
    ap.add_argument("--out", default="data/ik_fair_96")
    ap.add_argument("--length", type=int, default=96)
    ap.add_argument("--turn-stride", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for old in glob.glob(os.path.join(args.out, "*.npz")):
        os.remove(old)

    rows = []
    for src in sorted(glob.glob(os.path.join(args.walk, "*.npz"))):
        tag = os.path.basename(src)
        dst = os.path.join(args.out, tag)
        copy_window(src, dst, 0, args.length, "walk")
        m = window_metrics(np.load(dst)["head"])
        rows.append(dict(tag=tag, behavior="walk", start=0, n=args.length, **m))

    for src in sorted(glob.glob(os.path.join(args.turn, "*_turn_ep*.npz"))):
        tag = os.path.basename(src)
        dst = os.path.join(args.out, tag)
        start = best_turn_start(src, args.length, args.turn_stride)
        copy_window(src, dst, start, args.length, "turn")
        m = window_metrics(np.load(dst)["head"])
        rows.append(dict(tag=tag, behavior="turn", start=start, n=args.length, **m))

    np.save(os.path.join(args.out, "manifest.npy"), rows, allow_pickle=True)
    shutil.copyfile(__file__, os.path.join(args.out, "make_ik_equal_windows.py"))
    for row in rows:
        print(f"{row['tag']:24s} {row['behavior']:4s} start={row['start']:3d} "
              f"dx={row['dx']:+.3f} dy={row['dy']:+.3f} "
              f"end={row['end_heading_deg']:+.1f} tan={row['tangent_change_deg']:+.1f}")
    print(f"\n{len(rows)} clips x {args.length} frames -> {args.out}")


if __name__ == "__main__":
    main()
