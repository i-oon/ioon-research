"""Flatten the per-condition behaviour collections into one directory per embodiment.

`collect_ik.py` and `render_b1_replay.py` each write a directory per condition, and every one of
them restarts its episode numbering, so `c10f10t10_ep6.npz` exists twelve times over meaning twelve
different behaviours. Copying them into one place would collide on the filename; keeping them apart
means every downstream script has to know the twelve directory names.

**The condition has to travel inside the npz, not in the path.** `channel_screen.py` and the probes
group frames by behaviour to ask whether a channel separates behaviours *within* a robot before
asking whether it crosses robots, and a label recoverable only by parsing a directory name is one
refactor away from being lost. Each clip therefore gains:

    condition     'speed_c7.1', 'turn_s0.29', 'side_R_c6', ...
    behaviour     'speed' | 'turn' | 'side'   -- the axis, for the 4/4/4 balance check
    level         index of this condition within its axis, 0-3

Episode numbers are `axis_index * 1000 + condition_index * 100 + clip_index`, which keeps them
unique, keeps the condition recoverable arithmetically, and leaves room to pair across embodiments
later: the same condition slot on the two robots gets the same number, so if `lambda_cross` is ever
turned on it pairs matched behaviours rather than matched filenames. It is 0.0 in every Stage 2 run
today (F71's collection notes), so nothing depends on that yet.

  .venv/bin/python3 scripts/dataset/merge_behaviour_dirs.py \\
      --src data/beh12_hex --out data/beh12_c10f10t10_flat --embodiment hexapod
"""
import argparse
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AXES = ("speed", "turn", "side")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of per-condition subdirectories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embodiment", required=True)
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{args.out} already has files; remove it rather than mixing runs")
    os.makedirs(out, exist_ok=True)

    conditions = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    per_axis = {a: [c for c in conditions if c.startswith(a)] for a in AXES}
    counts = {a: len(v) for a, v in per_axis.items()}
    if len(set(counts.values())) != 1:
        # the balance is the point of the design, so an uneven set is an error rather than a warning
        raise SystemExit(f"axes are not balanced: {counts}")

    total, per_condition = 0, {}
    for axis_i, axis in enumerate(AXES):
        for cond_i, cond in enumerate(per_axis[axis]):
            paths = sorted(p for p in glob.glob(os.path.join(src, cond, "*.npz"))
                           if "manifest" not in os.path.basename(p))
            for clip_i, path in enumerate(paths):
                with np.load(path, allow_pickle=True) as clip:
                    data = {k: clip[k] for k in clip.files}
                episode = axis_i * 1000 + cond_i * 100 + clip_i
                data["expert_episode"] = np.array(episode)
                data["condition"] = np.array(cond)
                data["behaviour"] = np.array(axis)
                data["level"] = np.array(cond_i)
                data["embodiment"] = np.array(args.embodiment)
                np.savez_compressed(
                    os.path.join(out, f"{args.embodiment}_ep{episode}.npz"), **data)
                total += 1
            per_condition[cond] = len(paths)

    print(f"{total} clips -> {args.out}")
    for axis in AXES:
        row = "  ".join(f"{c}:{per_condition[c]}" for c in per_axis[axis])
        print(f"  {axis:<6} {row}")
    n = set(per_condition.values())
    print(f"clips per condition: {sorted(n)}"
          + ("" if len(n) == 1 else "   <- uneven, the 4/4/4 balance is broken"))


if __name__ == "__main__":
    main()
