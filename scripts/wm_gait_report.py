"""Gait comparison between world-model predicted commands and the IK ground truth.

Reads the replay npz written by sim/render_wm_prediction.py, where both command sequences
were driven through the same physics and the same camera, and produces:

  gait_<tag>.png    stance/swing bars per leg, predicted above ground truth. Black is stance
                    (foot loaded). A healthy stick-insect walk shows a tripod: FL/MR/HL in
                    antiphase with FR/ML/HR.
  replay_<tag>.mp4  the two runs side by side, predicted on the left.
  metrics printed   duty factor per leg, tripod antiphase score, distance travelled.

Duty factor is the fraction of the clip a foot is loaded; the IK gait sits near 0.5-0.6.
The tripod score is the correlation between the two tripod groups' contact counts, so -1 is
a textbook alternating tripod and 0 means the groups move independently.

Run from the repository root:
  .venv/bin/python3 scripts/wm_gait_report.py --replay results/wm/replay/<name>.npz
"""
import argparse
import glob
import os

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
TRIPOD_A = [0, 4, 2]   # FL, MR, HL
TRIPOD_B = [3, 1, 5]   # FR, ML, HR
THRESHOLD = 0.5        # newtons; same gate wm/data/dataset.py uses for contact labels


def contacts(forces):
    return forces > THRESHOLD


def tripod_score(down):
    a = down[:, TRIPOD_A].sum(axis=1).astype(float)
    b = down[:, TRIPOD_B].sum(axis=1).astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def draw(ax, down, title):
    for i, leg in enumerate(LEGS):
        stance = down[:, i]
        ax.fill_between(np.arange(len(stance)), i + 0.1, i + 0.9,
                        where=stance, step="post", color="black", linewidth=0)
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels(LEGS, fontsize=9)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, len(down))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def joint_limits(morph):
    """Min and max each joint reaches across every clip of this body, in degrees.

    A command outside this range asks the leg for a pose the body never adopts, which is what
    the videos show as legs folding into the abdomen. It needs no per-frame ground truth, only
    the body's own range, so it also applies to a body with no expert data and across
    embodiments.
    """
    pattern = os.path.join(ROOT, "data", "ik_walk_100_framed", f"{morph}_*.npz")
    actions = np.concatenate([np.degrees(np.load(p)["actions"]) for p in sorted(glob.glob(pattern))])
    return actions.min(axis=0), actions.max(axis=0)


def out_of_range(commands, limits):
    low, high = limits
    outside = (commands < low) | (commands > high)
    excursion = np.maximum(low - commands, 0) + np.maximum(commands - high, 0)
    return float(outside.mean()), float(excursion.max())


def annotate(frame, title, heads, index, scale):
    """Upscale and stamp the caption plus distance travelled so far onto one frame."""
    image = Image.fromarray(frame).convert("RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    step = heads[min(index, len(heads) - 1)] - heads[0]
    bar = 15 * scale // 2
    draw.rectangle([0, 0, image.width, bar], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255))
    draw.rectangle([0, image.height - bar, image.width, image.height], fill=(0, 0, 0))
    draw.text((4, image.height - bar + 2),
              f"step {index:>2}   forward {step[0]:+.3f} m   side {step[1]:+.3f} m",
              fill=(255, 255, 255))
    return np.asarray(image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--label", default="", help="caption for the predicted pass")
    ap.add_argument("--scale", type=int, default=2, help="upscale frames so the caption is legible")
    ap.add_argument("--out_dir", default="results/wm/gait")
    args = ap.parse_args()

    data = np.load(args.replay, allow_pickle=True)
    morph, epoch = str(data["morph"]), int(data["epoch"])
    tag = os.path.splitext(os.path.basename(args.replay))[0]
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    down = {"predicted": contacts(data["pred_forces"]), "ground_truth": contacts(data["gt_forces"])}
    print(f"body '{morph}' ({'held out' if bool(data['held_out']) else 'trained on'}), "
          f"epoch {epoch}, clip {str(data['clip'])}")
    print(f"{'':<14}" + "".join(f"{leg:>7}" for leg in LEGS) + f"{'tripod':>9}{'forward m':>11}")
    for name in ("predicted", "ground_truth"):
        duty = down[name].mean(axis=0)
        heads = data["pred_heads" if name == "predicted" else "gt_heads"]
        travelled = float(np.linalg.norm(heads[-1][:2] - heads[0][:2]))
        print(f"{name:<14}" + "".join(f"{d:>7.2f}" for d in duty)
              + f"{tripod_score(down[name]):>9.2f}{travelled:>11.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    draw(axes[0], down["predicted"], "driven by world-model predicted commands")
    draw(axes[1], down["ground_truth"], "driven by IK ground-truth commands")
    axes[1].set_xlabel("simulation step", fontsize=9)
    fig.suptitle(
        f"Gait on body '{morph}' -- {'held out' if bool(data['held_out']) else 'trained on'}, "
        f"epoch {epoch}\nblack = foot loaded (stance). Both passes: same scene, same physics, "
        "open-loop joint targets.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    gait_path = os.path.join(out_dir, f"gait_{tag}.png")
    fig.savefig(gait_path, dpi=140)
    print(f"-> {gait_path}")

    if "pred_actions" in data:
        limits = joint_limits(morph)
        print(f"\ncommands outside the range body '{morph}' ever uses, over 100 episodes")
        for name, key in (("predicted", "pred_actions"), ("ground_truth", "gt_actions")):
            fraction, worst = out_of_range(np.degrees(data[key]), limits)
            print(f"  {name:<14} {100 * fraction:5.1f}% of commands, worst excursion {worst:.1f} deg")

    pred_frames, gt_frames = data["pred_frames"], data["gt_frames"]
    n = min(len(pred_frames), len(gt_frames))
    label = args.label or ("ridge probe on z" if "probe" in tag else "world model")
    video_path = os.path.join(out_dir, f"replay_{tag}.mp4")
    writer = imageio.get_writer(video_path, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for i in range(n):
        left = annotate(pred_frames[i], label, data["pred_heads"], i, args.scale)
        right = annotate(gt_frames[i], "IK ground truth", data["gt_heads"], i, args.scale)
        gap = np.full((left.shape[0], 6, 3), 255, np.uint8)
        writer.append_data(np.concatenate([left, gap, right], axis=1))
    writer.close()
    print(f"-> {video_path}  (left: {label}, right: IK ground truth, {n} frames)")


if __name__ == "__main__":
    main()
