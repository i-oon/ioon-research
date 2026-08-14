"""Do the two femur-to-tibia groups actually walk differently, or is the split only in the latent?

Forcing the hexapod frames into two clusters splits them perfectly by whether the femur is longer
than the tibia -- cleanly in the learned latent, weakly in the frozen encoder, and not at all by
coxa scale or episode. That is a claim about the representation. This checks the behaviour it is
supposed to be a representation of.

Two outputs, from frames and forces already recorded, so no simulator is involved:

  ratio_gaits.mp4   the bodies side by side on the same expert episode, ordered by femur/tibia
                    ratio, with a gap marking the boundary between the groups
  ratio_gaits.png   contact pattern per leg over time, same order, plus stance fraction

All bodies walk the *same* expert episode, so anything that differs between panels is the geometry
responding to an identical intent, not a different intent. That is the whole point of the IK
dataset and it is what makes the comparison fair.

Read the figure before believing the video: a gait looks different on screen whenever the legs are
different lengths, whether or not the contact timing changed at all.

  .venv/bin/python3 scripts/compare_ratio_gaits.py
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
from wm.bodies import CONTACT_THRESHOLD  # noqa: E402

# from sim/scene/make_leg_morphology.py -- the unscaled segment lengths of the base insect
BASE_FEMUR_M = 0.3429
BASE_TIBIA_M = 0.4139

# Deliberately includes c10f10t06 and c06f10t06, which wm.bodies.EXCLUDED_BODIES bars from
# training. This script exists to show what a femur longer than its tibia does to the gait, and
# those two are the demonstration -- excluding them would remove the subject. Every other script
# derives its bodies from wm.bodies.bodies_in; this one must not.
BODIES = ["c10f10t10", "c06f10t10", "c10f10t06", "c06f10t06", "c10f06t06"]
LEG_NAMES = ["FL", "ML", "HL", "FR", "MR", "HR"]


def ratio(body):
    """femur / tibia for a body named c{coxa}f{femur}t{tibia}, scales in tenths."""
    femur = BASE_FEMUR_M * int(body[4:6]) / 10
    tibia = BASE_TIBIA_M * int(body[7:9]) / 10
    return femur / tibia


def label(frame, text, height=22):
    """A caption strip above a frame, drawn as pixels so no font dependency is needed."""
    fig = plt.figure(figsize=(frame.shape[1] / 100, height / 100), dpi=100)
    fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=8)
    fig.canvas.draw()
    strip = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return np.concatenate([strip, frame], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=6)
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "ik_walk_8body"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "wm", "gait"))
    args = ap.parse_args()

    # ordered by ratio so the boundary is a position in the row, not something to hunt for
    order = sorted(BODIES, key=ratio)
    clips = {}
    for body in order:
        path = os.path.join(args.data_dir, f"{body}_ep{args.episode}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        data = np.load(path)
        clips[body] = (data["frames"], data["forces"] > CONTACT_THRESHOLD)

    print(f"episode {args.episode}, ordered by femur/tibia:")
    for body in order:
        femur = BASE_FEMUR_M * int(body[4:6]) / 10
        tibia = BASE_TIBIA_M * int(body[7:9]) / 10
        side = "femur longer" if ratio(body) > 1 else "tibia longer"
        print(f"  {body}  femur {femur:.3f} m  tibia {tibia:.3f} m  "
              f"ratio {ratio(body):.2f}  {side}")

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(len(clips[b][0]) for b in order)

    video = os.path.join(args.out_dir, f"ratio_gaits_ep{args.episode}.mp4")
    writer = imageio.get_writer(video, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    height = clips[order[0]][0].shape[1]
    thin = np.full((height + 22, 4, 3), 235, np.uint8)
    thick = np.full((height + 22, 18, 3), 40, np.uint8)      # marks the group boundary
    for t in range(n):
        panels = []
        for i, body in enumerate(order):
            if i:
                crossed = (ratio(order[i - 1]) < 1) != (ratio(body) < 1)
                panels.append(thick if crossed else thin)
            panels.append(label(clips[body][0][t], f"{body}   f/t {ratio(body):.2f}"))
        writer.append_data(np.concatenate(panels, axis=1))
    writer.close()
    print(f"\n-> {video}  ({n} frames)")

    fig, axes = plt.subplots(len(order), 1, figsize=(11, 1.5 * len(order)), sharex=True)
    for ax, body in zip(axes, order):
        contact = clips[body][1][:n]
        ax.imshow(contact.T, aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=[0, n, 6, 0])
        ax.set_yticks(np.arange(6) + 0.5)
        ax.set_yticklabels(LEG_NAMES, fontsize=7)
        stance = contact.mean(axis=1)
        side = "femur longer" if ratio(body) > 1 else "tibia longer"
        ax.set_ylabel(f"{body}\nf/t {ratio(body):.2f}", fontsize=8)
        ax.set_title(f"{side}   stance fraction {stance.mean():.3f} "
                     f"+/- {stance.std():.3f}", fontsize=8, loc="left")
        for spine in ax.spines.values():
            spine.set_color("#c0392b" if ratio(body) > 1 else "#2471a3")
            spine.set_linewidth(2)
    axes[-1].set_xlabel("frame", fontsize=9)
    fig.suptitle(f"Same expert episode {args.episode}, ordered by femur/tibia ratio\n"
                 "red = femur longer, blue = tibia longer; black means the foot is loaded",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    figure = os.path.join(args.out_dir, f"ratio_gaits_ep{args.episode}.png")
    plt.savefig(figure, dpi=140)
    print(f"-> {figure}")

    print("\nstance fraction, which is the one contact statistic both groups share:")
    for body in order:
        stance = clips[body][1][:n].mean(axis=1)
        print(f"  {body}  ratio {ratio(body):.2f}  mean {stance.mean():.3f}  "
              f"spread {stance.std():.3f}")


if __name__ == "__main__":
    main()
