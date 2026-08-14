"""What the contact labels look like per body, and whether the threshold that makes them is sound.

The expert foot trajectory comes from a **real stick insect**, so it is a variable wave-like gait
rather than a clean alternating tripod, and it should not be scored against one. Per-leg contact
periods on the reference body come out at 6, 22, 18, 4, 9 and 20 frames -- an animal, not a
central pattern generator. Every body follows the same Cartesian targets, so all of them inherit
the same variability and comparisons between them stay fair.

The tripod separation printed below is therefore **descriptive, not a quality bar**: within-tripod
contact agreement minus across-tripod. It is near zero for every body, which is the expected
reading for a wave gait. It is kept only because one body separates from the rest on it --
`c10f10t06`, the large-dead-zone body, is the only negative value, meaning its legs group *worse*
than arbitrary triples. That is one more mark against a body already known to veer.

What does have to be sound is the threshold, because stance fraction is the shared phase label in
the cross-embodiment probe and the phase term in the latent variance decomposition. Measured, the
force distribution is sharply bimodal -- a swing mode near 0.1 N, a stance mode near 6-10 N -- and
the 0.27 N cut sits in the empty valley between them with 1.8% of samples within +/-0.07 N of it.
The labels are not a thresholding artefact.

Two figures:

  gait_diagram_quality.png   contact raster per body, with duty factor per leg
  foot_force_threshold.png   the raw forces the contacts are cut from, against the threshold, on a
                             log scale because they span 0.04 to 20 N

  .venv/bin/python3 scripts/plot_gait_quality.py
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
from wm.bodies import CONTACT_THRESHOLD  # noqa: E402

BASE_FEMUR_M, BASE_TIBIA_M = 0.3429, 0.4139
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
TRIPOD_A, TRIPOD_B = [0, 4, 2], [3, 1, 5]      # FL+MR+HL against FR+ML+HR

BODIES = [
    ("c10f10t10", "reference, tibia longer"),
    ("c10f10t08", "femur longer, safe dead zone"),
    ("c10f08t06", "femur longer, safe dead zone"),
    ("c10f10t06", "femur longer, large dead zone"),
]


def geometry(body):
    femur = BASE_FEMUR_M * int(body[4:6]) / 10
    tibia = BASE_TIBIA_M * int(body[7:9]) / 10
    return abs(femur - tibia) * 1000, femur / tibia


def tripod_separation(contact):
    """Within-tripod agreement minus across-tripod agreement.

    Near zero is the expected reading for a wave gait and is not a defect. Reported because the
    large-dead-zone body is the only one to go negative -- its legs group worse than arbitrary
    triples -- which separates it from bodies that walk.
    """
    within = np.mean([(contact[:, i] == contact[:, j]).mean()
                      for tripod in (TRIPOD_A, TRIPOD_B)
                      for i in tripod for j in tripod if i < j])
    across = np.mean([(contact[:, i] == contact[:, j]).mean()
                      for i in TRIPOD_A for j in TRIPOD_B])
    return within - across


def load(data_dir, body, episodes):
    forces = [np.load(p)["forces"] for p in
              sorted(glob.glob(os.path.join(data_dir, f"{body}_ep*.npz")))[:episodes]]
    return np.concatenate(forces)


def gait_figure(forces, out, frames):
    fig, axes = plt.subplots(len(BODIES), 1, figsize=(12, 2.05 * len(BODIES)), sharex=True)
    for ax, (body, note) in zip(axes, BODIES):
        contact = forces[body][:frames] > CONTACT_THRESHOLD
        dead, ratio = geometry(body)
        ax.imshow(contact.T, aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=[0, frames, 6, 0], vmin=0, vmax=1)
        ax.set_yticks(np.arange(6) + 0.5)
        ax.set_yticklabels([f"{leg}  {contact[:, i].mean():.2f}"
                            for i, leg in enumerate(LEGS)], fontsize=8)
        # the two tripods, marked so a reader can check the alternation by eye
        for i in TRIPOD_A:
            ax.get_yticklabels()[i].set_color("#c0392b")
        for i in TRIPOD_B:
            ax.get_yticklabels()[i].set_color("#2471a3")
        separation = tripod_separation(forces[body] > CONTACT_THRESHOLD)
        ax.set_ylabel(f"{body}\nf/t {ratio:.2f}, dead {dead:.0f} mm", fontsize=8.5)
        ax.set_title(f"{note}     tripod separation {separation:+.3f}", fontsize=9, loc="left")
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color("#c0392b" if dead > 92.5 else "#2471a3")
    axes[-1].set_xlabel("frame at 20 Hz", fontsize=9)
    fig.suptitle("Expert foot targets come from a real stick insect: a variable wave gait, not a "
                 "clean tripod\n"
                 "every body follows the same targets, so all inherit the same variability. "
                 "black = foot loaded, number beside each leg is its duty factor", fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"-> {out}")


def force_figure(forces, out, body, frames):
    raw = forces[body][:frames]
    fig, (traces, hist) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.6, 1]})

    for i, leg in enumerate(LEGS):
        traces.semilogy(raw[:, i], lw=1.0, label=leg, alpha=.85)
    traces.axhline(CONTACT_THRESHOLD, color="#c0392b", ls="--", lw=1.8)
    traces.text(1, CONTACT_THRESHOLD * 1.25, f"threshold {CONTACT_THRESHOLD} N",
                color="#c0392b", fontsize=9)
    traces.set_xlabel("frame at 20 Hz", fontsize=9)
    traces.set_ylabel("foot contact force, N (log)", fontsize=9)
    traces.set_title(f"{body}: the forces the contact labels are cut from", fontsize=10)
    traces.legend(fontsize=8, ncol=6, loc="upper center")
    traces.grid(alpha=.25)

    everything = forces[body].ravel()
    hist.hist(np.log10(everything), bins=70, color="#7f8c8d")
    hist.axvline(np.log10(CONTACT_THRESHOLD), color="#c0392b", ls="--", lw=1.8)
    near = float((np.abs(everything - CONTACT_THRESHOLD) < 0.07).mean())
    hist.set_xlabel("log10 force, N", fontsize=9)
    hist.set_ylabel("samples", fontsize=9)
    hist.set_title(f"all legs, all episodes\n{near*100:.1f}% within +/-0.07 N of the cut",
                   fontsize=10)

    fig.suptitle("Does the threshold decide the gait? A cut sitting in a gap is a decision the "
                 "data supports; one sitting in a peak is not.", fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data", "ik_walk_bracket"))
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "wm", "gait"))
    args = ap.parse_args()

    forces = {body: load(args.data_dir, body, args.episodes) for body, _ in BODIES}
    os.makedirs(args.out_dir, exist_ok=True)
    frames = min(args.frames, min(len(v) for v in forces.values()))

    gait_figure(forces, os.path.join(args.out_dir, "gait_diagram_quality.png"), frames)
    force_figure(forces, os.path.join(args.out_dir, "foot_force_threshold.png"),
                 BODIES[0][0], frames)

    print()
    for body, note in BODIES:
        contact = forces[body] > CONTACT_THRESHOLD
        dead, ratio = geometry(body)
        print(f"  {body}  f/t {ratio:.2f}  dead {dead:5.1f} mm  "
              f"tripod separation {tripod_separation(contact):+.3f}  "
              f"stance {contact.mean(axis=1).mean():.3f}  {note}")


if __name__ == "__main__":
    main()
