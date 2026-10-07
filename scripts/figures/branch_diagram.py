"""Slide figure: how a branch is made, from the real data.

Top-down centre-of-mass paths of all 24 branches that start from one saved state of one main clip (hexapod c10 and B1):
the 10 frames before the switch are shared (black), the 20 after fan out, coloured by the new command's family; the
clip's own command (no change) is dashed. Paths are expressed in the body frame at the switch (x forward, y left).
Plus the egocentric frames of four branches 1 s after the switch.

    .venv/bin/python3 scripts/figures/branch_diagram.py
"""
import collections
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CW = os.path.join(ROOT, "data/counterfactual_walks")
FAM_COL = {"fwd": "#1565c0", "bwd": "#6a1b9a", "turn_left": "#2e7d32", "turn_right": "#9e9d24",
           "side_L": "#c62828", "side_R": "#ef6c00"}


def yaw_of(q):
    """Yaw of a quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def group(body, src_hint=None):
    d = os.path.join(CW, f"{body}_branches_train")
    G = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            G[(str(z["cf_source"]), int(z["cf_t"]))].append(p)
        if len(G) > 6 and all(len(v) == 24 for v in G.values()):
            break
    key = sorted(k for k, v in G.items() if len(v) == 24)[0]
    return key, G[key]


def panel(ax, axf, body, title):
    key, paths = group(body)
    # body frame at the switch: +x = the direction the clip's OWN command (no change) moves over the 20 frames after it
    # (both example sources walk forward); robust to the camera-quaternion convention
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            if bool(z["cf_own"]):
                com = np.asarray(z["com_pos"])[:, :2]; b = int(z["first_pair"])
                v = com[-1] - com[b]; th = np.arctan2(v[1], v[0])
                sw = (np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]]), com[b])
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            com = np.asarray(z["com_pos"])[:, :2]
            b = int(z["first_pair"])
            q = np.asarray(z["cam_pose"])[b, 3:7]
            fam = str(z["family"])
            own = bool(z["cf_own"])
            frames = z["frames"] if "frames" in z.files else None
            name = str(z["cf_command_name"])
        R, origin = sw
        xy = (com - origin) @ R.T
        ax.plot(xy[:b + 1, 0], xy[:b + 1, 1], color="black", lw=2.2, zorder=3)
        ax.plot(xy[b:, 0], xy[b:, 1], color=FAM_COL.get(fam, "grey"), lw=1.6 if not own else 2.4,
                ls="--" if own else "-", alpha=0.9)
        if frames is not None and name in panel.pick.get(body, ()) and len(axf[body]) < 4:
            axf[body].append((name, frames[min(b + 20, len(frames) - 1)]))
    ax.plot(0, 0, "o", color="black", ms=7, zorder=4)
    ax.set_title(f"{title}: one saved state (black dot), 10 shared frames, then 24 commands for 20 frames (1 s)", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("forward (m, body frame at the switch)"); ax.set_ylabel("left (m)")


panel.pick = {}


def main():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.2, 1])
    axf = {"c10": [], "b1": []}
    for body in ("c10", "b1"):
        key, paths = group(body)
        names = []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                names.append((str(z["family"]), str(z["cf_command_name"]), bool(z["cf_own"])))
        pick = [n for f, n, o in names if o][:1]
        for fam in ("turn_left", "side_R", "bwd"):
            pick += [n for f, n, o in names if f == fam and n not in pick][:1]
        panel.pick[body] = pick
    panel(fig.add_subplot(gs[0, 0:2]), axf, "c10", "hexapod")
    panel(fig.add_subplot(gs[0, 2:4]), axf, "b1", "Unitree B1")
    for i, body in enumerate(("c10", "b1")):
        sub = gs[1, 2 * i:2 * i + 2].subgridspec(1, 4)
        for j, (name, fr) in enumerate(axf[body][:4]):
            a = fig.add_subplot(sub[0, j]); a.imshow(fr); a.set_axis_off(); a.set_title(name, fontsize=7)
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in FAM_COL.items()]
    handles += [plt.Line2D([], [], color="black", lw=2, label="shared prefix"),
                plt.Line2D([], [], color="grey", lw=2, ls="--", label="own command (no change)")]
    fig.suptitle("How a branch is made: one saved state, all 24 commands (bottom: the camera view 1 s after the switch)", y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.965))
    out = os.path.join(ROOT, "results/deck/weekly_1008/branch_diagram.png")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=130)
    print("->", out)


if __name__ == "__main__":
    main()
