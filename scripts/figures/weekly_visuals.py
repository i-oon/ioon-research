"""Visuals for report/weekly_update.md, built from existing data and results (no simulation, no training).

    .venv/bin/python3 scripts/figures/weekly_visuals.py        -> results/deck/weekly/

  bodies_<behaviour>.mp4 / .png   c10 | c08 | B1 doing the same behaviour; third-person (top), ego (bottom)
  action_visibility.png           Pearson r of Froude read from real frames vs steps ahead k
  readout.png                     Pearson r of Froude read by each route (B1, stride 5)
  reads_vs_shapes.png             normalised score, Froude head reads z vs shapes z, 9 tests x 2 seeds
  physics_summary.png             physics closed-loop error, every pretraining, B1 and c08
  physics_<goal>.mp4              hexapod goal (third-person) | c08 physics ego, reads z | shapes z,
                                  with goal and achieved forward / yaw Froude burned in
Numbers in the plots are the ones in the update's tables (copied here, single source for the figures).
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from wm.policy.planner import condition_of  # noqa: E402

OUT = os.path.join(ROOT, "results/deck/weekly")
READS, SHAPES, NEUTRAL, RANDOM = "#5b7fa6", "#d9822b", "#8c8c8c", "#b0b0b0"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def label(img, text, scale=0.5):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_mp4(path, frames, fps=10):
    """H.264 / yuv420p (plays in VS Code and browsers; OpenCV's mp4v does not)."""
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=8, ffmpeg_params=["-movflags", "+faststart"]) as w:
        for f in frames:
            w.append_data(np.ascontiguousarray(f))


def first_clip(d, cond):
    for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
        if condition_of(p) == cond:
            return np.load(p, allow_pickle=True)["frames"]
    return None


def bodies():
    sets = {"forward": ("speed_c8.8", "speed_c8.8", "speed_vx0.50"),
            "turn": ("turn_s0.56", "turn_s0.56", "turn_w0.075"),
            "sideways": ("side_L_lvl1", "side_L_lvl1", "side_L_lvl1")}
    names = ("hexapod c10", "hexapod c08", "B1 quadruped")
    allo = ("data/allocentric/beh12_c10f10t10_flat", "data/allocentric/beh12_c08f09t09_flat",
            "data/allocentric/beh12_b1_flat")
    ego = ("data/egocentric/beh12_c10f10t10_ego_flat", "data/egocentric/beh12_c08f09t09_ego_flat",
           "data/egocentric/beh12_b1_ego_flat")
    for tag, conds in sets.items():
        A = [first_clip(d, c) for d, c in zip(allo, conds)]
        E = [first_clip(d, c) for d, c in zip(ego, conds)]
        n = min(len(x) for x in A + [e for e in E if e is not None])
        frames = []
        for t in range(n):
            top = np.hstack([label(A[i][t], f"{names[i]}  third-person") for i in range(3)])
            rows = [top]
            if all(e is not None for e in E):
                rows.append(np.hstack([label(E[i][t], f"{names[i]}  egocentric") for i in range(3)]))
            frames.append(np.vstack(rows))
        write_mp4(os.path.join(OUT, f"bodies_{tag}.mp4"), frames)
        cv2.imwrite(os.path.join(OUT, f"bodies_{tag}.png"), cv2.cvtColor(frames[n // 2], cv2.COLOR_RGB2BGR))


def action_visibility():
    k = [1, 2, 5, 11]
    r = {"forward": [0.20, 0.28, 0.53, 0.60], "lateral": [0.15, 0.26, 0.62, 0.77], "yaw": [0.15, 0.19, 0.44, 0.77]}
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for (ch, v), c in zip(r.items(), ("#1b5e20", "#0d47a1", "#e65100")):
        ax.plot(k, v, marker="o", lw=2, ms=6, color=c, label=ch)
    ax.axvline(5, color=NEUTRAL, ls=":", lw=1)
    ax.text(5.2, 0.12, "stride 5", color=NEUTRAL)
    ax.set(xlabel="steps ahead k", ylabel="Pearson r (Froude read from real frames)",
           title="How far ahead an action becomes visible", ylim=(0, 1), xticks=k)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "action_visibility.png"), dpi=150); plt.close(fig)


def readout():
    # current pipeline (Froude head shapes z, no joint-command decoder), hexapod c10 (pretrained body),
    # mean of seeds S0 / S1 (F288)
    rows = [("direct: body(proj(a))", (0.99, 0.92, 0.98)),
            ("ITM on the candidate's own recorded frames", (0.95, 0.74, 0.74)),
            ("ITM on the real future from the shared start (= a perfect FTM)", (0.18, 0.40, 0.28)),
            ("rollout: ITM on the FTM prediction", (0.25, 0.36, 0.60))]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(rows))
    for j, (ch, c) in enumerate(zip(("forward", "lateral", "yaw"), ("#1b5e20", "#0d47a1", "#e65100"))):
        ax.barh(y + (j - 1) * 0.26, [r[1][j] for r in rows], height=0.24, color=c, label=ch)
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis()
    ax.set(xlabel="Pearson r across 24 actions (read Froude vs true)", xlim=(0, 1),
           title="Reading Froude from each route (hexapod c10, current pipeline, mean of 2 seeds)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "readout.png"), dpi=150); plt.close(fig)


def reads_vs_shapes():
    tests = ["c10 direct", "c10 rollout (lib)", "c10 rollout (cur)", "c08 direct", "c08 rollout (lib)",
             "c08 rollout (cur)", "B1 direct", "B1 rollout (lib)", "B1 rollout (cur)"]
    reads = [(0.80, 0.73), (0.60, 0.42), (0.49, 0.52), (0.61, 0.55), (0.38, 0.26), (0.39, 0.30),
             (0.82, 0.59), (0.45, 0.00), (0.40, 0.21)]
    shapes = [(0.91, 0.90), (0.72, 0.73), (0.64, 0.59), (0.80, 0.79), (0.47, 0.54), (0.52, 0.43),
              (0.82, 0.82), (0.62, 0.44), (0.46, 0.28)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(tests))
    for i in x:
        ax.plot([i - 0.12, i + 0.12], [np.mean(reads[i]), np.mean(shapes[i])], color=RANDOM, lw=1, zorder=1)
    for vals, off, c, lab in ((reads, -0.12, READS, "Froude head reads z"),
                             (shapes, 0.12, SHAPES, "Froude head shapes z")):
        for s in range(2):
            ax.scatter(x + off, [v[s] for v in vals], s=40, color=c, edgecolor="white", linewidth=1.5,
                       zorder=2, label=lab if s == 0 else None, marker="o" if s == 0 else "s")
    ax.set_xticks(x, tests, rotation=35, ha="right")
    ax.set(ylabel="normalised score (0 = random, 1 = oracle)", ylim=(-0.05, 1),
           title="Letting the Froude head shape z (circles S0, squares S1; w = 11)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "reads_vs_shapes.png"), dpi=150); plt.close(fig)


def physics_summary():
    # current pipeline only: with / without the joint-command decoder, seeds S0 / S1
    runs = [("with joint-command decoder", (0.072, 0.073), (0.073, 0.098), (0.053, 0.050), (0.071, 0.078)),
            ("without (current)", (0.061, 0.063), (0.081, 0.108), (0.047, 0.045), (0.085, 0.055))]
    cols = ["#9ab3cf", SHAPES]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, (body, di, ri, rnd) in zip(axes, (("B1", 1, 2, 0.089), ("c08", 3, 4, 0.126))):
        for k, run in enumerate(runs):
            for m, (idx, hatch) in enumerate(((di, None), (ri, "//"))):
                v = run[idx]
                xpos = m * 3 + k
                ax.bar(xpos, np.mean(v), color=cols[k], hatch=hatch, edgecolor="white", width=0.85,
                       label=run[0] if (m == 0 and body == "B1") else None)
                ax.scatter([xpos, xpos], v, color="black", s=8, zorder=3)
        ax.axhline(rnd, color="black", ls="--", lw=1)
        ax.text(-0.45, rnd + 0.002, "random", ha="left", fontsize=8)
        ax.set_xticks([0.5, 3.5], ["direct", "rollout (hatched)"])
        ax.set_title(f"{body}: physics closed loop")
    axes[0].set_ylabel("error: mean L2, achieved vs goal Froude\n(lower is better; dots = seeds)")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "physics_summary.png"), dpi=150)
    plt.close(fig)


YLIM = {0: (-0.05, 0.25), 2: (-0.05, 0.12)}   # same axes in every physics clip (forward, yaw Froude)


def physics_clip(goal_ep, tag, mech="direct"):
    """Hexapod goal (third-person) | c08 in physics, current pipeline (ego), with Froude traces."""
    base = os.path.join(ROOT, "results/wm/closed_loop/physics")
    run = np.load(f"{base}/FM0/c08/{mech}_w11/c08f09t09_hexapod_{goal_ep}.npz", allow_pickle=True)
    goal = np.load(os.path.join(ROOT, f"data/allocentric/beh12_c10f10t10_flat/hexapod_{goal_ep}.npz"),
                   allow_pickle=True)["frames"]
    goal_ego = np.load(os.path.join(ROOT, f"data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/hexapod_{goal_ep}.npz"),
                       allow_pickle=True)["frames"]
    n = min(len(run["frames"]), len(goal), len(run["achieved_froude"]))
    g = run["goal_froude_t"][:n]
    frames = []
    for t in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(7.68, 2.2), dpi=100)
        for ax, ch, nm in zip(axes, (0, 2), ("forward", "yaw")):
            ax.plot(g[:, ch], color="black", lw=2, label="goal (hexapod c10)")
            ax.plot(run["achieved_froude"][:t + 1, ch], color=SHAPES, lw=2, label="c08 in physics")
            ax.set_xlim(0, 65); ax.set_ylim(*YLIM[ch]); ax.set_title(f"{nm} Froude", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]; plt.close(fig)
        top = np.hstack([label(goal[t], "goal: hexapod c10"), label(goal_ego[t], "goal: its ego view"),
                         label(run["frames"][t], f"c08 in physics (ego), {mech}")])
        plot = cv2.resize(plot, (top.shape[1], int(plot.shape[0] * top.shape[1] / plot.shape[1])))
        frames.append(np.vstack([top, plot]))
    sfx = "" if mech == "direct" else f"_{mech}"
    write_mp4(os.path.join(OUT, f"physics_{tag}{sfx}.mp4"), frames)
    cv2.imwrite(os.path.join(OUT, f"physics_{tag}{sfx}.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))


def b1_physics_clip(goal_ep, tag, mech="direct", base="results/wm/closed_loop/physics/FM0/b1", window=11,
                    goal_tp="data/allocentric/beh12_c10f10t10_flat/hexapod_{ep}.npz", out=None):
    """Hexapod goal (third-person) | B1 walking under its own policy in physics (third-person, re-rendered
    from the saved MuJoCo states) | the B1's ego view the planner read, with Froude traces.
    base / window / goal_tp (goal third-person clip, {ep} = goal_ep) / out: defaults = the original figure."""
    base = os.path.join(ROOT, base)
    run = np.load(f"{base}/{mech}_w{window}/b1_hexapod_{goal_ep}.npz", allow_pickle=True)
    tp = "thirdperson" if mech == "direct" else f"thirdperson_{mech}"
    third = np.load(f"{base}/{tp}_b1_hexapod_{goal_ep}.npz", allow_pickle=True)["frames"]
    goal = np.load(os.path.join(ROOT, goal_tp.format(ep=goal_ep)), allow_pickle=True)["frames"]
    n = min(len(run["frames"]), len(third), len(goal), len(run["achieved_froude"]))
    g = run["goal_froude_t"][:n]
    frames = []
    for t in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(7.68, 2.2), dpi=100)
        for ax, ch, nm in zip(axes, (0, 2), ("forward", "yaw")):
            ax.plot(g[:, ch], color="black", lw=2, label="goal (hexapod c10)")
            ax.plot(run["achieved_froude"][:t + 1, ch], color=SHAPES, lw=2, label="B1 in physics")
            ax.set_xlim(0, 65); ax.set_ylim(*YLIM[ch]); ax.set_title(f"{nm} Froude", fontsize=9)
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]; plt.close(fig)
        top = np.hstack([label(goal[t], "goal: hexapod c10"), label(third[t], f"B1 in physics (policy), {mech}"),
                         label(run["frames"][t], "B1 ego view (planner input)")])
        plot = cv2.resize(plot, (top.shape[1], int(plot.shape[0] * top.shape[1] / plot.shape[1])))
        frames.append(np.vstack([top, plot]))
    sfx = "" if mech == "direct" else f"_{mech}"
    od = os.path.join(ROOT, out) if out else OUT
    write_mp4(os.path.join(od, f"b1_physics_{tag}{sfx}.mp4"), frames)
    cv2.imwrite(os.path.join(od, f"b1_physics_{tag}{sfx}.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    readout(); physics_summary()
    physics_clip("hexapod_ep1303".split("_")[1], "turn")
    physics_clip("hexapod_ep302".split("_")[1], "forward")
    b1_physics_clip("ep1303", "turn"); b1_physics_clip("ep302", "forward")
    physics_clip("ep1303", "turn", "rollout"); physics_clip("ep302", "forward", "rollout")
    b1_physics_clip("ep1303", "turn", "rollout"); b1_physics_clip("ep302", "forward", "rollout")
    print("->", os.path.relpath(OUT, ROOT))
