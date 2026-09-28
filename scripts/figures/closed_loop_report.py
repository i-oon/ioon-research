"""Figures and 4-panel clips for the rendered closed loop (F255): direct vs rollout (control) vs
rollout (cycle), per goal condition, from the npz files `close_loop_direct_froude.py
--goal_timevarying` saves.

    .venv/bin/python3 scripts/figures/closed_loop_report.py

    <out>/<condition>.png            goal vs achieved Froude over time (forward / lateral / yaw)
    <out>/<condition>_<run>.mp4      2x2: goal ego | B1 ego (the frames the planner read) /
                                     goal allo | B1 allo, Froude and running error burned in
    <out>/summary.png                mean error per condition, every run, with library bounds

Achieved Froude is the executed candidate's recorded body_motion at the index it was posed from,
the same grading `selection_eval.py` uses. Frames are aligned by index t on both bodies, the same
alignment the error is graded on. KINEMATIC loop: the B1 is posed from recorded motion and cannot
fall -- the clips show what was SELECTED from real rendered observations, not physical control.
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

GOALS = {"turn_s0.29": "hexapod_ep1202", "turn_s0.56": "hexapod_ep1303",
         "side_L_lvl0": "hexapod_ep2000", "side_R_lvl1": "hexapod_ep2301",
         "speed_c7.1": "hexapod_ep103", "speed_c8.8": "hexapod_ep302"}
# run label -> (subdirectory of --loop_dir, mechanism prefix, colour, line style)
RUNS = {"direct (cycle_fz, w11)": ("direct_cycle_fz_w11", "direct", "#2e7d32", "-"),
        "rollout control (ctrl_fz, w11)": ("ctrl_fz_w11", "rollout", "#c62828", "-."),
        "rollout + cycle (cycle_fz, w11)": ("cycle_fz_w11", "rollout", "#1565c0", "-")}
LIB = (0.0310, 0.1264)   # oracle, random -- selection_eval.py, same six goals and library
CH = ("forward", "lateral", "yaw")


def load_run(loop_dir, sub, mech, goal):
    hits = glob.glob(os.path.join(loop_dir, sub, f"{mech}-physics_{goal}_*.npz"))
    return np.load(hits[0], allow_pickle=True) if hits else None


def plot_condition(cond, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    first = next(iter(runs.values()))
    goal = first["goal_froude_t"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    title = []
    for label, d in runs.items():
        err = np.linalg.norm(d["achieved_froude"] - d["goal_froude_t"], axis=1).mean()
        title.append(f"{label.split(' (')[0]} {err:.3f}")
    for j, ax in enumerate(axes):
        ax.plot(goal[:, j], color="black", lw=2, label="goal (hexapod, physics)")
        for label, d in runs.items():
            _, _, col, ls = RUNS[label]
            ax.plot(d["achieved_froude"][:, j], color=col, ls=ls, lw=1.6, label=f"achieved -- {label}")
        ax.set_ylabel(CH[j])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("timestep (decide every 2 steps, rendered ego frame each decision)")
    fig.suptitle(f"Rendered closed loop, B1 <- hexapod goal  |  goal={cond}\n"
                 f"mean L2 error: " + "   ".join(title), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def summary_plot(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(table)
    labels = list(RUNS)
    x = np.arange(len(conds))
    wd = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for k, label in enumerate(labels):
        vals = [table[c].get(label, np.nan) for c in conds]
        bars = ax.bar(x + (k - (len(labels) - 1) / 2) * wd, vals, wd * 0.92, color=RUNS[label][2],
                      label=f"{label}  mean {np.nanmean(vals):.3f}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(LIB[1], color="grey", ls="--", lw=1)
    ax.text(len(conds) - 0.45, LIB[1] + 0.003, "random pick", color="grey", fontsize=8, ha="right")
    ax.axhline(LIB[0], color="grey", ls=":", lw=1)
    ax.text(len(conds) - 0.45, LIB[0] + 0.003, "library oracle", color="grey", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("mean L2 error to goal Froude (lower is better)")
    ax.set_title("Rendered closed loop (CoppeliaSim ego camera, kinematic), six hexapod goals, B1 library")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fmt(v):
    return f"fwd {v[0]:+.3f} lat {v[1]:+.3f} yaw {v[2]:+.3f}"


def make_video(cond, label, d, goal_ego, goal_allo, out, fps=10):
    import cv2
    P, BAR = 256, 22

    def tile(img, head, sub, head_bg=(40, 40, 40), sub_col=(120, 230, 120)):
        t = np.full((BAR * 2 + P, P, 3), 0, np.uint8)
        t[:BAR] = head_bg
        cv2.putText(t, head, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        t[BAR:2 * BAR] = (20, 20, 20)
        cv2.putText(t, sub, (6, BAR + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, sub_col, 1, cv2.LINE_AA)
        t[2 * BAR:] = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        return t

    n = len(d["ego_frames"])
    ach, gt = d["achieved_froude"], d["goal_froude_t"]
    err = np.linalg.norm(ach - gt, axis=1)
    chosen = d["chosen"]
    video = []
    for t in range(n):
        gi = min(t, len(goal_ego) - 1)
        top = np.hstack([tile(goal_ego[gi], "GOAL - hexapod (ego)", fmt(gt[t])),
                         tile(d["ego_frames"][t], "B1 (ego) - what the planner saw", fmt(ach[t]))])
        bot = np.hstack([tile(goal_allo[min(t, len(goal_allo) - 1)], "GOAL - hexapod (allo)",
                              f"condition {cond}", sub_col=(200, 200, 200)),
                         tile(d["frames"][t], "B1 (allo) - KINEMATIC, cannot fall",
                              f"picked {chosen[t]}", sub_col=(200, 200, 200))])
        foot = np.full((50, 2 * P, 3), 17, np.uint8)
        cv2.putText(foot, f"{label}   step {t}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(foot, f"|achieved-goal| {err[t]:.3f}   running mean {err[:t + 1].mean():.3f}"
                          f"   (library oracle {LIB[0]:.3f}, random {LIB[1]:.3f})",
                    (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 220, 255), 1, cv2.LINE_AA)
        video.append(cv2.cvtColor(np.vstack([top, bot, foot]), cv2.COLOR_BGR2RGB))
    import imageio.v2 as imageio          # H.264, as the deck's other clips (cv2's mp4v will not play in a browser)
    imageio.mimwrite(out, np.stack(video), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop_dir", default="results/wm/closed_loop/cycle_live_fixed")
    ap.add_argument("--goal_ego_dir", default="data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout")
    ap.add_argument("--goal_allo_dir", default="data/allocentric/beh12_c10f10t10_flat")
    ap.add_argument("--video_runs", nargs="*", default=["rollout control (ctrl_fz, w11)",
                                                        "rollout + cycle (cycle_fz, w11)"])
    ap.add_argument("--out", default="results/deck/cycle_closed_loop")
    ap.add_argument("--preset", choices=("cycle", "stride"), default="cycle",
                    help="cycle: F257's arms; stride: stride 1 vs stride 5 (F261), out-of-sample fits")
    args = ap.parse_args()
    if args.preset == "stride":
        RUNS.clear()
        RUNS.update({"direct stride 5 (w11)": ("direct_stride5_w11", "direct", "#2e7d32", "-"),
                     "rollout stride 1 (w11)": ("rollout_stride1_w11", "rollout", "#c62828", "-."),
                     "rollout stride 5 (w11)": ("rollout_stride5_w11", "rollout", "#1565c0", "-")})
        args.loop_dir = "results/wm/closed_loop/stride_live"
        args.out = "results/deck/stride_closed_loop"
        args.video_runs = ["rollout stride 1 (w11)", "rollout stride 5 (w11)", "direct stride 5 (w11)"]
    if args.preset == "cycle" and os.path.isdir(os.path.join(ROOT, args.loop_dir, "rollout_cycle_fz_w11")):   # F256 re-run layout
        for label, (sub, mech, col, ls) in list(RUNS.items()):
            RUNS[label] = (sub if sub.startswith("direct_") else f"rollout_{sub}", mech, col, ls)
    loop_dir = os.path.join(ROOT, args.loop_dir)
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)

    table = {}
    for cond, goal in GOALS.items():
        runs = {}
        for label, (sub, mech, _, _) in RUNS.items():
            d = load_run(loop_dir, sub, mech, goal)
            if d is not None:
                runs[label] = d
        if not runs:
            continue
        table[cond] = {k: float(np.linalg.norm(v["achieved_froude"] - v["goal_froude_t"], axis=1).mean())
                       for k, v in runs.items()}
        plot_condition(cond, runs, os.path.join(out, f"{cond}.png"))
        ge = np.load(os.path.join(ROOT, args.goal_ego_dir, goal + ".npz"), allow_pickle=True)
        ga = np.load(os.path.join(ROOT, args.goal_allo_dir, goal + ".npz"), allow_pickle=True)
        assert str(ge["condition"]) == str(ga["condition"]) == cond, (goal, cond)
        for label in args.video_runs:
            if label in runs:
                tag = RUNS[label][0]
                make_video(cond, label, runs[label], ge["frames"], ga["frames"],
                           os.path.join(out, f"{cond}_{tag}.mp4"))
        print(cond, "  ".join(f"{k.split(' (')[0]} {v:.4f}" for k, v in table[cond].items()), flush=True)
    summary_plot(table, os.path.join(out, "summary.png"))
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
