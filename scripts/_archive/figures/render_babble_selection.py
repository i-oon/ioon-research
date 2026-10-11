"""Slide-21-style figures and clips for action selection over a B1 motor-babble library.

    .venv/bin/python3 scripts/figures/render_babble_selection.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/ckpt_full_b1_babble_v2_ego_flat.pt \\
        --candidates_dir data/egocentric/b1_babble_v2_ego_flat --tag babble_v2 \\
        --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout

Per goal condition (one hexapod clip each), same direct-scoring methodology as
froude_match_timevarying.py: at every horizon-step t the planner picks the candidate whose
`body_head(proj(a))` is closest to the goal's Froude at t, graded by that candidate's own recorded Froude.

    <tag>_<condition>.png   goal vs picked vs library-oracle Froude (forward / lateral / yaw) over time
    <tag>_<condition>.mp4   2x2 like modeD_direct_vision.mp4: goal ego | picked B1 ego / goal allo | picked B1 allo, Froude burned in
                            (index t on both -- the same alignment the error is graded on)
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load       # noqa: E402
from final_2x2x2_test import build_planner          # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "render"))
from three_panel_loop import panel, strip, _text    # noqa: E402  same panel style as modeD_direct_vision.mp4

JOINT_ORDER = [f"{leg}_{seg}_joint" for leg in ("FR", "FL", "RR", "RL") for seg in ("hip", "thigh", "calf")]
LABELS = ("forward", "lateral", "yaw")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def caption(frame, lines, height=48):
    f_big, f_small = ImageFont.truetype(BOLD, 14), ImageFont.truetype(FONT, 12)
    h, w = frame.shape[:2]
    canvas = Image.new("RGB", (w, h + height), (17, 17, 17))
    canvas.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((6, h + 4), lines[0], font=f_big, fill=(255, 255, 255))
    d.text((6, h + 26), lines[1], font=f_small, fill=(170, 170, 170))
    return np.asarray(canvas)


def _R(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def stitch(raw, picks, steps, H):
    """One continuous body: each step applies the picked clip's body-frame motion increments and joint
    angles for frames t..t+H, integrated from a single start pose (close_loop_direct_froude.py's scheme)."""
    pos, quat = np.array([0.0, 0.0, float(raw[picks[0]]["base_pos"][0, 2])]), np.array([1.0, 0, 0, 0])
    P, Q, J = [], [], []
    for i, t in zip(picks, steps):
        r = raw[i]
        for m in range(t, min(t + H, len(r["joint_pos"]) - 1)):
            q0, q1 = r["base_quat"][m], r["base_quat"][m + 1]
            pos = pos + _R(quat) @ (_R(q0).T @ (r["base_pos"][m + 1] - r["base_pos"][m]))
            dq = _qmul(q0 * np.array([1, -1, -1, -1]), q1)
            quat = _qmul(quat, dq); quat = quat / np.linalg.norm(quat)
            P.append(pos.copy()); Q.append(quat.copy()); J.append(r["joint_pos"][m + 1])
    return np.array(P), np.array(Q), np.array(J)


def render_replay(pos, quat, jpos, tmp, name, ego):
    traj = os.path.join(tmp, name + ".npz")
    np.savez(traj, base_pos=pos, base_quat=quat, joint_pos=jpos, dt=0.02,
             joint_order_sdk=np.array(JOINT_ORDER))
    out = os.path.join(tmp, "ego" if ego else "allo")
    cmd = [sys.executable, os.path.join(ROOT, "sim/render/render_b1_replay.py"), "--scene",
           os.path.join(ROOT, "sim/env/b1_flat.ttt"), "--traj", traj, "--out", out, "--align_yaw"]
    if ego:
        cmd += ["--ego", "--ego_seed", "0"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    return np.load(os.path.join(out, name + ".npz"))["frames"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--goal_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--display_name", default="", help="name shown in the figure title (default: the tag)")
    ap.add_argument("--cond_label", default="", help="condition name shown in the figure (default: the condition id)")
    ap.add_argument("--conditions", default="", help="comma list; empty = all")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--goal_allo_dir", default="data/allocentric/beh12_c10f10t10_flat")
    ap.add_argument("--cand_allo_dir", default="", help="unused since the stitched replay (kept for old calls)")
    ap.add_argument("--ylims", default="", help="fwd_lo,fwd_hi,lat_lo,lat_hi,yaw_lo,yaw_hi: one fixed scale "
                                                "for every plot (match the goal-reading plots); empty = autoscale")
    ap.add_argument("--library_label", default="babble library",
                    help='legend text for the picks, e.g. "expert library" for the ground-truth B1 library')
    ap.add_argument("--plots_only", action="store_true", help="skip the videos (no CoppeliaSim needed)")
    ap.add_argument("--out_dir", default="results/deck")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = args.horizon
    planner = build_planner(args.ckpt, os.path.join(ROOT, args.candidates_dir), "b1", H,
                            free_offset=False, device=args.device, per_condition=999)
    spec, hspec = REGISTRY["b1"], REGISTRY["hexapod"]
    mean, std = planner.mean_s, planner.std_s
    cand_bm = [np.asarray(load(c["path"], spec)["body_motion"])[:, :3] for c in planner.candidates]
    cand_frames = [np.load(c["path"], allow_pickle=True)["frames"] for c in planner.candidates]
    raw = [dict(np.load(c["path"], allow_pickle=True)) for c in planner.candidates]
    names = [os.path.splitext(os.path.basename(c["path"]))[0] for c in planner.candidates]

    goal_by_cond = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            goal_by_cond.setdefault(str(z["condition"]), p)
    want = [c for c in args.conditions.split(",") if c] or list(goal_by_cond)
    out = os.path.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)

    fixed = [float(x) for x in args.ylims.split(",")] if args.ylims else None
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for cond in want:
        gp = goal_by_cond[cond]
        gclip = load(gp, hspec)
        goal = np.asarray(gclip["body_motion"])[:, :3]
        gframes = np.load(gp, allow_pickle=True)["frames"]
        gallo = np.load(os.path.join(ROOT, args.goal_allo_dir, os.path.basename(gp)), allow_pickle=True)["frames"]
        g_std = (goal - mean) / std
        n = len(g_std) - H
        steps, picks, sel, orc = [], [], [], []
        for t in range(0, n, H):
            _, i, _, _ = planner.act(g_std[t], t)
            local = np.array([b[t:t + H].mean(0) if len(b) > t else b[-1] for b in cand_bm])
            best = int(np.linalg.norm(local - goal[t], axis=1).argmin())
            steps.append(t); picks.append(i); sel.append(local[i]); orc.append(local[best])
        steps, sel, orc = np.array(steps), np.array(sel), np.array(orc)
        gs = goal[steps]
        e_sel = np.linalg.norm(sel - gs, axis=1).mean()
        e_orc = np.linalg.norm(orc - gs, axis=1).mean()

        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(steps, gs[:, j], color="black", lw=2, label="goal (hexapod, physics)")
            ax.plot(steps, sel[:, j], color="#2a7a3b", lw=1.8, label=f"picked from {args.library_label}")
            ax.plot(steps, orc[:, j], color="gray", ls=":", lw=1.8, label="best the library could have done")
            ax.set_ylabel(LABELS[j])
        if fixed:
            for j, ax in enumerate(axes):
                ax.set_ylim(fixed[2 * j], fixed[2 * j + 1])
        for arr in (gs, sel, orc):
            lo, hi = np.minimum(lo, arr.min(0)), np.maximum(hi, arr.max(0))
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{args.display_name or args.tag}: direct scoring over the B1 candidate library\n"
                          f"goal: {args.cond_label or cond}  |  mean err: picked={e_sel:.4f}  library best={e_orc:.4f}")
        axes[-1].set_xlabel(f"timestep (horizon={H}, goal read at every step)")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{args.tag}_{cond}.png"), dpi=120)
        plt.close(fig)

        if args.plots_only:
            print(f"{cond:>14}  picked {e_sel:.4f}  library-best {e_orc:.4f}  -> {args.tag}_{cond}.png")
            continue
        # one continuous body re-rendered in ONE room (same as modeD_direct_vision.mp4): the picked clips
        # supply joint angles and motion increments, never pixels, so the wall cannot change between picks
        P, Q, J = stitch(raw, picks, steps, H)
        with tempfile.TemporaryDirectory() as tmp:
            r_ego = render_replay(P, Q, J, tmp, "run", True)
            r_allo = render_replay(P, Q, J, tmp, "run", False)
        frames = []
        for m in range(min(len(r_ego), len(r_allo))):
            k = min(m // H, len(steps) - 1)
            t, i = steps[k], picks[k]
            f = min(t + m % H, len(gframes) - 1)
            err = np.linalg.norm(sel[k] - gs[k])
            run = np.linalg.norm(sel[:k + 1] - gs[:k + 1], axis=1).mean()
            tl = panel(gframes[f], "GOAL - source body (ego)", "goal Froude at this step", (60, 60, 60), gs[k])
            tr = panel(r_ego[m], "NEW BODY (ego)", "re-rendered in one room", (60, 60, 60), sel[k])
            bl = panel(gallo[min(f, len(gallo) - 1)], "GOAL - source body (allo)",
                       "same condition, DIFFERENT take", (90, 60, 0))
            br = panel(r_allo[m], "NEW BODY (allocentric)",
                       "REPLAYED GROUND TRUTH - not control", (0, 0, 140), sel[k])
            grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
            foot = np.full((44, grid.shape[1], 3), 20, np.uint8)
            _text(foot, f"{args.tag}   goal {cond}   frame {f}   picked: {names[i]}",
                  16, (255, 255, 255), 0.40, 1, centre=False)
            _text(foot, f"|pick-goal| {err:.3f}   running mean {run:.3f}   (library best {e_orc:.3f})",
                  36, (160, 220, 160), 0.40, 1, centre=False)
            frames.append(np.vstack([grid, foot]))
        imageio.mimwrite(os.path.join(out, f"{args.tag}_{cond}.mp4"), np.stack(frames).astype(np.uint8),
                         fps=20, macro_block_size=1)
        print(f"{cond:>14}  picked {e_sel:.4f}  library-best {e_orc:.4f}  -> {args.tag}_{cond}.png/.mp4")
    print("DATA RANGE per channel (min,max):", [(round(float(lo[i]), 3), round(float(hi[i]), 3)) for i in range(3)])


if __name__ == "__main__":
    main()
