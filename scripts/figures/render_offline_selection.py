"""Offline action selection clips + figures (NO PHYSICS) from selection_eval.py --dump npz files.

    .venv/bin/python3 scripts/figures/render_offline_selection.py \\
        --recorded results/deck/weekly_1008/sel_dump/recorded.npz \\
        --vision results/deck/weekly_1008/sel_dump/vision.npz \\
        --turn turn_s0.29 --forward speed_c8.8 --out results/deck/weekly_1008

Per goal and goal source: mp4 (goal hexapod ego | B1 ego frames of the DIRECT picks | B1 ego frames of the
ROLLOUT (roll_live) picks; Froude strip below) and png (Froude traces). The B1 panels show the picked
candidate clip's own recorded frames at the picked time index (kinematic replay of recorded clips, the
same assumption selection_eval grades with): nothing is simulated, so a room change between picks is a
cut between clips. E = mean L2 between the picked clip's recorded local Froude and the RECORDED goal Froude
(the grading of selection_eval, also for the vision goal). Also writes selection_summary.txt (all goals).
"""
import argparse
import os

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAB = ("forward", "lateral", "yaw")
NAME = "round1_branches_s0_rr"
MODES = (("direct", "direct", (31, 119, 180)), ("roll_live", "rollout", (255, 127, 14)))
SRC = {"recorded": "recorded goal Froude (physics / proprioceptive goal)",
       "vision": "goal read from the goal video by the model (vision goal)"}
P = 256


def key(d, mode, c, f):
    return d[f"{NAME}|w21|{mode}|{c}|{f}"]


def text(img, s, y, col=(255, 255, 255), sc=0.45, x=6):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)


def strip_plot(d, c, src, upto, width, height=300):
    g = d[f"goal|{c}"]
    t = key(d, "direct", c, "t")
    fig, axes = plt.subplots(1, 3, figsize=(width / 100, height / 100), dpi=100)
    for j, ax in enumerate(axes):
        ax.plot(t, g[t, j], color="black", lw=2, label="goal (recorded)")
        if src == "vision":
            ax.plot(t, d[f"{NAME}|{c}|vision_goal"][t, j], color="black", ls="--", lw=1.3,
                    label="goal read from video")
        for mode, lab, col in MODES:
            a = key(d, mode, c, "achieved")
            ax.plot(t, a[:, j], color=np.array(col) / 255, lw=1.6, label=lab)
        ax.axvline(t[min(upto, len(t) - 1)], color="gray", lw=1)
        ax.set_title(LAB[j], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("frame (0.05 s)", fontsize=7)
    axes[0].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return cv2.resize(img, (width, height))


def video(d, c, src, out):
    paths = [str(p) for p in d["cand_paths"]]
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    gframes = np.load(str(d[f"goal_path|{c}"]), allow_pickle=True)["frames"]
    cframes = {}
    t = key(d, "direct", c, "t")
    h = int(d["horizon"])
    g = d[f"goal|{c}"]
    W = 3 * P
    strips = [strip_plot(d, c, src, k, W) for k in range(len(t))]
    frames = []
    for k, tk in enumerate(t):
        for m in range(h):
            f = min(tk + m, len(gframes) - 1)
            row = [cv2.resize(gframes[f], (P, P))]
            heads = ["GOAL: hexapod (ego)"]
            for mode, lab, col in MODES:
                i = int(key(d, mode, c, "cand")[k])
                if i not in cframes:
                    cframes[i] = np.load(paths[i], allow_pickle=True)["frames"]
                cf = cframes[i]
                row.append(cv2.resize(cf[min(tk + m, len(cf) - 1)], (P, P)))
                heads.append(f"B1 {lab} pick: {names[i]} t={tk}")
            top = np.hstack(row).copy()
            for q, s in enumerate(heads):
                top[:22, q * P:(q + 1) * P] = (top[:22, q * P:(q + 1) * P] * 0.35).astype(np.uint8)
                text(top, s, 15, (255, 255, 255), 0.40, q * P + 4)
            foot = np.full((44, W, 3), 20, np.uint8)
            run = [key(d, mode, c, "err")[:k + 1].mean() for mode, _, _ in MODES]
            text(foot, f"goal {c} | {src} goal | frame {f}   goal Froude {g[tk, 0]:+.3f} {g[tk, 1]:+.3f} {g[tk, 2]:+.3f}",
                 17, sc=0.42)
            text(foot, f"running E: direct {run[0]:.3f}", 37, MODES[0][2], 0.45)
            text(foot, f"rollout {run[1]:.3f}", 37, MODES[1][2], 0.45, x=230)
            frames.append(np.vstack([top, foot, strips[k]]))
    title = np.full((30, W, 3), 0, np.uint8)
    text(title, "OFFLINE action selection (no physics): recorded B1 clips picked per step", 20, sc=0.5)
    frames = [np.vstack([title, fr]) for fr in frames]
    imageio.mimwrite(out, np.stack(frames), fps=10, macro_block_size=1)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PW = 256                     # panel width (3 panels = 768 px, the old slide figure's width)
CH = {"turn": (0, 2), "forward": (0, 1)}   # forward goal: yaw is flat (|goal yaw| 0.01), lateral shown instead
HEAD = ("goal: hexapod c10", "B1, direct pick", "B1, rollout pick")


def pil_text(img, s, xy, size=14, col=(255, 255, 255)):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, s, font=ImageFont.truetype(FONT, size), fill=col)
    return np.asarray(im).copy()


def bar(s, w, h=22, size=14):
    return pil_text(np.zeros((h, w, 3), np.uint8), s, (6, 2), size)


def panel_row(imgs, heads, pw):
    row = np.hstack([cv2.resize(im, (pw, pw), interpolation=cv2.INTER_AREA) for im in imgs])
    hb = np.hstack([bar(hd, pw, 20 if pw < 256 else 22, 12 if pw < 256 else 14) for hd in heads])
    return np.vstack([hb, row])


def plots(d, c, src, tag, upto=None, width=3 * PW, height=210):
    """Two small Froude plots (forward + yaw, or forward + lateral for the forward goal); data exactly as dumped."""
    g = d[f"goal|{c}"]
    t = key(d, "direct", c, "t")
    E = {lab: key(d, mode, c, "err").mean() for mode, lab, _ in MODES}
    fig, axes = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=100)
    for ax, j in zip(axes, CH[tag]):
        ax.plot(t, g[t, j], color="black", lw=1.5, label="goal (recorded)")
        if src == "vision":
            ax.plot(t, d[f"{NAME}|{c}|vision_goal"][t, j], color="0.55", ls="--", lw=1.5, label="goal read from video")
        for mode, lab, col in MODES:
            ax.plot(t, key(d, mode, c, "achieved")[:, j], color=np.array(col) / 255, lw=1.5,
                    label=f"{lab} (E {E[lab]:.3f})")
        if upto is not None:
            ax.axvline(upto, color="0.6", lw=0.8)
        ax.set_title(f"{LAB[j]} Froude", fontsize=9)
        ax.grid(True, color="0.9", lw=0.6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("frame", fontsize=7)
    axes[0].legend(fontsize=6, frameon=True, framealpha=0.85, edgecolor="none", loc="best")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return cv2.resize(img, (width, height)), E


def frame_images(d, c, k, m, cache, allo):
    """goal + 2 picks at decision step k, sub-frame m: ego frames (stored) and, if allo, third-person frames."""
    paths = [str(p) for p in d["cand_paths"]]
    t = key(d, "direct", c, "t")
    tk = int(t[k])
    src = [str(d[f"goal_path|{c}"])] + [paths[int(key(d, mode, c, "cand")[k])] for mode, _, _ in MODES]
    ego, al = [], []
    for p in src:
        if p not in cache:
            e = np.load(p, allow_pickle=True)["frames"]
            a = (np.load(os.path.join(allo, os.path.basename(p)))["frames"] if allo else None)
            cache[p] = (e, a)
        e, a = cache[p]
        f = min(tk + m, len(e) - 1)
        ego.append(e[f]); al.append(a[f] if allo else None)
    names = [os.path.splitext(os.path.basename(p))[0] for p in src]
    return ego, al, names, tk


def compose(d, c, src, tag, k, m, cache, allo, strip):
    W = 3 * PW
    ego, al, names, tk = frame_images(d, c, k, m, cache, allo)
    title = pil_text(np.full((28, W, 3), 255, np.uint8), f"Offline selection (no physics) \u2014 {tag} goal, "
                f"{'recorded Froude' if src == 'recorded' else 'goal read from video'}", (6, 5), 15, (0, 0, 0))
    parts = [title]
    if allo:
        parts.append(panel_row(al, [f"{HEAD[0]} (third person)", HEAD[1], HEAD[2]], PW))
        eh = [f"ego: {n}" for n in names]
        er = panel_row(ego, eh, 160)
        pad = np.zeros((er.shape[0], W - er.shape[1], 3), np.uint8)
        parts.append(np.hstack([er, pad]))
        parts[-1] = pil_text(parts[-1], f"ego views (model input)\nframe {tk + m}", (3 * 160 + 12, 70), 13)
    else:
        parts.append(panel_row(ego, [f"{HEAD[0]} (ego)", f"{HEAD[1]} (ego)", f"{HEAD[2]} (ego)"], PW))
    sep = np.full((6, W, 3), 255, np.uint8)
    img = np.vstack(parts + [sep, strip])
    return img


def png(d, c, src, out, tag, allo=None):
    t = key(d, "direct", c, "t")
    k = len(t) // 2
    strip, E = plots(d, c, src, tag)
    img = compose(d, c, src, tag, k, 0, {}, allo, strip)
    imageio.imwrite(out, img)
    return E


def video_allo(d, c, src, tag, out, allo):
    t = key(d, "direct", c, "t")
    h = int(d["horizon"])
    cache = {}
    frames = []
    for k, tk in enumerate(t):
        strip, _ = plots(d, c, src, tag, upto=int(tk))
        for m in range(h):
            frames.append(compose(d, c, src, tag, k, m, cache, allo, strip))
    tmp = out[:-4] + ".tmp.mp4"
    imageio.mimwrite(tmp, np.stack(frames), fps=10, macro_block_size=1)
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--vision", required=True)
    ap.add_argument("--turn", required=True)
    ap.add_argument("--forward", required=True)
    ap.add_argument("--out", default="results/deck/weekly_1008")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--allo", default="results/deck/weekly_1008/allo_cache",
                    help="third-person frames from render_allo_selection.py; '' = ego-only outputs only")
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    D = {s: dict(np.load(os.path.join(ROOT, p), allow_pickle=True)) for s, p in
         (("recorded", a.recorded), ("vision", a.vision))}
    conds = sorted(k.split("|", 1)[1] for k in D["recorded"] if k.startswith("goal|"))
    lines = ["OFFLINE action selection (no physics), B1 library rr_b1_clips_heldout (24 clips), goals "
             "rr_c10_clips_heldout (24),", f"model {NAME} + B1 projector (joint), window 21, horizon 2; "
             "rollout = roll_live. E = mean L2 picked vs recorded goal Froude.",
             "score = (random - E) / (random - oracle), averaged over goals as in selection_eval", "",
             f"{'goal':<16} {'oracle':>7} {'random':>7} | {'rec dir':>7} {'rec roll':>8} | {'vis dir':>7} {'vis roll':>8}"]
    tot = {}
    for c in conds:
        o, r = D["recorded"][f"bounds|{c}"]
        cells = []
        for s in ("recorded", "vision"):
            for mode in ("direct", "roll_live"):
                e = key(D[s], mode, c, "err").mean()
                tot.setdefault((s, mode), []).append(e)
                cells.append(e)
        tot.setdefault("o", []).append(o)
        tot.setdefault("r", []).append(r)
        lines.append(f"{c:<16} {o:7.4f} {r:7.4f} | {cells[0]:7.4f} {cells[1]:8.4f} | {cells[2]:7.4f} {cells[3]:8.4f}")
    o, r = np.mean(tot["o"]), np.mean(tot["r"])
    m = {k: np.mean(v) for k, v in tot.items() if isinstance(k, tuple)}
    sc = {k: (r - v) / (r - o) for k, v in m.items()}
    lines.append(f"{'mean':<16} {o:7.4f} {r:7.4f} | {m['recorded', 'direct']:7.4f} {m['recorded', 'roll_live']:8.4f}"
                 f" | {m['vision', 'direct']:7.4f} {m['vision', 'roll_live']:8.4f}")
    lines.append(f"{'score':<16} {'':7} {'':7} | {sc['recorded', 'direct']:+7.2f} {sc['recorded', 'roll_live']:+8.2f}"
                 f" | {sc['vision', 'direct']:+7.2f} {sc['vision', 'roll_live']:+8.2f}")
    lines.append("")
    for tag, c in (("turn", a.turn), ("forward", a.forward)):
        for s in ("recorded", "vision"):
            E = png(D[s], c, s, os.path.join(out, f"selection_{tag}_{s}.png"), tag)
            if a.allo:
                al = os.path.join(ROOT, a.allo)
                png(D[s], c, s, os.path.join(out, f"selection_{tag}_{s}_allo.png"), tag, al)
                if not a.no_video:
                    video_allo(D[s], c, s, tag, os.path.join(out, f"selection_{tag}_{s}_allo.mp4"), al)
            lines.append(f"clip selection_{tag}_{s}: goal {c}  E direct {E['direct']:.4f}  rollout {E['rollout']:.4f}")
            if not a.no_video:
                video(D[s], c, s, os.path.join(out, f"selection_{tag}_{s}.mp4"))
    open(os.path.join(out, "selection_summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
