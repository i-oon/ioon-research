"""Three-panel comparison video for one closed-loop run: goal | new body's ego | new body allocentric.

    .venv/bin/python3 scripts/render/three_panel_loop.py \\
        --run results/wm/closed_loop/direct_froude/direct-vision_hexapod_ep0_b1_ep2.npz \\
        --out results/wm/closed_loop/direct_froude/video/modeD.mp4

**What each panel is, and why the caption on the right one is not optional.**

  left    the SOURCE body's egocentric view -- the goal behaviour, the only thing the loop is
          given. For a vision goal this is literally the input the goal was read from.
  middle  the NEW body's egocentric view as the loop runs. This is the view the rollout mechanism
          consumes as `e_t`; under direct it is rendered but never read.
  right   the new body allocentric, i.e. what the behaviour looks like from outside.

**Panels are aligned by ELAPSED TIME, not frame index.** The source body records at 20 Hz and the
new body at 50 Hz, so frame `t` is 0.05t seconds for one and 0.02t for the other. Matching them by
index puts the source 2.5x further through its motion in every pair of panels, which reads as "the
goal is turning and the new body is not" when both are doing the same thing. The scene itself is
self-similar (`room_for` keeps room/body at 29.4 for both), so it is only the clock that differed.

**The right panel looks good for a reason that is not the controller.** This loop is KINEMATIC: each
step replays one frame of whichever recorded clip won, so the motion is always a real recorded gait
and cannot fall over or look unstable. The panel shows whether the right clip was SELECTED, not
whether anything was controlled. The burned-in caption says so, because the clip will outlive the
slide it was made for.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PANEL = 256
BAR = 26          # caption strip under each panel
TITLE = 22        # title strip above


def _text(img, s, y, colour=(255, 255, 255), scale=0.42, thick=1, centre=True):
    import cv2
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(2, (img.shape[1] - w) // 2) if centre else 4
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def strip(text, width, height, bg, fg=(255, 255, 255), scale=0.42):
    img = np.full((height, width, 3), bg, np.uint8)
    _text(img, text, height - 8, fg, scale)
    return img


def panel(frame, title, caption, cap_colour, froude=None, second=None):
    import cv2
    f = cv2.resize(frame, (PANEL, PANEL), interpolation=cv2.INTER_AREA).copy()
    if froude is not None:
        # burned in so the comparison is read off NUMBERS, not off how the optic flow feels
        rows = 20 if second is None else 38
        band = f[:rows].astype(np.int16) - 90
        f[:rows] = np.clip(band, 0, 255).astype(np.uint8)
        _text(f, f"fwd {froude[0]:+.3f}  lat {froude[1]:+.3f}  yaw {froude[2]:+.3f}", 14,
              (120, 255, 120), 0.38)
        if second is not None:
            _text(f, f"true {second[0]:+.3f} {second[1]:+.3f} {second[2]:+.3f}", 32,
                  (160, 160, 255), 0.36)
    return np.vstack([strip(title, PANEL, TITLE, (30, 30, 30)),
                      f,
                      strip(caption, PANEL, BAR, cap_colour)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="npz written by close_loop_direct_froude.py")
    ap.add_argument("--goal_clip", default="", help="defaults to the goal recorded in the run")
    ap.add_argument("--goal_dir", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat")
    ap.add_argument("--goal_allo_dir", default="data/allocentric/beh12_c10f10t10_flat",
                    help="allocentric recording of the goal body. **A DIFFERENT rollout of the same "
                         "condition, not the same one** -- CoppeliaSim does not repeat (F97), so "
                         "no allocentric clip is frame-synced to an egocentric one. Labelled as "
                         "such on the panel; it shows what the goal behaviour looks like from "
                         "outside, not what that exact ego clip was doing frame by frame.")
    args = ap.parse_args()

    import cv2
    d = np.load(os.path.join(ROOT, args.run), allow_pickle=True)
    allo = d["frames"]
    ego = d["ego_frames"] if "ego_frames" in d.files and len(d["ego_frames"]) else None
    mech, gsrc = str(d["mechanism"]), str(d["goal_source"])
    mode = {("direct", "physics"): "A", ("rollout", "physics"): "B",
            ("rollout", "vision"): "C", ("direct", "vision"): "D"}.get((mech, gsrc), "?")
    chosen = [str(c) for c in d["chosen"]]

    goal_path = args.goal_clip or os.path.join(ROOT, args.goal_dir, str(d["goal"]))
    with np.load(goal_path, allow_pickle=True) as g:
        goal_frames = g["frames"]

    goal_allo = None
    _ga = os.path.join(ROOT, args.goal_allo_dir, str(d["goal"]))
    if os.path.exists(_ga):
        with np.load(_ga, allow_pickle=True) as ga:
            goal_allo = ga["frames"]

    from wm.data.embodiment import REGISTRY, load
    goal_emb = str(d["goal_embodiment"])
    # **The goal the run actually consumed, not the recorded number.** Under goal_source=vision the
    # loop never sees the recorded value -- showing it here would caption the clip with a target the
    # planner was not given. Both are displayed: what the video says the goal IS, and what the
    # system READ it as.
    goal_used = (np.asarray(d["goal_used_froude"]) if "goal_used_froude" in d.files
                 else np.asarray(load(goal_path, REGISTRY[goal_emb])["body_motion"]).mean(0))
    goal_ref = (np.asarray(d["goal_reference_froude"]) if "goal_reference_froude" in d.files
                else goal_used)
    goal_fr = goal_used
    new_dt = float(d["dt"])
    goal_dt = {"hexapod": 0.05, "b1": 0.02, "gecko": 0.02}[goal_emb]

    # Froude of each candidate condition, so the picked clip's numbers can be shown as they change
    import glob as _g
    by_cond = {}
    for cp in sorted(_g.glob(os.path.join(ROOT, args.candidates_dir, "*.npz"))):
        with np.load(cp, allow_pickle=True) as cd:
            cond = str(cd["condition"])
        by_cond.setdefault(cond, []).append(
            np.asarray(load(cp, REGISTRY[str(d["embodiment"])])["body_motion"]).mean(0))
    by_cond = {k: np.mean(v, 0) for k, v in by_cond.items()}

    n = len(allo)
    if ego is None:
        print("!! this run has no ego_frames -- re-run the loop after the ego-saving change")
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    # **libx264, not OpenCV's default mpeg4.** `cv2.VideoWriter_fourcc(*"mp4v")` writes MPEG-4
    # Part 2, which browsers and most players refuse with a bare "error occurred while loading the
    # video file" -- every other renderer in this repo already uses imageio/libx264 for exactly
    # this reason, and the clips that play fine are h264.
    import imageio.v2 as imageio
    out_path = os.path.join(ROOT, args.out)
    out_frames = []
    for t in range(n):
        # elapsed-time alignment: both columns show the same number of seconds of motion
        gi = min(int(round(t * new_dt / goal_dt)), len(goal_frames) - 1)
        pick = chosen[t] if t < len(chosen) else ""
        pick_fr = by_cond.get(pick.replace("warm:", ""))
        # 2x2: rows = ego / allocentric, cols = goal body / new body
        tl = panel(goal_frames[gi], "GOAL - source body (ego)",
                   f"read {'from video' if gsrc == 'vision' else 'as a number'}",
                   (60, 60, 60), goal_fr, second=goal_ref)
        tr = (panel(ego[min(t, len(ego) - 1)], "NEW BODY (ego)", "the view the loop sees",
                    (60, 60, 60), pick_fr)
              if ego is not None else np.zeros_like(tl))
        bl = (panel(goal_allo[min(gi, len(goal_allo) - 1)], "GOAL - source body (allo)",
                    "same condition, DIFFERENT take", (90, 60, 0))
              if goal_allo is not None else np.zeros_like(tl))
        br = panel(allo[t], "NEW BODY (allocentric)",
                   "REPLAYED GROUND TRUTH - not control", (0, 0, 140), pick_fr)
        grid = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])
        W = grid.shape[1]
        # two lines: the grid is 512 wide, and one line of this overflowed the frame
        foot = np.full((44, W, 3), 20, np.uint8)
        _text(foot, f"mode {mode} ({mech}, {gsrc} goal)   t={t*new_dt:.2f}s   picked: {pick or '-'}",
              16, (255, 255, 255), 0.40, 1, centre=False)
        line2 = (f"|pick-goal| {np.abs(pick_fr - goal_fr).sum():.3f}" if pick_fr is not None else "")
        if gsrc == "vision":
            line2 += f"    |goal read err| {np.abs(goal_used - goal_ref).sum():.3f}"
        _text(foot, line2, 36, (160, 220, 160), 0.40, 1, centre=False)
        out_frames.append(np.vstack([grid, foot]))
    imageio.mimsave(out_path, out_frames, fps=args.fps, codec="libx264",
                    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"-> {args.out}  ({n} frames, mode {mode}, "
          f"2x2 grid)")


if __name__ == "__main__":
    main()
