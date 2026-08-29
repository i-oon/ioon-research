"""Watch a closed-loop run beside the demonstration it was asked to reproduce.

The scoring script says upright and on-speed. It cannot say whether the gait looks like walking,
and this project's rule is to render before believing a body -- the two clips in
`data/fwd_hex8body` that do not walk passed every summary statistic they were checked against.

**The frame carries its own numbers.** Every figure in this project does, because a picture of two
insects is equally consistent with a working controller and a broken one. The header states which
run this is, which channel it was scored on, what the demonstration achieved, what the run achieved,
and the verdict; the footer states the behaviour chosen for the step on screen.

**Warm-start steps are marked.** Until the handover the right pane is the demonstration's own
commands replayed, so nothing before it is evidence about the controller.

  .venv/bin/python3 sim/render/render_closed_loop.py results/wm/closed_loop/full/*.npz
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
from npz_to_video import write_mp4  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from diagnostics.score_closed_loop import channel_for, dominant, summarise  # noqa: E402

INK = (16, 17, 20)
WHITE, DIM = (245, 245, 245), (150, 152, 158)
GREEN, RED, AMBER = (110, 210, 140), (235, 115, 95), (235, 190, 95)
HEADER, TITLES, FOOTER, LEGEND = 58, 20, 24, 22


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = f"/usr/share/fonts/truetype/dejavu/{name}"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


F_TITLE, F_BODY, F_SMALL = _font(14, bold=True), _font(12), _font(10)


def compose(ref, got, header, sub, pick, want, t, n, warm):
    """One video frame: header, two labelled panes, per-step footer, colour legend."""
    h, w = ref.shape[:2]
    total_h = HEADER + TITLES + h + FOOTER + LEGEND
    im = Image.new("RGB", (w * 2, total_h), INK)
    d = ImageDraw.Draw(im)

    d.text((8, 5), header, font=F_TITLE, fill=WHITE)
    # Two short lines rather than one long one: at 512 px the single line was cut off mid-word,
    # which is worse than no caption because it looks like the number ends where the frame does.
    for i, line in enumerate(sub):
        d.text((8, 23 + i * 14), line, font=F_SMALL, fill=DIM)

    y = HEADER
    d.text((8, y + 4), "DEMONSTRATION", font=F_SMALL, fill=DIM)
    d.text((w + 8, y + 4), "CLOSED LOOP", font=F_SMALL, fill=DIM)
    im.paste(Image.fromarray(ref), (0, y + TITLES))
    im.paste(Image.fromarray(got), (w, y + TITLES))
    d.line([(w, y), (w, y + TITLES + h)], fill=(60, 62, 68))

    y += TITLES + h
    d.text((8, y + 6), f"step {t + 1} / {n}", font=F_BODY, fill=DIM)
    if t < warm:
        d.text((w + 8, y + 6), f"warm start: replaying {want}", font=F_BODY, fill=AMBER)
    else:
        d.text((w + 8, y + 6), f"chose  {pick}", font=F_BODY,
               fill=GREEN if pick == want else RED)

    y += FOOTER
    # Laid out from measured widths. Fixed offsets overlapped: "matches the demonstration" ran
    # straight into "a different behaviour" at this frame size.
    dx = 8
    for colour, text in ((GREEN, "matches"), (RED, "different"), (AMBER, "warm start, not planned")):
        d.rectangle([dx, y + 7, dx + 7, y + 14], fill=colour)
        d.text((dx + 11, y + 5), text, font=F_SMALL, fill=DIM)
        dx += 11 + int(d.textlength(text, font=F_SMALL)) + 18
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--demo_dir", default="data/beh12_c10f10t10_flat")
    ap.add_argument("--goal_dir", default="data/beh12_c08f09t09_flat",
                    help="where to find the goal clip when it came from another robot")
    ap.add_argument("--out", default="results/wm/closed_loop/video")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for pattern in args.runs for p in sorted(glob.glob(pattern))] or args.runs
    for path in paths:
        with np.load(path, allow_pickle=True) as d:
            got = d["frames"]
            chosen = np.asarray(d["chosen"], dtype=str)
            want = str(d["condition"])
            demo_name = str(d["demo"])
            # **The right pane is what the planner was asked to reach**, which is not always the
            # demonstration: a cross-embodiment run is driven by another robot's clip and showing
            # the driven robot's own demonstration instead hides what the run was.
            goal_name = str(d["goal"]) if "goal" in d.files else ""
        demo_path = os.path.join(ROOT, args.demo_dir, demo_name)
        cross_goal = bool(goal_name) and goal_name != demo_name
        if cross_goal:
            found = [p for p in (os.path.join(ROOT, args.goal_dir, goal_name),
                                 os.path.join(ROOT, args.demo_dir, goal_name)) if os.path.exists(p)]
            if not found:
                raise SystemExit(f"goal {goal_name} not found; pass --goal_dir")
            demo_path, demo_name = found[0], goal_name
        if not os.path.exists(demo_path):
            raise SystemExit(f"demonstration {demo_name} not found under {args.demo_dir}")
        ref = np.load(demo_path, allow_pickle=True)["frames"]

        # the same numbers the scorer reports, so the video cannot drift from the table
        row = summarise(path)
        gold = summarise(demo_path, window=row["window"])
        # **Shared with the scorer, so a frame's header cannot contradict the table.** And when the
        # goal came from another robot there is no rate to report: the two bodies walk at different
        # scales, so comparing this one's speed against the other's is a number about nothing.
        key = channel_for(row["condition"], gold)
        err = (float("nan") if cross_goal
               else abs(row[key] - gold[key]) / max(abs(gold[key]), 1e-6))
        held = row["height1"] / max(row["height0"], 1e-6)
        verdict = "".join(("S" if (err == err and err < 0.15) else "-",
                           "B" if dominant(row) == dominant(gold) else "-",
                           "A" if held > 0.75 else "-"))
        warm = row["warm"]
        planned = [c for c in chosen if not c.startswith("warm:")]
        hit = sum(c == want for c in planned) / max(len(planned), 1)

        header = f"{os.path.splitext(os.path.basename(path))[0]}   demonstration: {want}"
        tail = ("cross-embodiment goal, rate not comparable" if err != err
                else f"error {err:.1%}")
        sub = [f"{key} Froude   demo {gold[key]:+.3f}  ->  ran {row[key]:+.3f}   "
               f"{tail}   verdict {verdict}",
               f"{hit:.0%} of planned steps chose {want}   |   warm start: {warm} steps"]

        # the travel gate can stop a run early; the demonstration is a fixed 66 frames
        n = min(len(got), len(ref), len(chosen))
        out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".mp4")
        write_mp4(out, (compose(ref[t], got[t], header, sub, chosen[t], want, t, n, warm)
                        for t in range(n)), args.fps)
        shown = "  n/a " if err != err else f"{err:>6.1%}"
        print(f"{os.path.relpath(out, ROOT)}   {n} frames   {shown}  {verdict}  {hit:.0%} on target")


if __name__ == "__main__":
    main()
