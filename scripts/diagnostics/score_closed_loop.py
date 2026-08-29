"""Score a closed-loop run against the criteria fixed in advance, on slide 20.

    S.R. speed       |Fr_achieved - Fr_demo| / Fr_demo  < 15%
    S.R. behaviour   the dominant body channel matches the demonstration's
    S.R. survival    body height held, the robot did not fall

Reported with the **graded error beside the binary rate**, because a run that misses 15% by a
point and a run that fell over are not the same failure and a success rate cannot tell them apart.

**Survival is not optional here**, unlike in the source paper's manipulation benchmarks: a gripper
that fails a grasp is still standing, a hexapod that fails a gait is on its side. Measured as the
median body height over the second half against the first, so a slow collapse counts.

**The behaviour class is read from the body, not from what the planner selected.** A planner that
picks the right candidate and produces the wrong motion has failed, and reading its own choice back
would hide exactly that.

  .venv/bin/python3 scripts/diagnostics/score_closed_loop.py results/wm/closed_loop/*.npz
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from wm.data.embodiment import body_velocity, yaw_rate  # noqa: E402

SPEED_TOLERANCE = 0.15
# A fall is unambiguous well before the body reaches the floor; 0.75 of the starting height is
# below anything a walking clip in `data/beh12_c10f10t10_flat` shows.
HEIGHT_FLOOR = 0.75


def channels(head, quat, dt, embodiment):
    """Forward, lateral and yaw, dimensionless, over the whole clip."""
    v = body_velocity(head, quat, dt, embodiment)
    height = float(np.median(head[:, 2]))
    # `yaw_rate` returns (T, 1), `body_velocity` returns (T, 2); concatenate rather than stack, or
    # numpy broadcasts the column against the two and raises somewhere unhelpful
    w = yaw_rate(quat, dt, embodiment, height)
    return np.concatenate([v, np.asarray(w).reshape(len(v), 1)], axis=1)


def summarise(path, window=None):
    with np.load(path, allow_pickle=True) as d:
        # **The two robots store the body track under different names**: the insect writes
        # `head`/`body_quat` off the abdomen, the B1 writes `base_pos`/`base_quat` from MuJoCo,
        # and the quaternion conventions differ too. Read whichever is present rather than
        # assuming the insect's -- assuming it is why this raised on the first B1 file.
        if "head" in d.files:
            head, quat = d["head"].astype("float64"), d["body_quat"].astype("float64")
        elif "base_pos" in d.files:
            head, quat = d["base_pos"].astype("float64"), d["base_quat"].astype("float64")
        else:
            raise SystemExit(f"{os.path.basename(path)}: no body track "
                             "(`head`/`body_quat` or `base_pos`/`base_quat`)")
        dt = float(d["dt"]) if "dt" in d.files else 0.05
        embodiment = str(d["embodiment"]) if "embodiment" in d.files else "hexapod"
        demo = str(d["demo"]) if "demo" in d.files else ""
        chosen = d["chosen"] if "chosen" in d.files else None
        want = str(d["condition"]) if "condition" in d.files else ""
        goal = str(d["goal"]) if "goal" in d.files else ""
        kinematic = bool(d["kinematic"]) if "kinematic" in d.files else False
        fell_at = int(d["fell_at"]) if "fell_at" in d.files else -1
    c = channels(head, quat, dt, embodiment)
    # **Steps driven by the warm start are the demonstration replayed, not the planner's work.**
    # Scoring them was worth 21 points of apparent speed accuracy on a run whose behaviour picks
    # had not improved at all: `--warm_start 10` of 20 steps put half the demonstration inside the
    # number. The channels are still computed over the whole clip, so `np.gradient` and the
    # smoothing window do not see an artificial edge at the handover; only the summary is masked.
    warm = 0
    if chosen is not None:
        labels = np.asarray(chosen, dtype=str)
        warm = int(np.sum(np.char.startswith(labels, "warm:")))
    lo, hi = (warm, len(c)) if window is None else window
    hi = min(hi, len(c))
    c, head_p = c[lo:hi], head[lo:hi]
    if len(c) < 3:
        raise SystemExit(f"{os.path.basename(path)}: only {len(c)} steps to score")
    half = len(head_p) // 2
    return {"path": path, "demo": demo, "goal": goal, "condition": want, "dt": dt,
            "embodiment": embodiment,
            "forward": float(np.median(c[:, 0])), "lateral": float(np.median(c[:, 1])),
            "yaw": float(np.median(c[:, 2])),
            "fell_at": fell_at, "planned_steps": len(c),
            "height0": float(np.median(head_p[:half, 2])),
            "height1": float(np.median(head_p[half:, 2])),
            "chosen": chosen, "steps": len(head_p), "warm": warm, "window": (lo, hi),
            "kinematic": kinematic}


CHANNEL = {"forward": "forward", "sideways": "lateral", "turn": "yaw"}


def dominant(row):
    """Which body channel carries this motion, on the dimensionless scale the two robots share."""
    order = [("forward", abs(row["forward"])), ("sideways", abs(row["lateral"])),
             ("turn", abs(row["yaw"]))]
    return max(order, key=lambda kv: kv[1])[0]


def channel_for(condition, ref):
    """The channel a condition is *named* for, falling back to the largest when the name says
    nothing.

    **Choosing by magnitude graded every turn as a walk.** Forward speed exceeds yaw in all four
    turn conditions on both bodies -- 0.136 against 0.088 even at `turn_s0.56` -- so `S.R. speed`
    measured forward travel on turning runs and **turn rate was never scored in any closed loop**
    (F108). Shared with the renderer so a video's header cannot disagree with the table.
    """
    named = {"turn": "yaw", "side": "lateral", "speed": "forward"}
    for prefix, ch in named.items():
        if str(condition).startswith(prefix):
            return ch
    return CHANNEL[dominant(ref)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="closed-loop npz files")
    ap.add_argument("--demo_dir", default="data/beh12_c10f10t10_flat",
                    help="where the demonstration clips live, for the reference Froude")
    ap.add_argument("--goal_dir", default="",
                    help="where the **goal** clips live, when the goal is a different robot from "
                         "the one driven. **The goal is what a run has to be scored against, not "
                         "the demonstration**: in a cross-embodiment run `--demo` supplies only "
                         "the start state and the warm-start commands and is held fixed and "
                         "neutral while the goal varies (F109, F110), so grading against it reads "
                         "a forward B1 walk for a run whose goal was an insect turning. Defaults "
                         "to `--demo_dir`, which is right when goal and demonstration coincide.")
    args = ap.parse_args()

    print(f"{'run':<30}{'steps':>11}{'channel':>9}{'demo':>9}{'got':>9}{'err':>8}{'class':>11}"
          f"{'height':>9}{'  verdict'}")
    passes = {"speed": 0, "behaviour": 0, "survival": 0}
    scored = {"survival": 0}
    graded = []
    for path in args.runs:
        row = summarise(path)
        # The run records both, and they are the same file unless `--goal` was given.
        crosses = bool(row["goal"]) and row["goal"] != row["demo"]
        ref_name = row["goal"] if crosses else row["demo"]
        ref_dir = (args.goal_dir or args.demo_dir) if crosses else args.demo_dir
        demo_path = os.path.join(ROOT, ref_dir, ref_name)
        if not os.path.exists(demo_path):
            raise SystemExit(f"reference {ref_name} not found under {ref_dir}; pass --demo_dir "
                             "(or --goal_dir when the goal is another robot)")
        # **The reference has to cover the same steps.** Both clips start from a standstill and
        # spend their first second accelerating; scoring our 10 planned steps against the
        # demonstration's whole 66 compares a start-up transient against a settled walk, and made
        # the cold start look like a planner failure when part of it is the robot getting going.
        ref = summarise(demo_path, window=row["window"])

        # **Score the channel the demonstration is actually about.** The criterion on slide 20 was
        # written when every clip walked forwards. On a sideways clip the *forward* Froude is 0.015
        # -- near zero by construction -- and a relative error against it is a ratio of two noise
        # figures: the first run of `side_R_lvl0` read 34.2% while tracking its lateral speed to
        # within a fifth of that. The channel is chosen from the demonstration, never from the run,
        # or a run that drifted into a different behaviour would be graded on the one it drifted to.
        # **The condition of the *reference*, not of the run.** They differ only in a
        # cross-embodiment run, where the run inherits its demonstration's condition -- and taking
        # that one grades an insect's turn on the B1 demonstration's forward channel.
        key = channel_for(ref["condition"] or row["condition"], ref)
        err = abs(row[key] - ref[key]) / max(abs(ref[key]), 1e-6)
        ok_speed = err < SPEED_TOLERANCE
        ok_class = dominant(row) == dominant(ref)
        held = row["height1"] / max(row["height0"], 1e-6)
        ok_alive = held > HEIGHT_FLOOR
        # **A run that stopped because it fell hides the fall from this test.** The loop truncates
        # the recording at the fall, so the second half never contains the collapse and the height
        # ratio reads as healthy. Two B1 physics runs scored `A` while having fallen at steps 29
        # and 37. Where the loop recorded the step it fell on, that is the answer.
        if row.get("fell_at", -1) >= 0:
            ok_alive = False
            held = float(row.get("fell_at")) / max(row.get("planned_steps", 1), 1)
        # **Survival is not a result in a kinematic run.** The body is posed frame by frame, so it
        # passes whatever the planner chose. Reported as `.` rather than `A` so a table cannot be
        # read as if the two kinds of run had earned the same column.
        if row["kinematic"]:
            ok_alive = None
        passes["speed"] += ok_speed
        passes["behaviour"] += ok_class
        # **A kinematic run is not scored here, it is excluded.** `ok_alive` is None when the body
        # was posed rather than simulated, and `bool(None)` counted it as a fall -- so a loop that
        # cannot fall reported "S.R. survival 0%", the most alarming number on the page, as an
        # artefact of the tally. Counted separately so the rate divides by what was graded.
        if ok_alive is not None:
            passes["survival"] += ok_alive
            scored["survival"] += 1
        graded.append(err)
        verdict = "".join(("S" if ok_speed else "-", "B" if ok_class else "-",
                           "." if ok_alive is None else ("A" if ok_alive else "-")))
        span = f"{row['window'][0]}-{row['window'][1]}"
        print(f"{os.path.basename(path):<30}{span:>11}{key:>9}{ref[key]:>9.3f}{row[key]:>9.3f}"
              f"{err:>8.1%}{dominant(row):>11}{held:>9.2f}   {verdict}")

    n = len(args.runs)
    print(f"\n{'':<30}{'rate':>8}{'  of':>5}")
    for key in ("speed", "behaviour", "survival"):
        of = scored.get(key, n)
        if not of:
            print(f"S.R. {key:<25}{'n/a':>8}{0:>5}")
        else:
            print(f"S.R. {key:<25}{passes[key] / of:>8.0%}{of:>5}")
    print(f"\nmedian speed error {np.median(graded):.1%}, worst {np.max(graded):.1%}")
    print("verdict letters: S speed within 15%, B right behaviour class, A still standing")
    if any(summarise(p)["kinematic"] for p in args.runs):
        print("`.` in the third column: a kinematic run, where the body is posed rather than")
        print("simulated and cannot fall. Its survival column is not evidence of anything.")


if __name__ == "__main__":
    main()
