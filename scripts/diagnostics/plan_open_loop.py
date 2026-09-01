"""Run the planner over recorded frames, before any simulator is involved.

The closed loop has two independent things that can be wrong -- the planner's choice and the
execution -- and this exercises the first alone. Frames come from a held-out clip instead of from
CoppeliaSim; everything else is the control-time path, projector and forward model only.

    at each step t of a demonstration clip:
        e_t     the demonstration's own frame          (in the loop this comes from the camera)
        e_goal  the demonstration h steps ahead        (what we are trying to reach)
        the planner scores every behaviour and picks one

**Success is picking the demonstration's own condition.** A planner that cannot do that on the
demonstration's own frames cannot do it on frames its own actions produced, so this is a necessary
condition and a cheap one.

Two numbers, and the second is the one to trust:

  exact      the chosen candidate is the demonstration's condition
  family     the chosen candidate is the right *behaviour* -- speed, turn or sideways -- whatever
             the level. F90 measured level discrimination at 57.8% against a 25% chance, so a
             planner that gets the family right and the level wrong is behaving as measured rather
             than failing.

  .venv/bin/python3 scripts/diagnostics/plan_open_loop.py --ckpt wm/runs/beh12_hexonly/best.pt
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import LatentPlanner, condition_of  # noqa: E402


def family(condition):
    return condition.rsplit("_", 1)[0] if "_" in condition else condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="")
    ap.add_argument("--candidates_dir", default="data/allocentric/beh12_c10f10t10_flat")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--demo_dir", default="",
                    help="clips supplying `e_t`, the frames the planner is standing in. Defaults "
                         "to --candidates_dir, i.e. the same robot")
    ap.add_argument("--goal_dir", default="",
                    help="clips supplying `e_goal`, which may be **another robot**. This is the "
                         "cross-embodiment question asked without a simulator: the planner scores "
                         "a raw MSE between a predicted B1 embedding and a goal embedding, and if "
                         "that goal is an insect frame the distance is dominated by which robot is "
                         "in the picture rather than by what it is doing. Selection at chance here "
                         "would mean the loop's dynamics are innocent and the metric is the fault")
    ap.add_argument("--goal_embodiment", default="")
    ap.add_argument("--mismatch", action="store_true",
                    help="pair each demonstration with a goal from a **different behaviour "
                         "family**. The control for `--center`: the shift moves the goal onto the "
                         "driven robot's manifold, so a gain could be the goal becoming reachable "
                         "rather than its content becoming readable. Under a mismatched goal, a "
                         "planner reading the content should follow the *goal* and score badly "
                         "against the demonstration; one ignoring it should be unmoved. Both "
                         "numbers are printed for exactly that reason.")
    ap.add_argument("--center_mode", choices=("clip", "dataset"), default="clip",
                    help="which mean `--center` subtracts. **`clip` removes the behaviour along "
                         "with the robot** and was measured doing exactly that: a clip's mean is "
                         "66 frames of one behaviour, and on these robots the behaviour lives "
                         "largely in that mean -- a turning insect's average posture differs from "
                         "a walking one's -- so subtracting it took goal-following from 38.9% to "
                         "22.1% (F125). `dataset` averages over `--center_clips` clips of each "
                         "robot instead, which removes 'this is an insect' and keeps 'this clip "
                         "is a turn'.")
    ap.add_argument("--center_clips", type=int, default=12,
                    help="clips per robot used to estimate a dataset mean")
    ap.add_argument("--center", action="store_true",
                    help="translate the goal clip into the driven robot's mean appearance before "
                         "scoring. **The first-order correction and nothing more**: it removes a "
                         "constant offset between the two robots and leaves any difference in "
                         "variance structure untouched, so partial recovery is the expected "
                         "outcome if appearance is the problem. Only the goal is moved -- `e_t` is "
                         "the forward model's input and shifting it would trade one confound for "
                         "another")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--demos", type=int, default=12, help="demonstration clips to run")
    ap.add_argument("--stride", type=int, default=5, help="replan every N frames of the demo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    planner = LatentPlanner.from_checkpoint(
        ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment,
        os.path.join(ROOT, args.projector) if args.projector else "",
        horizon=args.horizon, device=str(device))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # **Demonstrations must not be the candidates themselves.** `load_candidates` takes the first
    # clip of each condition; taking the demo from the same file would ask the planner to match a
    # sequence against its own frames, which it would do perfectly and mean nothing.
    demo_dir = args.demo_dir or args.candidates_dir
    used = {c["path"] for c in planner.candidates} if demo_dir == args.candidates_dir else set()
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, demo_dir, "*.npz")))
             if os.path.basename(p) not in {os.path.basename(u) for u in used}]
    if not paths:
        raise SystemExit("every clip is in the candidate set; nothing left to demonstrate with")
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:args.demos]]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    offset = offset_for(checkpoint, args.embodiment)
    spec = REGISTRY[args.embodiment]
    goal_dir = args.goal_dir or demo_dir
    goal_emb = args.goal_embodiment or args.embodiment
    goal_spec = REGISTRY[goal_emb]
    # **The goal clip is matched to the demonstration by condition, not by index.** Asking the
    # planner to reach an insect's turn from a B1 clip is the question; pairing them arbitrarily
    # would ask a different one each time.
    # **Keyed by (behaviour, level), not by the condition string.** The two robots name their
    # conditions after their own controls -- `speed_vx0.30` against `speed_c5.8`, `turn_w0.008`
    # against `turn_s0.05` -- so only the sideways names coincide. Matching on the string silently
    # kept the sideways demonstrations and dropped everything else, which is the one family that
    # fails on every measurement this project has made.
    def key_of(path):
        with np.load(path, allow_pickle=True) as z:
            return (str(z["behaviour"]), int(z["level"])) if "behaviour" in z.files else (None, None)
    goals = {}
    for gp in sorted(glob.glob(os.path.join(ROOT, goal_dir, "*.npz"))):
        goals.setdefault(key_of(gp), gp)
    if args.mismatch:
        # deterministic: each family's demonstrations get the next family's goal, same level
        # **The `behaviour` field holds three values -- `speed`, `turn`, `side` -- not the four
        # condition families.** A rotation written over `side_L`/`side_R` matches almost nothing and
        # drops the demonstrations silently: the first version of this ran 2 demonstrations out of 8
        # and reported the result as though it had run all of them.
        order = ["speed", "turn", "side"]
        rotated = {}
        for (beh, lvl), gp in goals.items():
            if beh in order:
                rotated[(order[(order.index(beh) - 1) % len(order)], lvl)] = gp
        missing = {k for k in goals if k not in rotated}
        if missing:
            raise SystemExit(f"--mismatch left {len(missing)} demonstration keys unpaired: {missing}")
        goals = rotated

    dataset_mean = {}
    if args.center and args.center_mode == "dataset":
        for tag, d in (("driven", demo_dir), ("goal", goal_dir)):
            files = sorted(glob.glob(os.path.join(ROOT, d, "*.npz")))[:args.center_clips]
            spec_here = spec if tag == "driven" else goal_spec
            acc = None
            for fp in files:
                ee = encode_clip(encoder, load(fp, spec_here)["frames"], args.chunk).float()
                acc = ee.mean(0, keepdim=True) if acc is None else acc + ee.mean(0, keepdim=True)
            dataset_mean[tag] = acc / len(files)
            print(f"{tag} mean over {len(files)} clips of {d}")

    print(f"{len(planner.candidates)} candidates, {len(paths)} demonstrations, "
          f"horizon {args.horizon}, replan every {args.stride} frames\n")
    exact, fam, total = 0, 0, 0
    goal_fam = 0
    confusion = collections.Counter()
    for path in paths:
        clip = load(path, spec)
        e = encode_clip(encoder, clip["frames"], args.chunk).float()
        if offset is not None:
            e = e - offset.to(e.device)
        want = condition_of(path)
        if goal_dir == demo_dir:
            e_goal_src = e
        else:
            k = key_of(path)
            if k not in goals:
                continue
            g = load(goals[k], goal_spec)
            goal_condition = condition_of(goals[k])
            e_goal_src = encode_clip(encoder, g["frames"], args.chunk).float()
            if offset is not None:
                e_goal_src = e_goal_src - offset.to(e.device)
        if args.center and args.center_mode == "dataset":
            # remove each robot's own appearance, estimated across its clips, and put the goal in
            # the driven robot's frame -- the between-clip differences the behaviour lives in survive
            e_goal_src = (e_goal_src - dataset_mean["goal"].to(e_goal_src.device)
                          + dataset_mean["driven"].to(e_goal_src.device))
        elif args.center:
            # **Only the goal moves.** `e` is also the forward model's input, and shifting that
            # puts the model off the distribution it was fitted on -- which would confound the
            # correction with an out-of-distribution error. So the goal is translated into the
            # driven robot's appearance frame instead: same behaviour content, B1 mean.
            e_goal_src = e_goal_src - e_goal_src.mean(0, keepdim=True) + e.mean(0, keepdim=True)
        picks = []
        for t in range(1, len(e) - args.horizon - planner.action_lag - 1, args.stride):
            h = planner.horizon_at(t)
            if t + h >= len(e_goal_src):
                break
            _, i, _ = planner.act(e[t:t + 1], e_goal_src[t + h], t)
            got = planner.candidates[i]["condition"]
            picks.append(got)
            exact += got == want
            fam += family(got) == family(want)
            if goal_dir != demo_dir:
                goal_fam += family(got) == family(goal_condition)
            total += 1
            confusion[(want, got)] += 1
        top = collections.Counter(picks).most_common(1)[0]
        print(f"{want:<16} -> {top[0]:<16} {top[1]}/{len(picks)} steps"
              f"{'' if family(top[0]) == family(want) else '   WRONG FAMILY'}")

    print(f"\nexact condition  {exact / max(total, 1):.1%}   "
          f"chance {1 / len(planner.candidates):.1%}")
    fams = len({family(c['condition']) for c in planner.candidates})
    print(f"right behaviour  {fam / max(total, 1):.1%}   chance {1 / fams:.1%}"
          f"   (against the demonstration)")
    if goal_dir != demo_dir:
        print(f"goal's behaviour {goal_fam / max(total, 1):.1%}   chance {1 / fams:.1%}"
              f"   (against the clip the planner was shown)")
    print(f"{total} decisions over {len(paths)} demonstrations")

    worst = [(f"{w} -> {g}", n) for (w, g), n in confusion.most_common()
             if w != g and family(w) != family(g)][:5]
    if worst:
        print("\nmost common cross-behaviour confusions:")
        for label, n in worst:
            print(f"  {label:<40}{n}")


if __name__ == "__main__":
    main()
