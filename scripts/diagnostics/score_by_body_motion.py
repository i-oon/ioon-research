"""Score candidates by the body motion they produce, not by embedding distance.

**Why a different coordinate at all.** The planner scores `||FDM(e_t, proj(a)) - e_goal||`, a raw
distance in V-JEPA2 space. That was never a shared coordinate between two robots -- F125 measured
the correction that would make it one failing under every estimator tried -- and F127 measured
something worse: the argmin does not track the goal even *within* one robot, scoring 18-23% against
the behaviour it was shown against a 28% chance rate.

This asks the same question in the one coordinate the two robots genuinely share. `lambda_body`
trains a head that reads **body motion** off the latent, in dimensionless units both bodies are
measured in (F58, F65, F66), so

    score(a) = | body_head(proj(a)) - body_motion(goal clip) |

is goal-conditioned by construction: the goal enters as a physical quantity rather than as a
picture that happens to contain the wrong robot.

**The acceptance test is the mismatched column and nothing else** (F127). Scored against the
demonstration, a rule can pass without reading the goal at all -- that is how F123's 55.8% and
F126's 73% both survived until their controls were run. So this script computes matched *and*
mismatched goals in the same pass and reports both, and the number that decides is the mismatched
one: **above 28% or this is not planning either.**

**What it cannot do, stated in advance.** The checkpoint's body head is `body_dim 1`,
`body_channels ['0']` -- forward speed alone. Turning and sideways are not represented in the
shared coordinate at all, so this rule can only separate behaviours that differ in forward speed.
A negative result is therefore evidence about *this* head, not about body-motion scoring in
general; a positive one would be worth widening the head for.

    .venv/bin/python3 scripts/diagnostics/score_by_body_motion.py \\
        --ckpt wm/runs/beh12_hexonly/stage3_b1_nce_s0.pt --data data/beh12_b1_flat
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import FAMILY, gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import body_velocity  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def forward_speed(path, embodiment):
    """The clip's dimensionless forward speed per frame -- channel 0 of the shared target."""
    with np.load(os.path.join(ROOT, path), allow_pickle=True) as z:
        if "head" in z.files:
            pos, quat = z["head"].astype("float64"), z["body_quat"].astype("float64")
        else:
            pos, quat = z["base_pos"].astype("float64"), z["base_quat"].astype("float64")
        dt = float(z["dt"]) if "dt" in z.files else 0.05
    return body_velocity(pos, quat, dt, embodiment)[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="", help="defaults to --ckpt, which carries one")
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 3, 5, 10])
    ap.add_argument("--cache", default="results/wm/cache/b1.pt")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--goal_dir", default="",
                    help="take the target body motion from **another robot's** clips, matched by "
                         "(behaviour, level). This is the question the project exists to answer, "
                         "asked in the one coordinate the two robots share: the target is a "
                         "dimensionless forward speed, so no appearance has to cancel and no "
                         "embedding distance is involved.")
    ap.add_argument("--goal_embodiment", default="hexapod")
    ap.add_argument("--limit", type=int, default=240)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    name = args.embodiment

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), name, encoder, ck, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    md = MotionDecoder(cfg, {name: clips[0]["a"].shape[1]}).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    if md.body_head is None:
        raise SystemExit("this checkpoint has no body head (lambda_body 0)")
    saved = torch.load(os.path.join(ROOT, args.projector or args.ckpt),
                       map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
    proj.load_state_dict(saved["projector"])

    # the head predicts a standardised target; put the measured speed on the same scale
    mean, std = ck["body_stats"]
    mean, std = float(np.asarray(mean).ravel()[0]), float(np.asarray(std).ravel()[0])
    speed = {c["path"]: forward_speed(os.path.join(args.data, c["path"]), name) for c in clips}

    # cross-embodiment goals: one clip per (behaviour, level) of the other robot, and its measured
    # forward speed. Keyed on the recorded fields, never on the condition string -- the two robots
    # name their conditions after their own controls and only the sideways names coincide (F125).
    cross = {}
    if args.goal_dir:
        for gp in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
            with np.load(gp, allow_pickle=True) as z:
                key = (str(z["behaviour"]), int(z["level"]))
                cond = str(z["condition"])
            if key not in cross:
                cross[key] = (cond, forward_speed(os.path.join(args.goal_dir,
                                                               os.path.basename(gp)),
                                                  args.goal_embodiment))
        print(f"cross-embodiment goals: {len(cross)} conditions from {args.goal_dir}")

    clip_key = {}
    for c in clips:
        with np.load(os.path.join(ROOT, args.data, c["path"]), allow_pickle=True) as z:
            clip_key[c["path"]] = (str(z["behaviour"]), int(z["level"]))

    cand, seen = {}, set()
    for i, c in enumerate(clips):
        if c["cond"] not in seen:
            cand[c["cond"]] = i; seen.add(c["cond"])
    conds = sorted(cand)
    val = [(i, t) for i, c in enumerate(clips) if i not in cand.values() for t in range(c["n"])]

    # deterministic family rotation, one non-candidate clip per family
    by_family = {}
    for i, c in enumerate(clips):
        if i not in cand.values():
            by_family.setdefault(FAMILY(c["cond"]), i)
    order = sorted(by_family)
    mismatch = {f: by_family[order[(order.index(f) + 1) % len(order)]] for f in order}

    g = torch.Generator().manual_seed(0)
    picks = val if len(val) <= args.limit else [val[i] for i in
                                               torch.randperm(len(val), generator=g)[:args.limit]]
    print(f"{len(clips)} clips | {len(cand)} candidates | body head {cfg.body_dim}-D, "
          f"channels {cfg.body_channels} | rotation "
          + ", ".join(f"{f}->{FAMILY(clips[i]['cond'])}" for f, i in sorted(mismatch.items())))
    print(f"\n  {'horizon':>8}{'matched':>10}{'mismatched vs demo':>21}{'MISMATCHED VS GOAL':>21}"
          f"{'n':>7}")

    with torch.no_grad():
        for h in args.horizons:
            hit = {"matched": 0, "mm_demo": 0, "mm_goal": 0}
            n = 0
            for c, t in picks:
                if t + h >= clips[c]["n"]:
                    continue
                gc = mismatch.get(FAMILY(clips[c]["cond"]), c)
                if t + h >= clips[gc]["n"]:
                    continue
                acts, keep = [], []
                for k in conds:
                    src = cand[k]
                    if t + h < clips[src]["n"]:
                        acts.append(torch.stack([clips[src]["a"][t + i] for i in range(h)]))
                        keep.append(k)
                if len(keep) < 2:
                    continue
                a = torch.stack(acts).to(device)
                C = len(keep)
                z = proj(a.reshape(C * h, -1), name).reshape(C, h, -1)
                pred = md.body(None, z.reshape(C * h, -1)).reshape(C, h).mean(1)
                if args.goal_dir:
                    # the demonstration's own (behaviour, level) supplies the matched cross goal;
                    # the rotation supplies the mismatched one
                    dk = clip_key[clips[c]["path"]]
                    gk = clip_key[clips[gc]["path"]]
                    if dk not in cross or gk not in cross:
                        continue
                for tag, source in (("matched", c), ("mm", gc)):
                    if args.goal_dir:
                        key = clip_key[clips[source]["path"]]
                        target = float(np.mean(cross[key][1][t:t + h]))
                    else:
                        target = float(np.mean(speed[clips[source]["path"]][t:t + h]))
                    err = (pred - (target - mean) / std).abs()
                    got = FAMILY(keep[int(err.argmin())])
                    if tag == "matched":
                        hit["matched"] += got == FAMILY(clips[c]["cond"])
                    else:
                        hit["mm_demo"] += got == FAMILY(clips[c]["cond"])
                        hit["mm_goal"] += got == FAMILY(clips[gc]["cond"])
                n += 1
            d = max(n, 1)
            print(f"  {h:>8}{hit['matched']/d:>10.0%}{hit['mm_demo']/d:>21.0%}"
                  f"{hit['mm_goal']/d:>21.0%}{n:>7}")

    print("\n  chance is 28%. **Only the last column decides** -- the first two are passable by a")
    print("  rule that names the behaviour already visible and never reads the goal (F123, F127).")


if __name__ == "__main__":
    main()
