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
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
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
    ap.add_argument("--mode", choices=("A", "B", "C", "D"), default="A",
                    help="**what the score is allowed to use.** "
                         "`A` scores `body_head(proj(a))` against the goal clip's *measured* speed "
                         "-- no frames, no forward model, which makes it action-to-speed regression "
                         "rather than planning. "
                         "`B` puts the world model back: roll the FDM h steps from `e_t` on the "
                         "candidate's latent, read the resulting transition with the ITM, and take "
                         "the body motion of that. "
                         "`C` also takes the target from the goal robot's **frames** -- "
                         "`body_head(ITM(g_t, g_t+h))` -- so no recorded trajectory value enters "
                         "either side and the comparison is frames and rollout only. **C is the "
                         "condition the project's claim actually needs.** "
                         "`D` is C with the rollout deleted -- the target still comes from the "
                         "goal robot's frames, the candidate side is `body_head(proj(a))` again. "
                         "**D is the control that prices the world model**: A and C differ in two "
                         "things at once, since A is also handed a measured trajectory value, so "
                         "only D against C isolates what rolling the forward model buys.")
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

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
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
    cross, cross_e = {}, {}
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
        if args.mode in ("C", "D"):
            enc2 = VJEPA2FrameEncoder(dtype=torch.float32)
            gcache_path = os.path.join(ROOT, "results/wm/cache/bodycal_hexapod.pt")
            gcache = (torch.load(gcache_path, map_location="cpu")
                      if os.path.exists(gcache_path) else {})
            n0 = len(gcache)
            gclips = gather(os.path.join(ROOT, args.goal_dir), args.goal_embodiment, enc2, ck,
                            gcache, args.chunk, max(1, cfg.action_lag), device)
            if len(gcache) > n0:
                torch.save(gcache, gcache_path)
            del enc2
            torch.cuda.empty_cache()
            for gc_ in gclips:
                with np.load(os.path.join(ROOT, args.goal_dir, gc_["path"]),
                             allow_pickle=True) as z_:
                    k_ = (str(z_["behaviour"]), int(z_["level"]))
                if k_ in cross and k_ not in cross_e:
                    cross_e[k_] = gc_["e"]
            print(f"  mode C: {len(cross_e)} goal clips encoded, targets read from their frames")

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
            # **Per family as well as pooled.** A pooled number hides a reshuffle: F124 measured the
            # aggregate holding at 36% while turning rose 18 points and forward fell 20. The family
            # here is the *goal's*, since that is what the rule was asked for.
            per = {}
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
                if args.mode in ("A", "D"):
                    # (C*h, body_dim) -> (C, h, body_dim) -> mean over the horizon
                    pred = md.body(None, z.reshape(C * h, -1)).reshape(C, h, -1).mean(1)
                else:
                    # roll the forward model on each candidate, then read the transition it
                    # implies. **`ITM` on a pair h apart is wider than it was trained on**, which
                    # is part of what is under test rather than a flaw in the test (does_rollout).
                    e0 = clips[c]["e"][t].float().to(device).unsqueeze(0)
                    e = e0.expand(C, -1, -1)
                    for i in range(h):
                        e = ftm(e, z[:, i])
                    pred = md.body(None, itm(e0.expand(C, -1, -1), e))
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(-1)
                if args.goal_dir:
                    # the demonstration's own (behaviour, level) supplies the matched cross goal;
                    # the rotation supplies the mismatched one
                    dk = clip_key[clips[c]["path"]]
                    gk = clip_key[clips[gc]["path"]]
                    if dk not in cross or gk not in cross:
                        continue
                for tag, source in (("matched", c), ("mm", gc)):
                    if args.mode in ("C", "D"):
                        key = clip_key[clips[source]["path"]]
                        ge = cross_e.get(key)
                        if ge is None:
                            raise SystemExit(
                                # **Skipping here scored 0% on every column with a healthy `n`.**
                                # A missing goal clip silently continued past the counters, so the
                                # run looked like a measurement and was an empty loop.
                                f"no encoded goal clip for {key}; mode {args.mode} needs the goal "
                                "directory encoded before scoring")
                        if t + h >= len(ge):
                            continue
                        g0 = ge[t].float().to(device).unsqueeze(0)
                        g1 = ge[t + h].float().to(device).unsqueeze(0)
                        target_std = md.body(None, itm(g0, g1)).reshape(-1)
                    elif args.goal_dir:
                        key = clip_key[clips[source]["path"]]
                        target_std = torch.tensor(
                            [(float(np.mean(cross[key][1][t:t + h])) - mean) / std],
                            device=device, dtype=torch.float32)
                    else:
                        target_std = torch.tensor(
                            [(float(np.mean(speed[clips[source]["path"]][t:t + h])) - mean) / std],
                            device=device, dtype=torch.float32)
                    # **Distance over every shared channel, not just the first.** With
                    # `body_channels 0 1 2` a one-channel comparison would silently score forward
                    # speed alone and report it as a three-channel result.
                    if pred.dim() == 1:
                        pred = pred.unsqueeze(-1)
                    k = min(pred.shape[-1], target_std.numel())
                    err = (pred[:, :k] - target_std[:k]).pow(2).mean(-1)
                    got = FAMILY(keep[int(err.argmin())])
                    if tag == "matched":
                        hit["matched"] += got == FAMILY(clips[c]["cond"])
                    else:
                        hit["mm_demo"] += got == FAMILY(clips[c]["cond"])
                        gf = FAMILY(clips[gc]["cond"])
                        hit["mm_goal"] += got == gf
                        row = per.setdefault(gf, [0, 0])
                        row[0] += got == gf
                        row[1] += 1
                n += 1
            d = max(n, 1)
            print(f"  {h:>8}{hit['matched']/d:>10.0%}{hit['mm_demo']/d:>21.0%}"
                  f"{hit['mm_goal']/d:>21.0%}{n:>7}   "
                  + "  ".join(f"{k} {v[0]/max(v[1],1):.0%}" for k, v in sorted(per.items())))

    print("\n  chance is 28%. **Only the last column decides** -- the first two are passable by a")
    print("  rule that names the behaviour already visible and never reads the goal (F123, F127).")


if __name__ == "__main__":
    main()
