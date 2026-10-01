"""Candidate-selection quality on B1, one harness for every variant, with the library's own ceiling.

Same task and grading as `scripts/figures/plot_direct_vs_rollout.py` (physics goal read at every
`--horizon` steps from one hexapod goal clip per condition; the picked candidate is graded by its
TRUE local Froude at the same offset; mean L2 error, real Froude units), but:

  - scores several checkpoints x read-out windows in one process (encoder loaded once),
  - adds `roll_live`: rollout from the CURRENT frame -- the frame of the candidate executed on the
    previous decision (kinematic replay, the same assumption the grading already makes) -- instead
    of one fixed frame for the whole episode (`roll_fixed`, the plot script's stated simplification),
  - adds the library's own bounds: `oracle` (best candidate at every step by ground truth -- no
    selector can beat it) and `random` (mean over candidates). `gap` = fraction of the
    random-to-oracle gap a selector recovers, the number that says how much is left to win.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py \\
        --ckpt old=wm/runs/.../ckpt_lib_beh12_b1_ego_flat_cleantrain.pt \\
        --ckpt new_zwin=wm/runs/.../ckpt_lib_s4_zwin11.pt@z --windows 0 11

`name=path@z` reads windows by averaging z first then the body head (for heads fit with
`fit_body_head --z_window`); default `@pred` averages the head's per-step outputs.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))

from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import RolloutFroudePlanner, load_candidates  # noqa: E402
from final_2x2x2_test import build_planner  # noqa: E402

CONDITIONS = ["turn_s0.29", "turn_s0.56", "side_L_lvl0", "side_R_lvl1", "speed_c7.1", "speed_c8.8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path[@z|@pred]")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--goal_dir", default="data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 11])
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--modes", nargs="+", default=["direct", "roll_fixed", "roll_live"])
    ap.add_argument("--cache", default="results/wm/cache/selection_eval_cands.pt")
    ap.add_argument("--goal_source", choices=("physics", "vision"), default="physics",
                    help="physics: the goal clip's recorded body_motion (privileged). vision: the goal read "
                         "from the goal clip's frames through each model's own ITM + Froude head, pairs "
                         "(t, t+stride) averaged over a centred --goal_window; grading stays physical")
    ap.add_argument("--goal_window", type=int, default=11)
    ap.add_argument("--goal_cache", default="results/wm/cache/selection_eval_goals.pt")
    ap.add_argument("--rollout_readout", default="",
                    help="npz (coef, intercept; raw Froude) from counterfactual_readout_fit.py --save: "
                         "replaces the body head in the ROLLOUT planner's read-out only (F267)")
    ap.add_argument("--direct_window_align", choices=("centre", "forward"), default="centre",
                    help="direct planner's read-out window: centred on the decision (F251 default) or "
                         "forward from t+lag like the rollout (like-for-like comparison)")
    ap.add_argument("--embodiment", default="b1",
                    help="the controlled body's embodiment (candidates' action space); `hexapod` for a "
                         "held-out hexapod morphology such as c08f09t09")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    h = args.horizon
    cdir = os.path.join(ROOT, args.candidates_dir)
    spec_b1 = REGISTRY[args.embodiment]

    cands = load_candidates(cdir, args.embodiment, per_condition=999)
    motion = [np.asarray(load(c["path"], spec_b1)["body_motion"])[:, :3] for c in cands]

    def local(i, t):
        bm = motion[i]
        o = min(t, len(bm) - 1)
        return bm[o:o + max(min(h, len(bm) - o), 1)].mean(0)

    goals = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            c = str(d["condition"])
        if c in args.conditions and c not in goals:
            goals[c] = np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3]

    # the library's own bounds, independent of any model
    bounds = {}
    for c, g in goals.items():
        steps = list(range(0, len(g) - h, h))
        d = np.array([[np.linalg.norm(local(i, t) - g[t]) for i in range(len(cands))] for t in steps])
        bounds[c] = (d.min(1).mean(), d.mean())

    need_frames = any(m.startswith("roll") for m in args.modes) or args.goal_source == "vision"
    goal_paths = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.goal_dir, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            c = str(d["condition"])
        if c in goals and c not in goal_paths:
            goal_paths[c] = p
    gemb = {}
    if args.goal_source == "vision":
        gc = os.path.join(ROOT, args.goal_cache)
        gemb = torch.load(gc, map_location="cpu") if os.path.exists(gc) else {}
        miss = [p for p in goal_paths.values() if p not in gemb]
        if miss:
            from vjepa2_encoder import VJEPA2FrameEncoder
            encoder = VJEPA2FrameEncoder(dtype=torch.float32)
            for p in miss:
                gemb[p] = encode_clip(encoder, load(p, REGISTRY["hexapod"])["frames"], 2).cpu().half()
            del encoder
            torch.cuda.empty_cache()
            torch.save(gemb, gc)
    emb = {}
    if need_frames:
        cache_path = os.path.join(ROOT, args.cache)
        emb = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
        missing = [c["path"] for c in cands if c["path"] not in emb]
        if missing:
            from vjepa2_encoder import VJEPA2FrameEncoder
            encoder = VJEPA2FrameEncoder(dtype=torch.float32)
            for p in missing:
                emb[p] = encode_clip(encoder, load(p, spec_b1)["frames"], 2).cpu().half()
            del encoder
            torch.cuda.empty_cache()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(emb, cache_path)

    results = {}
    for spec_str in args.ckpt:
        name, rest = spec_str.split("=", 1)
        path, _, avg = rest.partition("@")
        avg = avg or "pred"
        ckpt = os.path.join(ROOT, path)
        planner = build_planner(ckpt, cdir, args.embodiment, h, free_offset=False, device=args.device,
                                per_condition=999)
        planner.window_average = avg
        planner.window_align = args.direct_window_align
        assert [c["path"] for c in planner.candidates] == [c["path"] for c in cands]
        rp, off = None, None
        if need_frames:
            rp = RolloutFroudePlanner.from_checkpoint(ckpt, cdir, embodiment=args.embodiment, horizon=h,
                                                      per_condition=999, device=args.device)
            rp.window_average = avg
            if args.rollout_readout:
                ro = np.load(os.path.join(ROOT, args.rollout_readout))
                W = torch.as_tensor(ro["coef"].T, dtype=torch.float32, device=args.device)
                b = torch.as_tensor(ro["intercept"], dtype=torch.float32, device=args.device)
                mu = torch.as_tensor(rp.mean_s, dtype=torch.float32, device=args.device)
                sd = torch.as_tensor(rp.std_s, dtype=torch.float32, device=args.device)
                # same output units as body_head (standardised Froude), so goals compare unchanged
                rp.md.body = lambda x, z: ((z.float() @ W + b) - mu) / sd
            assert [c["path"] for c in rp.candidates] == [c["path"] for c in cands]
            off = offset_for(torch.load(ckpt, map_location="cpu", weights_only=False), args.embodiment)

        vgoal = {}
        if args.goal_source == "vision":
            ck_ = torch.load(ckpt, map_location="cpu", weights_only=False)
            goff = offset_for(ck_, "hexapod")
            k = rp.stride
            for c, p in goal_paths.items():
                e = gemb[p].float()
                if goff is not None:
                    e = e - goff.float().reshape(1, *e.shape[1:])
                e = e.to(args.device)
                with torch.no_grad():
                    z = rp.itm(e[:-k], e[k:])
                    f = rp.md.body(None, z).float().cpu().numpy() * rp.std_s + rp.mean_s   # (T-k, 3) Froude
                half = args.goal_window // 2
                n = len(goals[c])
                vg = np.stack([f[max(0, t - half):min(len(f), t + half + 1)].mean(0) for t in range(min(n, len(f)))])
                if len(vg) < n:
                    vg = np.concatenate([vg, np.repeat(vg[-1:], n - len(vg), 0)])
                vgoal[c] = vg
                g = goals[c]
                r = [np.corrcoef(vg[:, j], g[:, j])[0, 1] for j in range(3)]
                print(f"  goal read {name} {c:<12} L2 {np.linalg.norm(vg - g, axis=1).mean():.4f}  "
                      f"r {r[0]:+.2f} / {r[1]:+.2f} / {r[2]:+.2f}", flush=True)

        def frame(i, t):
            e = emb[cands[i]["path"]]
            e = e[min(t, len(e) - 1)].float()
            if off is not None:
                e = (e - off.to(e.device)).reshape(e.shape[-2:])
            return e.to(args.device)

        for w in args.windows:
            planner.window = w
            if rp is not None:
                rp.window = w
            for c, g in goals.items():
                g_std = np.stack([planner.standardize(x) for x in (vgoal[c] if vgoal else g)])
                steps = list(range(0, len(g) - h, h))
                for mode in args.modes:
                    errs, prev = [], 0
                    e_fixed = frame(0, 0) if mode == "roll_fixed" else None
                    for t in steps:
                        if mode == "direct":
                            _, i, _, _ = planner.act(g_std[t], t)
                        else:
                            e_t = e_fixed if mode == "roll_fixed" else frame(prev, t)
                            gt = torch.as_tensor(g_std[t], dtype=torch.float32)
                            _, i, _ = rp.act(e_t, gt, t)
                        errs.append(np.linalg.norm(local(i, t) - g[t]))
                        prev = i
                    results[(name, w, mode, c)] = float(np.mean(errs))
            row = [f"{name:<12} w={w:<3}"]
            for mode in args.modes:
                m = np.mean([results[(name, w, mode, c)] for c in goals])
                o = np.mean([bounds[c][0] for c in goals])
                r = np.mean([bounds[c][1] for c in goals])
                row.append(f"{mode} {m:.4f} (gap {(r - m) / max(r - o, 1e-9):+.2f})")
            print("  ".join(row), flush=True)
        del planner, rp
        torch.cuda.empty_cache()

    o = np.mean([bounds[c][0] for c in goals])
    r = np.mean([bounds[c][1] for c in goals])
    print(f"\nlibrary bounds (mean over conditions): oracle {o:.4f}  random {r:.4f}")
    print("gap = (random - selector) / (random - oracle): 0 = random, 1 = oracle, <0 worse than random")
    print("\nper condition:")
    for c in goals:
        cells = "  ".join(f"{n}/w{w}/{m} {v:.4f}" for (n, w, m, cc), v in results.items() if cc == c)
        print(f"  {c:<12} oracle {bounds[c][0]:.4f} random {bounds[c][1]:.4f} | {cells}")


if __name__ == "__main__":
    main()
