"""Rank candidates on predicted body motion directly, skipping the embedding rollout.

    .venv/bin/python3 scripts/diagnostics/planning/rank_on_delta_state.py

`target_action_share.py` found that `z` carries the action's effect on **body motion** (ridge R2
0.359 from `z` alone) and almost nothing about the **embedding displacement** (0.005). F179's ranker
already uses the planning `z` and already scores in body-motion units, but it gets there the long
way:

    F179       proj(a) -> FTM rollout -> ITM -> body head -> delta-state
    direct     proj(a) ------------------------> body head -> delta-state

The embedding round-trip is exactly where the measurements say the action is drowned. This scores
both paths on the same candidates.

**Two arms, and only the first has ground truth without a simulator.**

    coarse     the twelve recorded conditions. Their true body motion is recorded, so the correct
               answer is known offline and this is a real ranking test.
    fine       0.5 sd perturbations, F179's setting. Their true outcome needs the simulator, so this
               file reports only how far apart the two paths place them -- **separation, not
               accuracy**. A path that cannot separate them cannot rank them, so a collapse here
               settles the question cheaply; a spread here does not settle it the other way.

**The direct path sees no state.** The body head is frame-blind and `proj` takes only the action, so
this scores what an action produces on average rather than what it produces from `e_t`. That is a
real limitation of the arm, not an oversight, and it is why the world model exists at all.

**Diagnosis only; trains nothing, and runs no simulator.**
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from teacher_student_insect import load_teacher  # noqa: E402
from wm.adapt3 import gather  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402

FAMILY = lambda c: "side" if c.startswith("side") else c.split("_")[0]  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--states", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu", mmap=True)
    clips = gather(os.path.join(ROOT, args.data), "hexapod", None, ck, cache, 2,
                   max(1, cfg.action_lag), device)

    # one clip per condition, with its recorded actions and its true mean body motion
    cand, truth = {}, {}
    for p in paths:
        d = load(p, REGISTRY["hexapod"])
        with np.load(p, allow_pickle=True) as z_:
            cond = str(z_["condition"])
        if cond not in cand:
            cand[cond] = torch.tensor(np.asarray(d["actions"]), dtype=torch.float32)
            truth[cond] = np.asarray(d["body_motion"])[:, channels].mean(0)
    conds = sorted(cand)
    print(f"{args.teacher}\n{len(clips)} clips, {len(conds)} conditions, channels {channels}\n")

    allacts = np.concatenate([np.asarray(load(p, REGISTRY["hexapod"])["actions"], np.float64)
                              for p in paths])
    joint_sd = torch.tensor(allacts.std(0), dtype=torch.float32, device=device)

    def delta_direct(a):
        return md.body(None, proj(a, "hexapod"))

    def delta_f179(a, e_t):
        z = proj(a, "hexapod")
        roll = e_t.expand(len(a), -1, -1)
        for _ in range(args.horizon):
            roll = ftm(roll, z)
        return md.body(None, itm(e_t.expand(len(a), -1, -1), roll))

    # ---- coarse: rank the twelve recorded conditions, ground truth known ----------------------
    hits = collections.Counter(); tot = 0
    chance_fam = np.mean([sum(FAMILY(c2) == FAMILY(c) for c2 in conds) / len(conds)
                          for c in conds])
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for gi, goal_cond in enumerate(conds):
            goal = torch.tensor((truth[goal_cond] - mean_s) / std_s, dtype=torch.float32,
                                device=device)
            for ci, c in enumerate(clips[:24]):
                e = c["e"].float()
                for t in range(5, min(len(e) - args.horizon - 1, 50), 15):
                    e_t = e[t:t + 1].to(device)
                    a = torch.stack([cand[k][min(t, len(cand[k]) - 1)] for k in conds]).to(device)
                    for tag, m in (("direct", delta_direct(a)),
                                   ("f179", delta_f179(a, e_t))):
                        if m.dim() == 1:
                            m = m.unsqueeze(-1)
                        k = min(m.shape[-1], len(channels))
                        pick = conds[int((m[:, :k] - goal[:k]).pow(2).mean(-1).argmin())]
                        hits[f"{tag}_exact"] += pick == goal_cond
                        hits[f"{tag}_family"] += FAMILY(pick) == FAMILY(goal_cond)
                    tot += 1
    print("  COARSE -- pick the condition whose recorded body motion matches the goal")
    print(f"  {'scoring path':>16}{'exact':>10}{'family':>10}{'n':>8}")
    for tag in ("direct", "f179"):
        print(f"  {tag:>16}{hits[f'{tag}_exact'] / max(tot, 1):>10.0%}"
              f"{hits[f'{tag}_family'] / max(tot, 1):>10.0%}{tot:>8}")
    print(f"  {'chance':>16}{1 / len(conds):>10.0%}{chance_fam:>10.0%}\n")

    # ---- fine: separation only, no ground truth without a simulator ---------------------------
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    seps = collections.defaultdict(list)
    with torch.no_grad():
        for c in clips[:args.states]:
            e = c["e"].float()
            t = 6
            e_t = e[t:t + 1].to(device)
            base = cand[c["cond"]][min(t, len(cand[c["cond"]]) - 1)].to(device).unsqueeze(0)
            pert = base + (args.sigma * joint_sd) * torch.randn(
                args.samples, base.shape[-1], generator=g).to(device)
            allc = torch.cat([base, pert])
            pool = torch.stack([cand[k][min(t, len(cand[k]) - 1)] for k in conds]).to(device)
            for tag, fn in (("direct", lambda a: delta_direct(a)),
                            ("f179", lambda a: delta_f179(a, e_t))):
                mp, mc = fn(allc), fn(pool)
                mp = mp if mp.dim() > 1 else mp.unsqueeze(-1)
                mc = mc if mc.dim() > 1 else mc.unsqueeze(-1)
                seps[f"{tag}_pert"].append(float(mp.std(0).mean()))
                seps[f"{tag}_cond"].append(float(mc.std(0).mean()))
    print("  FINE -- how far apart the paths place 0.5 sd perturbations (separation, not accuracy)")
    print(f"  {'scoring path':>16}{'perturbations':>16}{'conditions':>13}{'ratio':>9}")
    for tag in ("direct", "f179"):
        p, cc = np.mean(seps[f"{tag}_pert"]), np.mean(seps[f"{tag}_cond"])
        print(f"  {tag:>16}{p:>16.4f}{cc:>13.4f}{p / max(cc, 1e-9):>9.3f}")
    print("\n  ratio is perturbation spread over condition spread, in standardised body-motion")
    print("  units. A path near zero cannot rank these however good its accuracy is elsewhere.")


if __name__ == "__main__":
    main()
