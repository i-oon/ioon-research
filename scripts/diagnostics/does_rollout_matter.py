"""Does rolling the forward model change which action the planner picks?

**The planner's cost is almost entirely the rollout** -- twelve candidates times five steps is
sixty forward-model calls per control step, against one for a policy. This asks whether that buys
anything, because three separate results suggest it may not:

  * sweeping the planning horizon 1/3/5/10 on the B1 moved behaviour accuracy around with no
    ordering -- ten steps rolls *worse* than hold-still and plans no worse than five;
  * the forward model's rollout accuracy and its ability to rank actions came apart everywhere
    they were measured together (F98);
  * and if the answer is "no", the world model in this loop is a similarity function, not a
    predictor, and the honest next step is to train a policy rather than search at run time.

Three scoring rules, same candidates, same held-out clips, argmin of each:

    rollout    roll the FDM h steps on proj(a), score against the goal frame   what we do now
    direct     score proj(a) against ITM(e_t, e_goal) -- **no forward model at all**
    blind      score proj(a) against the mean latent -- **ignores the goal entirely**

`blind` is the control that matters. It cannot be right for a reason, so whatever it scores is the
rate reachable without using the goal, and any rule that fails to beat it is not planning.

    .venv/bin/python3 scripts/diagnostics/does_rollout_matter.py \\
        --ckpt wm/runs/beh12_hexonly/stage3_b1_nce.pt \\
        --projector wm/runs/beh12_hexonly/projector_stage3_nce.pt --data data/beh12_b1_flat
"""
import argparse
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
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


@torch.no_grad()
def evaluate(clips, val, cand, proj, itm, ftm, name, h, device, limit=240, seed=0):
    conds = sorted(cand)
    g = torch.Generator().manual_seed(seed)
    picks = val if len(val) <= limit else [val[i] for i in
                                           torch.randperm(len(val), generator=g)[:limit].tolist()]
    hit = {k: 0 for k in ("rollout", "direct", "blind")}
    n = 0
    z_bar = None
    for c, t in picks:
        if t + h >= clips[c]["n"]:
            continue
        acts, keep = [], []
        for k in conds:
            src = cand[k]
            if t + h < clips[src]["n"]:
                acts.append(torch.stack([clips[src]["a"][t + i] for i in range(h)]))
                keep.append(k)
        if len(keep) < 2:
            continue
        e_t = clips[c]["e"][t].float().to(device).unsqueeze(0)
        e_goal = clips[c]["e"][t + h].float().to(device).unsqueeze(0)
        a = torch.stack(acts).to(device)                       # C x h x action_dim
        C = len(keep)
        z = proj(a.reshape(C * h, -1), name).reshape(C, h, -1)
        if z_bar is None:
            z_bar = z.mean((0, 1), keepdim=False).unsqueeze(0)

        e = e_t.expand(C, -1, -1)
        for i in range(h):
            e = ftm(e, z[:, i])
        err_roll = ((e - e_goal) ** 2).flatten(1).mean(1)

        # **The goal enters through the ITM instead of the forward model.** `z_goal` is what the
        # inverse model reads off the pair (now, goal) -- at h > 1 that is a wider pair than it was
        # trained on, which is part of what is being tested rather than a flaw in the test.
        z_goal = itm(e_t, e_goal)
        err_direct = ((z[:, 0] - z_goal) ** 2).mean(1)
        err_blind = ((z[:, 0] - z_bar) ** 2).mean(1)

        truth = FAMILY(clips[c]["cond"])
        for key, err in (("rollout", err_roll), ("direct", err_direct), ("blind", err_blind)):
            hit[key] += FAMILY(keep[int(err.argmin())]) == truth
        n += 1
    return {k: v / max(n, 1) for k, v in hit.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 3, 5, 10])
    ap.add_argument("--cache", default="results/wm/cache/b1_all.pt")
    ap.add_argument("--chunk", type=int, default=2)
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
    saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, saved["action_dims"]).to(device).eval()
    proj.load_state_dict(saved["projector"])

    cand, seen = {}, set()
    for i, c in enumerate(clips):
        if c["cond"] not in seen:
            cand[c["cond"]] = i; seen.add(c["cond"])
    val = [(i, t) for i, c in enumerate(clips) if i not in cand.values()
           for t in range(c["n"])]
    fam_chance = np.mean([sum(FAMILY(k) == FAMILY(clips[i]["cond"]) for k in cand) / len(cand)
                          for i, _t in val[::37]])
    print(f"{len(clips)} clips | {len(cand)} candidates | scored on {len(val)} transitions "
          f"from clips no candidate came from\n")
    print(f"  {'horizon':>8}{'rollout':>10}{'direct':>9}{'blind':>8}{'n':>7}")
    for h in args.horizons:
        r, n = evaluate(clips, val, cand, proj, itm, ftm, name, h, device)
        print(f"  {h:>8}{r['rollout']:>10.0%}{r['direct']:>9.0%}{r['blind']:>8.0%}{n:>7}")
    print(f"\n  chance for the family score is {fam_chance:.0%}.\n")
    print("`rollout` is the planner as built. `direct` deletes the forward model and matches the")
    print("projected action against the inverse model's reading of (now, goal). `blind` ignores the")
    print("goal, so it is the rate available without planning at all.\n")
    print("**If `direct` matches `rollout`, sixty forward-model calls per control step buy nothing**")
    print("and the world model is acting as a similarity function. **If both match `blind`, the")
    print("goal is not being used** and neither rule is planning.")


if __name__ == "__main__":
    main()
