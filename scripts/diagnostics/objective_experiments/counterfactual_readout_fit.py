"""Can a read-out learn to read counterfactual transitions? (F266 follow-up)

F266: body(ITM(e_s, outcome)) reads the actions' Froude poorly when the action did not come from the
start state s, even for the real rendered outcome. Here a fresh ridge read-out is fitted on
counterfactual pairs and tested on held-out START STATES (leave-one-state-out over F259's 8 library
states; each state has 3 steps x 24 actions):

  A  fit on z = ITM(e_s, real outcome), test on held-out states' real outcomes
  B  head A applied to z = ITM(e_s, FTM(e_s, proj(a)))  (the rollout, trained on real)
  C  fit and test on the rollout's z directly (trained on imagined)
  -- baseline: the checkpoint's own body head on the same z (real / rollout)

Score: correlation across the 24 actions with their true Froude over [t, t+k), per state-step, mean.
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.policy.planner import action_chunk_at, load_candidates  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--cf_cache", default="results/wm/cache/counterfactual_horizon.pt")
    ap.add_argument("--save", default="", help="also fit the rollout-z head on ALL states and save it "
                    "(npz: coef, intercept) for `selection_eval.py --rollout_readout`")
    args = ap.parse_args()
    dev = "cuda"
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    bm = [np.asarray(load(c["path"], REGISTRY["b1"])["body_motion"])[:, :3] for c in cands]
    CF = torch.load(os.path.join(ROOT, args.cf_cache))
    m = Models(os.path.join(ROOT, args.ckpt), "b1", cands[0]["actions"].shape[1], dev)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
    k = m.stride
    rows = []                                    # (state, t, z_real, z_roll, truth, base_real, base_roll)
    for (t, s), rec in sorted(CF.items()):
        e_s = rec["e_s"].float().to(dev).expand(n, -1, -1)
        real = rec["traj"][:, k - 1].float().to(dev)
        a = np.stack([action_chunk_at(c["actions"], t + m.action_lag, k) if k > 1 else
                      c["actions"][t + m.action_lag] for c in cands])
        zr = m.itm(e_s, real)
        zi = m.itm(e_s, m.ftm_step(e_s, m.proj(torch.as_tensor(a, device=dev), "b1")))
        rd = lambda z: m.md.body(None, z).cpu().numpy() * std + mean
        rows.append((s, t, zr.cpu().numpy(), zi.cpu().numpy(), np.stack([b[t:t + k].mean(0) for b in bm]),
                     rd(zr), rd(zi)))
    states = sorted({r[0] for r in rows})
    score = {key: [] for key in ("base real", "A real", "base rollout", "B rollout (head fit on real)",
                                 "C rollout (head fit on rollout)")}
    for s in states:
        tr = [r for r in rows if r[0] != s]
        te = [r for r in rows if r[0] == s]
        Xr = np.concatenate([r[2] for r in tr]); Xi = np.concatenate([r[3] for r in tr])
        Y = np.concatenate([r[4] for r in tr])
        hr = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xr, Y)
        hi = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xi, Y)
        for r in te:
            truth = r[4]
            for key, pred in (("base real", r[5]), ("A real", hr.predict(r[2])), ("base rollout", r[6]),
                              ("B rollout (head fit on real)", hr.predict(r[3])),
                              ("C rollout (head fit on rollout)", hi.predict(r[3]))):
                score[key].append([corr(pred[:, j], truth[:, j]) for j in range(3)])
    print(f"stride {k}, {len(states)} held-out states x {len(rows) // len(states)} steps x {n} actions;"
          f" r across actions vs true Froude, fwd / lat / yaw")
    for key, v in score.items():
        print(f"  {key:<34}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)))
    if args.save:
        h = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(np.concatenate([r[3] for r in rows]),
                                                       np.concatenate([r[4] for r in rows]))
        np.savez(os.path.join(ROOT, args.save), coef=h.coef_, intercept=h.intercept_, alpha=h.alpha_,
                 ckpt=args.ckpt, stride=k)
        print(f"-> {args.save} (alpha {h.alpha_:.3g}, fit on all {len(rows)} state-steps)")


if __name__ == "__main__":
    main()
