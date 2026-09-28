"""Is the rollout's read-out the bottleneck? Across the 24 candidate actions from a fixed start state,
correlation with the actions' true Froude over [t, t+k) of four readings (F265 follow-up):

  direct      body(proj(a))                              no frames at all
  read real   body(ITM(e_s, e*_k(s,a)))                  the REAL rendered k-step outcome = perfect FTM
  read pred   body(ITM(e_s, FTM(e_s, proj(a))))          what rollout selection reads
  own-state   body(ITM(e_a[t], e_a[t+k])) on each candidate's own recorded frames (no counterfactual)

States and rendered outcomes: F259's cache (8 library states x 3 steps x 24 actions, 11 steps each).

    .venv/bin/python3 scripts/diagnostics/objective_experiments/readout_bottleneck_check.py \\
        --ckpt stride5=wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.policy.planner import action_chunk_at, load_candidates  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--cf_cache", default="results/wm/cache/counterfactual_horizon.pt")
    ap.add_argument("--emb_cache", default="results/wm/cache/selection_eval_cands.pt")
    args = ap.parse_args()
    dev = "cuda"
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    bm = [np.asarray(load(c["path"], REGISTRY["b1"])["body_motion"])[:, :3] for c in cands]
    CF = torch.load(os.path.join(ROOT, args.cf_cache))
    EM = torch.load(os.path.join(ROOT, args.emb_cache))
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        k = m.stride
        rd = lambda z: m.md.body(None, z).cpu().numpy() * std + mean
        R = {key: [] for key in ("direct", "read real", "read pred", "own-state")}
        for (t, s), rec in sorted(CF.items()):
            e_s = rec["e_s"].float().to(dev).expand(n, -1, -1)
            real = rec["traj"][:, k - 1].float().to(dev)
            a = np.stack([c["actions"][t + m.action_lag] if k == 1 else
                          action_chunk_at(c["actions"], t + m.action_lag, k) for c in cands])
            z = m.proj(torch.as_tensor(a, device=dev), "b1")
            own0 = torch.stack([EM[c["path"]][t].float() for c in cands]).to(dev)
            ownk = torch.stack([EM[c["path"]][t + k].float() for c in cands]).to(dev)
            truth = np.stack([b[t:t + k].mean(0) for b in bm])
            reads = {"direct": rd(z), "read real": rd(m.itm(e_s, real)),
                     "read pred": rd(m.itm(e_s, m.ftm_step(e_s, z))), "own-state": rd(m.itm(own0, ownk))}
            for key, f in reads.items():
                R[key].append([corr(f[:, j], truth[:, j]) for j in range(3)])
        print(f"\n=== {name} (stride {k}) -- r across 24 actions vs true Froude, fwd / lat / yaw")
        for key, v in R.items():
            print(f"  {key:<10}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
