"""Does the FTM predict the real k-step future of an action taken from a state it did not come from?

F258 judged one step and found every FTM weak (and the cycle term's gain a code, not physics). F259
showed the real future only carries the action over several steps. This scores prediction at k steps
against the rendered counterfactual trajectories F259 cached (`counterfactual_horizon.pt`: from each
library state s at t, every candidate a's motion applied for 11 steps, every frame rendered):

  stride-k model   one FTM step from e_s, driven by proj(a's k-command chunk at t+lag)
  stride-1 model   k FTM steps from e_s, each driven by proj(a's command at t+lag+j)

both compared with the real outcome e*_k(s, a). No ITM in the first two columns:
  dir top1   the action-specific part of the predicted change (centred over the 24 actions) points
             closest to the right action's real change (chance 1/24)
  dir cos    mean cosine between those two action-specific parts
Then, through each model's own ITM + body head (a read, so it can be gamed -- F258):
  pred r     read of the prediction vs the actions' true Froude over [t, t+k)
  real r     the same read of the REAL outcome e*_k -- the prediction should not beat this by much;
             if it does, it carries something the real future does not (F258's pass-through)

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_prediction_k.py \\
        --ckpt stride1=wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_insample.pt \\
        --ckpt stride5=wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_insample.pt --k 5
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

CH = ("forward", "lateral", "yaw")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--k", type=int, default=5, help="frames ahead to predict")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--cache", default="results/wm/cache/counterfactual_horizon.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, K = args.device, args.k
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    bm = [np.asarray(load(c["path"], REGISTRY["b1"])["body_motion"])[:, :3] for c in cands]
    n = len(cands)
    E = torch.load(os.path.join(ROOT, args.cache))

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        s_ = m.stride
        if K % s_:
            raise SystemExit(f"{name}: k={K} is not a multiple of its stride {s_}")
        top1, cosd, pred_r, real_r = [], [], [], []
        for (t, s), rec in sorted(E.items()):
            e_s = rec["e_s"].float().to(dev)
            real = rec["traj"][:, K - 1].float().to(dev)                     # (n, tok, dim)
            if m.offset is not None:
                off = m.offset.float().reshape(e_s.shape[1:]).to(dev)
                e_s, real = e_s - off, real - off
            e = e_s.expand(n, -1, -1)
            for j in range(K // s_):
                start = t + m.action_lag + j * s_
                a = np.stack([c["actions"][start] if s_ == 1 else action_chunk_at(c["actions"], start, s_)
                              for c in cands])
                e = m.ftm_step(e, m.proj(torch.as_tensor(a, device=dev), "b1"))
            dp = (e - e_s).flatten(1)
            dt = (real - e_s).flatten(1)
            dp, dt = dp - dp.mean(0, keepdim=True), dt - dt.mean(0, keepdim=True)
            C = torch.nn.functional.normalize(dp, dim=1) @ torch.nn.functional.normalize(dt, dim=1).T
            top1.append((C.argmax(1) == torch.arange(n, device=dev)).float().mean().item())
            cosd.append(C.diag().mean().item())
            # read at the model's own trained spacing: one ITM read per stride-sized hop is not
            # available for the prediction (only its endpoint), so read endpoint pairs at K; for a
            # stride-K model that is its trained spacing, for stride 1 it is K apart (stated caveat)
            truth = np.stack([b[t:t + K].mean(0) for b in bm])
            fp = m.md.body(None, m.itm(e_s.expand(n, -1, -1), e)).cpu().numpy() * std + mean
            fr = m.md.body(None, m.itm(e_s.expand(n, -1, -1), real)).cpu().numpy() * std + mean
            pred_r.append([corr(fp[:, j], truth[:, j]) for j in range(3)])
            real_r.append([corr(fr[:, j], truth[:, j]) for j in range(3)])
        f = lambda x: " / ".join(f"{v:.2f}" for v in np.nanmean(np.asarray(x), 0))
        print(f"{name:10s} stride {s_}  k={K}: dir top1 {np.mean(top1):.3f} (chance {1 / n:.3f})  "
              f"dir cos {np.mean(cosd):+.3f}  | pred r {f(pred_r)}  real r {f(real_r)}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
