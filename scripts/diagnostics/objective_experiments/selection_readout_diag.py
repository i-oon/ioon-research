"""Where does rollout-based selection lose the candidates? Per decision step, does each read-out's
predicted Froude track the candidates' TRUE Froude across candidates?

`selection_eval.py` measured rollout selection at or below random (library gap -0.13 to -0.25)
while `direct` closes 57-77% of it. This separates the read-out from the forward model:

  direct       body(proj(a))                                   no FTM, no frames
  itm_true     body(ITM(e_i[t], e_i[t+1]))                     candidate's REAL next frame: a perfect FTM
  roll_own     body(ITM(e_i[t], FTM(e_i[t], proj(a_i))))       FTM from the candidate's own state
  roll_shared  body(ITM(e_s[t], FTM(e_s[t], proj(a_i))))       FTM from one shared state (what selection does)

Per step t and body channel: Pearson r across candidates between prediction and truth (the
candidate's own smoothed body_motion at t), averaged over steps; plus the across-candidate spread
of the prediction relative to the truth's. If itm_true tracks and roll_* does not, the FTM is where
candidate identity is lost; if itm_true also fails, the ITM/head read-out is.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/selection_readout_diag.py \\
        --ckpt old=wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/ckpt_lib_beh12_b1_ego_flat_cleantrain.pt
"""
import argparse
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
from wm.evaluate import offset_for  # noqa: E402
from wm.policy.planner import RolloutFroudePlanner, load_candidates  # noqa: E402
from final_2x2x2_test import build_planner  # noqa: E402

CH = ["forward", "lateral", "yaw"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--cache", default="results/wm/cache/selection_eval_cands.pt")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, h = args.device, args.horizon
    cdir = os.path.join(ROOT, args.candidates_dir)
    spec = REGISTRY["b1"]
    cands = load_candidates(cdir, "b1", per_condition=999)
    motion = [np.asarray(load(c["path"], spec)["body_motion"])[:, :3] for c in cands]
    emb = torch.load(os.path.join(ROOT, args.cache), map_location="cpu")
    n_c = len(cands)
    T = min(min(len(m) for m in motion), min(len(emb[c["path"]]) for c in cands)) - h - 2

    for spec_str in args.ckpt:
        name, path = spec_str.split("=", 1)
        ckpt = os.path.join(ROOT, path)
        dp = build_planner(ckpt, cdir, "b1", h, free_offset=False, device=dev, per_condition=999)
        rp = RolloutFroudePlanner.from_checkpoint(ckpt, cdir, embodiment="b1", horizon=h,
                                                  per_condition=999, device=dev)
        lag = dp.action_lag
        off = offset_for(torch.load(ckpt, map_location="cpu", weights_only=False), "b1")

        def fr(tt):
            e = torch.stack([emb[c["path"]][tt].float() for c in cands])
            if off is not None:
                e = e - off.float().reshape(e.shape[1:])
            return e.to(dev)
        std = torch.as_tensor(dp.std_s, dtype=torch.float32)
        mean = torch.as_tensor(dp.mean_s, dtype=torch.float32)
        rs = {m: [] for m in ("direct", "itm_true", "roll_own", "roll_shared")}
        spread = {m: [] for m in rs}
        for t in range(0, T, h):
            truth = np.stack([m[t:t + h].mean(0) for m in motion])            # (n_c, 3)
            e_now, e_next = fr(t), fr(t + 1)
            a = torch.as_tensor(np.stack([c["actions"][t + lag] for c in cands]), device=dev)
            z_a = dp.proj(a, "b1")
            preds = {
                "direct": dp.md.body(None, z_a),
                "itm_true": dp.md.body(None, rp.itm(e_now, e_next)),
                "roll_own": dp.md.body(None, rp.itm(e_now, rp.ftm(e_now, z_a))),
                "roll_shared": dp.md.body(None, rp.itm(e_now[:1].expand(n_c, -1, -1),
                                                       rp.ftm(e_now[:1].expand(n_c, -1, -1), z_a))),
            }
            for m, p in preds.items():
                p = (p.cpu() * std + mean).numpy()                             # Froude units
                for j in range(3):
                    if truth[:, j].std() > 1e-9 and p[:, j].std() > 1e-12:
                        rs[m].append((j, np.corrcoef(p[:, j], truth[:, j])[0, 1]))
                    spread[m].append((j, p[:, j].std() / max(truth[:, j].std(), 1e-9)))
        print(f"\n=== {name} ===  (across {n_c} candidates, {len(range(0, T, h))} steps)")
        print(f"{'read-out':<12}" + "".join(f"{'r ' + c:>12}" for c in CH)
              + "".join(f"{'spread ' + c:>15}" for c in CH))
        for m in rs:
            r = [np.mean([v for j, v in rs[m] if j == k]) for k in range(3)]
            s = [np.mean([v for j, v in spread[m] if j == k]) for k in range(3)]
            print(f"{m:<12}" + "".join(f"{x:>12.3f}" for x in r) + "".join(f"{x:>15.3f}" for x in s))
        del dp, rp
        torch.cuda.empty_cache()
    print("\nr: mean across-candidate Pearson correlation between prediction and truth, per step.")
    print("spread: across-candidate std of prediction / std of truth (0 = predicts the same for all).")


if __name__ == "__main__":
    main()
