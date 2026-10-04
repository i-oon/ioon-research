#!/usr/bin/env bash
# F311 fix test: projector fitted through the frozen FTM + ITM + head (--objective rollout) vs the z fit (b1.pt / hex.pt).
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python3; R=results/eval/round1_branches_s0; CW=data/counterfactual_walks
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
$PY -m wm.fit_projector --ckpt wm/runs/round1_branches_s0/best.pt --hex_dir $CW/c10_clips_train --b1_dir $CW/b1_clips_train \
    --cache results/wm/cache/fitproj_v4.pt --objective rollout --out $R/ckpt/projector_rollout.pt 2>&1 | grep -v -i "warn\|Loading weights"
$PY - <<'PYEOF'
import torch
from wm.models.action_projector import action_dims_from
ck = torch.load("wm/runs/round1_branches_s0/best.pt", map_location="cpu", weights_only=False)
sv = torch.load("results/eval/round1_branches_s0/ckpt/projector_rollout.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv["projector"]; ck["action_dims"] = action_dims_from(sv)
torch.save(ck, "results/eval/round1_branches_s0/ckpt/both_rollout.pt")
PYEOF
for b in hexapod:c10 b1:b1; do
  echo "--- read-out ${b#*:} (rollout-fitted projector)"
  $PY $RO --embodiment ${b%%:*} --cf_dir $CW/${b#*:}_branches_heldout --ckpt rollproj=$R/ckpt/both_rollout.pt --pairs 1 11 2>&1 | grep -v -i "warn\|Loading weights" | grep -A3 "^==="
done
echo ROLLOUT_PROJ_DONE
