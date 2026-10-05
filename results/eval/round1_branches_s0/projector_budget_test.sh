#!/usr/bin/env bash
# F311 follow-up: does switching data fix B1 yaw at the SAME adaptation budget (~2,900 B1 frames)?
#   a: 44 steady clips   b: 22 clips + 47 branch files   c: 94 branch files   (branch files = 31 frames, stand-in for babble with switches)
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python3; R=results/eval/round1_branches_s0; CW=data/counterfactual_walks
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
run() {  # tag, extra args
  tag=$1; shift
  echo "=== fit $tag: $*"
  $PY -m wm.fit_projector --ckpt wm/runs/round1_branches_s0/best.pt --hex_dir $CW/c10_clips_train \
      --cache results/wm/cache/fitproj_v4.pt --objective rollout --out $R/ckpt/projector_$tag.pt "$@" 2>&1 \
      | grep -E "^b1|^hexapod|files|rollout epoch +15|^b1 +[0-9]" 
  $PY - "$tag" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
t = sys.argv[1]
ck = torch.load("wm/runs/round1_branches_s0/best.pt", map_location="cpu", weights_only=False)
sv = torch.load(f"results/eval/round1_branches_s0/ckpt/projector_{t}.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv["projector"]; ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"results/eval/round1_branches_s0/ckpt/both_{t}.pt")
PYEOF
  $PY $RO --embodiment b1 --cf_dir $CW/b1_branches_heldout --ckpt $tag=$R/ckpt/both_$tag.pt --pairs 11 2>&1 | grep -A3 "^==="
}
run a44clips   --b1_dir $CW/b1_clips_train --b1_n 44
run b22c47br   --b1_dir $CW/b1_clips_train --b1_n 22 --b1_extra_dir $CW/b1_branches_train --b1_extra_n 47
run c94br      --b1_dir "" --b1_extra_dir $CW/b1_branches_train --b1_extra_n 94
echo BUDGET_TEST_DONE
