#!/usr/bin/env bash
# Controlled before/after: adapt OLD (fmd_beh24_s0) and NEW (round1_hexonly_s0_rr) to B1 with the identical eval_suite
# hexonly recipe on the rr data; GPU, sequential.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; CW=data/counterfactual_walks
OUT=results/wm/closed_loop_rr/physics/controlled_turn056/ckpt
declare -A PT=([old]=wm/runs/_kept_for_weekly_slide/fmd_beh24_s0/best.pt [new]=wm/runs/round1_hexonly_s0_rr/best.pt)
for m in old new; do
  echo "=== $m ${PT[$m]} $(date)"
  nice -n 10 $PY -m wm.adapt --ckpt ${PT[$m]} --data $CW/rr_b1_clips_train --embodiment b1 --clips 44 --test_clips 4 --stratify \
      --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0 --out $OUT/${m}_adapted_b1.pt 2>&1 | tail -5
  nice -n 10 $PY -m wm.fit_projector --ckpt $OUT/${m}_adapted_b1.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $CW/rr_b1_clips_train \
      --cache_dir results/wm/cache/fitproj_files --out $OUT/${m}_projector_b1.pt 2>&1 | tail -3
  $PY - $OUT/${m}_adapted_b1.pt $OUT/${m}_projector_b1.pt $OUT/${m}_b1.pt <<'PYEOF'
import sys, os, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3] + ".tmp"); os.replace(sys.argv[3] + ".tmp", sys.argv[3]); print("merged", sys.argv[3])
PYEOF
done
echo ADAPT_DONE
