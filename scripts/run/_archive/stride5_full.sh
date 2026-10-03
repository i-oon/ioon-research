#!/usr/bin/env bash
# Stride-5 pretrain from scratch (F259) + the identical B1 pipeline, and the stride-1 baseline's
# in-sample refit so both are compared the same way. Same data and losses as
# beh24_hinge_cleansplit; the only change is --frame_stride 5 (action_chunk follows it).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
$PY -m wm.train --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
    --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
    --lambda_body 0.5 --lambda_hinge 0.5 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 \
    --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 50 --checkpoint_every 2 \
    --name beh24_stride5_cleansplit --resume auto >> results/wm/logs/beh24_stride5_cleansplit.log 2>&1
bash scripts/run/b1_pipeline_eval.sh wm/runs/beh24_stride5_cleansplit/best.pt \
    wm/runs/beh24_stride5_cleansplit/b1_lora_c3 stride5
# in-sample Stages 2 and 4 (fit on the candidate library, as OLD was built) for both strides
for D in wm/runs/beh24_stride5_cleansplit/b1_lora_c3 wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3; do
  $PY -m wm.fit_projector --ckpt $D/adapted_b1.pt --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
      --b1_dir data/egocentric/beh12_b1_ego_flat_cleantrain --out $D/projector_lib.pt | tail -3
  $PY -m wm.fit_body_head --ckpt $D/adapted_b1.pt --data data/egocentric/beh12_b1_ego_flat_cleantrain \
      --embodiment b1 --also hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
      --out $D/body_head_lib.pt | tail -3
  $PY - $D <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
d = sys.argv[1]
ck = torch.load(f"{d}/body_head_lib.pt", map_location="cpu", weights_only=False)
sv = torch.load(f"{d}/projector_lib.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{d}/ckpt_lib_insample.pt"); print("merged", d)
PYEOF
done
$PY scripts/diagnostics/objective_experiments/selection_eval.py \
    --ckpt stride1_in=wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_insample.pt \
    --ckpt stride5_in=wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_insample.pt \
    --ckpt stride1_out=wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_s4.pt \
    --ckpt stride5_out=wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt \
    --windows 0 11 | tee results/deck/window_sweep/stride5_selection.txt
echo STRIDE5_ALL_DONE
