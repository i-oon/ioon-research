#!/usr/bin/env bash
# Switching-pretrain test (plan 2026-09-26). Stride-5 pretrains, identical except data/seed:
#   B  beh24 + c10 switching babble      C  beh24 + c10 steady babble (control)      A2  beh24, seed 1
# (A = beh24_stride5_cleansplit, exists.) After each: B1 pipeline (selection + state/action shares),
# c08 zero-shot selection, B1 counterfactual read-out and 5-step prediction checks.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
COMMON="--val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5
 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5
 --epochs 50 --checkpoint_every 2 --resume auto"
BEH=hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
arm() {  # name, extra train args
  local N=$1; shift
  echo "=== $N $(date)"
  $PY -m wm.train $COMMON "$@" --name $N >> results/wm/logs/$N.log 2>&1 || { echo "TRAIN FAILED $N"; return; }
  R=wm/runs/$N
  bash scripts/run/b1_pipeline_eval.sh $R/best.pt $R/b1_lora_c3 $N
  mkdir -p $R/c08_zeroshot
  $PY -m wm.fit_projector --ckpt $R/best.pt --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
      --b1_dir "" --out $R/c08_zeroshot/projector_hex.pt | tail -3
  $PY - $R <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
r = sys.argv[1]
ck = torch.load(f"{r}/best.pt", map_location="cpu", weights_only=False)
sv = torch.load(f"{r}/c08_zeroshot/projector_hex.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv["projector"]; ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{r}/c08_zeroshot/ckpt_lib_zeroshot.pt")
PYEOF
  $PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
      --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat \
      --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout --cache results/wm/cache/selection_eval_c08.pt \
      --ckpt ${N}_c08=$R/c08_zeroshot/ckpt_lib_zeroshot.pt --windows 0 11 | grep -E "w=|bounds"
  $PY scripts/diagnostics/objective_experiments/readout_bottleneck_check.py --ckpt $N=$R/b1_lora_c3/ckpt_lib_s4.pt | grep -v Warning
  $PY scripts/diagnostics/objective_experiments/counterfactual_prediction_k.py --ckpt $N=$R/b1_lora_c3/ckpt_lib_s4.pt --k 5 | grep "k=5"
  echo "=== done $N $(date)"
}
arm beh24sw_stride5_B --sources $BEH hexapod=data/egocentric/babble_c10f10t10_switch_flat
arm beh24st_stride5_C --sources $BEH hexapod=data/egocentric/babble_c10f10t10_steady_flat
arm beh24_stride5_A2 --sources $BEH --seed 1
echo SWITCH_ARMS_DONE
