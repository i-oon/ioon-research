#!/usr/bin/env bash
# Workstation: evaluate pretrained runs copied back from the server
#   rsync -a ioon@SERVER:ioon-research/wm/runs/NAME wm/runs/
# through the same evaluation as every other arm (switch_pretrain_arms.sh's arm() minus training):
# B1 pipeline + eta^2 grid, c08 zero-shot, counterfactual read-out, 5-step prediction. Needs best.pt.
#   bash scripts/run/detach_eval_local.sh dt0_beh24_s0 dt0_beh24_s1
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
evalrun() {
  local N=$1 R=wm/runs/$1
  [ -f $R/best.pt ] || { echo "NO $R/best.pt"; return; }
  echo "=== $N $(date)"
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
for N in "$@"; do evalrun $N; done
echo DETACH_EVAL_DONE
