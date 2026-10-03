#!/usr/bin/env bash
# Held-out hexapod morphology c08f09t09, zero-shot: no c08 data anywhere. Per stride: the pretrain
# checkpoint + a hexapod projector fit on the pretraining body's own clips (c10f10t10) + the pretrain
# body head. Goals: the six c10f10t10 held-out clips (same as the B1 test); candidates: c08f09t09's
# 48 clips. Runs after the stride-10 chain (one GPU job at a time).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
until grep -q STRIDE10_ALL_DONE results/wm/logs/stride10_full_chain.log 2>/dev/null; do sleep 120; done
declare -A PT=([stride1]=wm/runs/beh24_hinge_cleansplit/best.pt
               [stride5]=wm/runs/beh24_stride5_cleansplit/best.pt
               [stride10]=wm/runs/beh24_stride10_cleansplit/best.pt)
ARGS=()
for s in stride1 stride5 stride10; do
  D=$(dirname ${PT[$s]})/c08_zeroshot; mkdir -p $D
  $PY -m wm.fit_projector --ckpt ${PT[$s]} --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
      --b1_dir "" --out $D/projector_hex.pt | tail -4
  $PY - ${PT[$s]} $D <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
pt, d = sys.argv[1], sys.argv[2]
ck = torch.load(pt, map_location="cpu", weights_only=False)
sv = torch.load(f"{d}/projector_hex.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv["projector"]; ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{d}/ckpt_lib_zeroshot.pt"); print("merged", d)
PYEOF
  ARGS+=(--ckpt ${s}_c08=$D/ckpt_lib_zeroshot.pt)
done
$PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
    --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat \
    --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
    --cache results/wm/cache/selection_eval_c08.pt "${ARGS[@]}" --windows 0 11 \
    | tee results/deck/window_sweep/c08_zeroshot_selection.txt
echo C08_DONE
