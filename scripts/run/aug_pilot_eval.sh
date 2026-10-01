#!/usr/bin/env bash
# Evaluate the augmentation pilots (scripts/run/aug_pilot.sh): fit the B1 projector (Stage 2 only; B1 is in
# pretraining), then B1 selection on (1) the v3 library = the training rendering ("clean", must not drop) and
# (2) the old-render library, never trained on, no adaptation ("rendering change", robustness).
#   bash scripts/run/aug_pilot_eval.sh pilot_augnorm_s0 pilot_augsafe_s0 pilot_augvsm_s0
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
CK=()
for N in "$@"; do
  D=wm/runs/$N/eval; mkdir -p $D
  $PY -m wm.fit_projector --ckpt wm/runs/$N/best.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt \
      --out $D/projector.pt 2>&1 | grep -v -i warn | tail -1
  $PY - wm/runs/$N/best.pt $D/projector.pt $D/ckpt.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
  CK+=(--ckpt "$N=$D/ckpt.pt")
done
echo "=== B1, training rendering (v3 library)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir data/egocentric_v3/beh12_b1_ego_flat_cleantrain \
    --cache results/wm/cache/selection_eval_cands_v3.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo "=== B1, rendering change (old-render library, no adaptation)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir data/egocentric/beh12_b1_ego_flat_cleantrain \
    --cache results/wm/cache/selection_eval_cands.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo AUG_PILOT_EVAL_DONE
