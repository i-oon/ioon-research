#!/usr/bin/env bash
# Full B1 pipeline for one pretrained checkpoint, then the rollout diagnostics and selection eval.
# Same recipe as wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3 (F249/F250), so arms that
# differ only in pretraining are compared through identical Stages 1, 2 and 4.
#   Stage 1  wm.adapt          LoRA rank 2 (default), 3 clips stratified, hinge 0.5
#   Stage 2  wm.fit_projector
#   Stage 4  wm.fit_body_head  with hexapod rehearsal
# then: state x action decomposition (hexapod val, pretrained; b1 library, adapted) and
# selection_eval (direct / roll_fixed / roll_live at w 0, 11).
#
#   bash scripts/run/b1_pipeline_eval.sh <pretrained.pt> <out_dir> <name>
set -euo pipefail
cd "$(dirname "$0")/../.."
PT=$1; D=$2; NAME=$3
PY=.venv/bin/python3
DIAG=scripts/diagnostics/objective_experiments
mkdir -p "$D"
LOG="$D/pipeline.log"
exec > >(grep --line-buffered -v Warning | tee -a "$LOG") 2>&1
echo "=== $NAME: $PT -> $D ($(date)) ==="

echo "--- pretrained, hexapod val, true z"
$PY $DIAG/rollout_state_action_anova.py --z_source true --embodiment hexapod \
    --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
    --cache results/wm/cache/anova_hex_beh24val.pt --ckpt "$NAME=$PT" | tail -6

$PY -m wm.adapt --ckpt "$PT" --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 \
    --clips 3 --stratify --lambda_hinge 0.5 --hinge_margin 0.1 --out "$D/adapted_b1.pt" | tail -8
$PY -m wm.fit_projector --ckpt "$D/adapted_b1.pt" \
    --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
    --b1_dir data/egocentric/beh24_b1_ego_flat_cleantrain --out "$D/projector_b1.pt" | tail -4
$PY -m wm.fit_body_head --ckpt "$D/adapted_b1.pt" --data data/egocentric/beh24_b1_ego_flat_cleantrain \
    --embodiment b1 --also hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
    --out "$D/body_head_b1_hex.pt" | tail -6
$PY - "$D" <<'EOF'
import sys, torch
from wm.models.action_projector import action_dims_from
d = sys.argv[1]
ck = torch.load(f"{d}/body_head_b1_hex.pt", map_location="cpu", weights_only=False)
sv = torch.load(f"{d}/projector_b1.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv)
ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{d}/ckpt_lib_s4.pt")
print("merged ->", f"{d}/ckpt_lib_s4.pt")
EOF

for zs in true proj; do
    echo "--- adapted, b1 library, z_source=$zs"
    $PY $DIAG/rollout_state_action_anova.py --z_source $zs --ckpt "$NAME=$D/ckpt_lib_s4.pt" | tail -6
done
echo "--- selection"
$PY $DIAG/selection_eval.py --ckpt "$NAME=$D/ckpt_lib_s4.pt" --windows 0 11 | head -4
echo "=== done $NAME ($(date)) ==="
