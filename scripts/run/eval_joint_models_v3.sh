#!/usr/bin/env bash
# Evaluate jointly pretrained (hexapod + B1) models as joint models: Stage 2 only (projectors for both
# bodies from pretraining's ITM), then B1 selection, c08 zero-shot, c10 same body, and shared-latent
# numbers. B1 on the v3 rendering. Sequential.   bash scripts/run/eval_joint_models_v3.sh RUN [RUN ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
CK=(); M=()
for N in "$@"; do
  R=wm/runs/$N; J=$R/stage2only; mkdir -p $J
  if [ ! -f $J/ckpt_stage2only.pt ]; then
    $PY -m wm.fit_projector --ckpt $R/best.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt --out $J/projector.pt 2>&1 | grep -v -i warn | tail -1
    $PY - $R/best.pt $J/projector.pt $J/ckpt_stage2only.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3]); print("merged ->", sys.argv[3])
PYEOF
  fi
  CK+=(--ckpt "$N=$J/ckpt_stage2only.pt"); M+=(--model "$N=$R/best.pt")
done
echo "=== B1 (v3 library)"; $PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir $LIB --cache results/wm/cache/selection_eval_cands_v3.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo "=== c08 zero-shot"; $PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
  --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
  --cache results/wm/cache/selection_eval_c08.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo "=== c10 same body"; $PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
  --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
  --cache results/wm/cache/anova_hex_beh24val.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo "=== shared latent"; $PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" "${M[@]}" --out results/deck/shared_latent_jointD3 2>&1 | grep -v -i warn | tail -4
echo EVAL_JOINT_V3_DONE
