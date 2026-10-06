#!/usr/bin/env bash
# Control for the babbling sweep: same base (hexapod-only, random rooms), same evaluation, but B1 adapted on 44 TUNED
# clips (rr_b1_clips_train, the old recipe with --stratify) instead of babbling. Separates "babbling data" from
# "random rooms / new base model" as the cause of the weak rollout.
cd "$(dirname "$0")/../.."
BASE=${1:-wm/runs/round1_hexonly_s0_rr/best.pt}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=results/eval/b1_tuned_control; mkdir -p $OUT/ckpt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
{
echo "=== tuned-clip control  base=$BASE  $(date)"
$PY -m wm.adapt --ckpt $BASE --data $CW/rr_b1_clips_train --embodiment b1 --clips 44 --test_clips 4 --stratify --lambda_hinge 0.5 \
    --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0 --out $OUT/ckpt/adapted.pt 2>&1 | quiet | tail -1
$PY -m wm.fit_projector --ckpt $OUT/ckpt/adapted.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $CW/rr_b1_clips_train \
    --cache results/wm/cache/fitproj_rr.pt --out $OUT/ckpt/projector.pt 2>&1 | quiet | tail -1
$PY - <<'PYEOF'
import torch
from wm.models.action_projector import action_dims_from
o = "results/eval/b1_tuned_control/ckpt/"
ck = torch.load(o + "adapted.pt", map_location="cpu", weights_only=False)
sv = torch.load(o + "projector.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, o + "b1.pt")
PYEOF
echo "--- selection B1, realistic library (babbling)"
$PY $SEL --conditions all --candidates_dir $CW/b1_babble_lib24 --goal_dir $CW/rr_c10_clips_heldout \
    --cache results/wm/cache/sel_b1_babble_lib24.pt --windows 21 --ckpt tuned44=$OUT/ckpt/b1.pt 2>&1 | grep -E "w=|bounds"
echo "--- selection B1, upper-bound library (tuned clips)"
$PY $SEL --conditions all --candidates_dir $CW/rr_b1_clips_heldout --goal_dir $CW/rr_c10_clips_heldout \
    --cache results/wm/cache/test_v4_b1_heldout.pt --windows 21 --ckpt tuned44=$OUT/ckpt/b1.pt 2>&1 | grep -E "w=|bounds"
echo "--- read-out B1 (rr branches)"
$PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt tuned44=$OUT/ckpt/b1.pt --pairs 11 2>&1 | quiet | grep -A3 "^==="
rm -f $OUT/ckpt/adapted.pt
} 2>&1 | tee $OUT/summary.txt
