#!/usr/bin/env bash
# Test grid (doc/STATUS.md "Adaptation direction"): STRUCTURED babbling (b1_sbabble_n$N: base behaviour -> switch at a fixed
# gait phase -> one-family random command -> back; near-same states, many futures) vs the random babbling results already in
# results/eval/b1_babble_sweep + b1_babble_variants. Per N: LoRA on ITM + FTM vs FTM frozen; plain z-fit projector; same
# evaluation (selection with the realistic and upper-bound libraries, B1 read-out on rr branches).
#   bash scripts/run/b1_sbabble_grid.sh wm/runs/round1_hexonly_s0_rr/best.pt [N ...]
cd "$(dirname "$0")/../.."
BASE=${1:?base checkpoint}; shift; NS=${*:-44 88}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=results/eval/b1_sbabble_grid; mkdir -p $OUT/ckpt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
ADAPT="--embodiment b1 --test_clips 4 --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0"
for N in $NS; do
  D=$CW/b1_sbabble_n$N
  {
  echo "=== structured babbling N=$N  base=$BASE  $(date)"
  for v in "full" "frz --freeze_ftm"; do
    set -- $v; tag=$1; shift; extra="$*"
    $PY -m wm.adapt --ckpt $BASE --data $D --clips $N $ADAPT $extra --out $OUT/ckpt/ad_${tag}_n$N.pt 2>&1 | quiet | tail -1
    $PY -m wm.fit_projector --ckpt $OUT/ckpt/ad_${tag}_n$N.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $D \
        --cache_dir results/wm/cache/fitproj_files --out $OUT/ckpt/proj_${tag}_n$N.pt 2>&1 | quiet | tail -1
    $PY - $OUT/ckpt/ad_${tag}_n$N.pt $OUT/ckpt/proj_${tag}_n$N.pt $OUT/ckpt/b1_${tag}_n$N.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
    rm -f $OUT/ckpt/ad_${tag}_n$N.pt
    echo "##### FTM $tag"
    echo "--- selection B1, realistic library (babbling)"
    $PY $SEL --conditions all --candidates_dir $CW/b1_babble_lib24 --goal_dir $CW/rr_c10_clips_heldout \
        --cache results/wm/cache/sel_b1_babble_lib24.pt --windows 21 --ckpt sb_${tag}_n$N=$OUT/ckpt/b1_${tag}_n$N.pt 2>&1 | grep -E "w=|bounds"
    echo "--- selection B1, upper-bound library (tuned clips)"
    $PY $SEL --conditions all --candidates_dir $CW/rr_b1_clips_heldout --goal_dir $CW/rr_c10_clips_heldout \
        --cache results/wm/cache/test_v4_b1_heldout.pt --windows 21 --ckpt sb_${tag}_n$N=$OUT/ckpt/b1_${tag}_n$N.pt 2>&1 | grep -E "w=|bounds"
    echo "--- read-out B1 (rr branches)"
    $PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt sb_${tag}_n$N=$OUT/ckpt/b1_${tag}_n$N.pt --pairs 11 2>&1 | quiet | grep -A3 "^==="
  done
  } 2>&1 | tee $OUT/n$N.txt
done
echo GRID_DONE
