#!/usr/bin/env bash
# Follow-up to b1_babble_sweep.sh (rollout weak at small N): per N, three adaptation variants, same evaluation.
#   A  LoRA on ITM + FTM (as the sweep)   + projector fitted through the rollout read (--objective rollout, F311)
#   B  LoRA on ITM only, FTM frozen      + plain z-fit projector
#   C  LoRA on ITM only, FTM frozen      + rollout-fitted projector
#   bash scripts/run/b1_babble_variants.sh wm/runs/round1_hexonly_s0_rr/best.pt [N ...]
cd "$(dirname "$0")/../.."
BASE=${1:?base checkpoint}; shift; NS=${*:-44 176}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=results/eval/b1_babble_variants; mkdir -p $OUT/ckpt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
merge() { $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
}
evalb1() {  # tag ckpt
  echo "--- selection B1, realistic library (babbling)"
  $PY $SEL --conditions all --candidates_dir $CW/b1_babble_lib24 --goal_dir $CW/rr_c10_clips_heldout \
      --cache results/wm/cache/sel_b1_babble_lib24.pt --windows 21 --ckpt $1=$2 2>&1 | grep -E "w=|bounds"
  echo "--- selection B1, upper-bound library (tuned clips)"
  $PY $SEL --conditions all --candidates_dir $CW/rr_b1_clips_heldout --goal_dir $CW/rr_c10_clips_heldout \
      --cache results/wm/cache/test_v4_b1_heldout.pt --windows 21 --ckpt $1=$2 2>&1 | grep -E "w=|bounds"
  echo "--- read-out B1 (rr branches)"
  $PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt $1=$2 --pairs 11 2>&1 | quiet | grep -A3 "^==="
}
ADAPT="--embodiment b1 --test_clips 4 --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0"
for N in $NS; do
  D=$CW/b1_babble_n$N
  {
  echo "=== N=$N  $(date)"
  $PY -m wm.adapt --ckpt $BASE --data $D --clips $N $ADAPT --out $OUT/ckpt/ad_full_n$N.pt 2>&1 | quiet | tail -1
  $PY -m wm.adapt --ckpt $BASE --data $D --clips $N $ADAPT --freeze_ftm --out $OUT/ckpt/ad_frz_n$N.pt 2>&1 | quiet | tail -1
  for v in "A full rollout" "B frz z" "C frz rollout"; do
    set -- $v; tag=$1; ad=$2; obj=$3
    $PY -m wm.fit_projector --ckpt $OUT/ckpt/ad_${ad}_n$N.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $D \
        --cache results/wm/cache/fitproj_rr.pt --objective $obj --out $OUT/ckpt/proj_${tag}_n$N.pt 2>&1 | quiet | tail -1
    merge $OUT/ckpt/ad_${ad}_n$N.pt $OUT/ckpt/proj_${tag}_n$N.pt $OUT/ckpt/b1_${tag}_n$N.pt
    echo "##### variant $tag (FTM ${ad}, projector ${obj})"
    evalb1 ${tag}_n$N $OUT/ckpt/b1_${tag}_n$N.pt
  done
  rm -f $OUT/ckpt/ad_full_n$N.pt $OUT/ckpt/ad_frz_n$N.pt
  } 2>&1 | tee $OUT/n$N.txt
done
echo VARIANTS_DONE
