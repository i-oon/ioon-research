#!/usr/bin/env bash
# This week's claim test: B1, absent from pretraining, adapted from its own babbling only (no task knowledge), with an
# adaptation-data sweep. Base model = hexapod-only pretraining on random-room data. Everything evaluated on random-room
# held-out data. Per budget N (babbling windows; nested subsets b1_babble_n{N}, +4 windows adapt.py keeps for its check):
#   LoRA r8 3000 steps with the Froude loss (eval_suite's hexonly recipe, no --stratify: babble has no behaviours)
#   -> B1 projector on the same windows -> selection with the REALISTIC library (24 babbling windows, k-means on B1's own
#   motion, b1_babble_lib24) and the UPPER-BOUND library (tuned B1 clips, rr_b1_clips_heldout); goals = rr c10 held-out
#   -> B1 counterfactual read-out on rr_b1_branches_heldout.
#   bash scripts/run/b1_babble_sweep.sh wm/runs/round1_hexonly_s0_rr/best.pt [N ...]
cd "$(dirname "$0")/../.."
BASE=${1:?base checkpoint}; shift; NS=${*:-11 22 44 88 176}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=results/eval/b1_babble_sweep; mkdir -p $OUT/ckpt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
$PY -m wm.fit_projector --ckpt $BASE --hex_dir $CW/rr_c10_clips_train --b1_dir "" --cache_dir results/wm/cache/fitproj_files \
    --out $OUT/ckpt/projector_hex.pt 2>&1 | quiet | tail -1
for N in $NS; do
  T=$OUT/n$N; mkdir -p $T
  {
  echo "=== N=$N babbling windows  base=$BASE  $(date)"
  $PY -m wm.adapt --ckpt $BASE --data $CW/b1_babble_n$N --embodiment b1 --clips $N --test_clips 4 --lambda_hinge 0.5 \
      --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0 --out $OUT/ckpt/adapted_n$N.pt 2>&1 | quiet | tail -2
  $PY -m wm.fit_projector --ckpt $OUT/ckpt/adapted_n$N.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $CW/b1_babble_n$N \
      --cache_dir results/wm/cache/fitproj_files --out $OUT/ckpt/projector_n$N.pt 2>&1 | quiet | tail -1
  $PY - "$OUT/ckpt/adapted_n$N.pt" "$OUT/ckpt/projector_n$N.pt" "$OUT/ckpt/b1_n$N.pt" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
  echo "--- selection B1, realistic library (babbling)"
  $PY $SEL --conditions all --candidates_dir $CW/b1_babble_lib24 --goal_dir $CW/rr_c10_clips_heldout \
      --cache results/wm/cache/sel_b1_babble_lib24.pt --windows 21 --ckpt n$N=$OUT/ckpt/b1_n$N.pt 2>&1 | grep -E "w=|bounds"
  echo "--- selection B1, upper-bound library (tuned clips)"
  $PY $SEL --conditions all --candidates_dir $CW/rr_b1_clips_heldout --goal_dir $CW/rr_c10_clips_heldout \
      --cache results/wm/cache/test_v4_b1_heldout.pt --windows 21 --ckpt n$N=$OUT/ckpt/b1_n$N.pt 2>&1 | grep -E "w=|bounds"
  echo "--- read-out B1 (rr branches)"
  $PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt n$N=$OUT/ckpt/b1_n$N.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
  } 2>&1 | tee $T/summary.txt
  rm -f $OUT/ckpt/adapted_n$N.pt          # 400 MB each; b1_n$N.pt keeps the adapted weights + projector
done
echo SWEEP_DONE
