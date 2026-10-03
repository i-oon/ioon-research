#!/usr/bin/env bash
# Step 4 of eval_suite.sh re-run after the streamed read-out fix (the first attempt ran out of RAM, 2026-10-03 22:56).
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python3; NAME=round1_branches_s0; OUT=results/eval/$NAME; CK=$OUT/ckpt; CW=data/counterfactual_walks
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
{
echo "--- counterfactual read-out hexapod (physics branches, heldout)"
$PY $RO --embodiment hexapod --cf_dir $CW/c10_branches_heldout --ckpt $NAME=$CK/hex.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
echo "--- counterfactual read-out B1 (physics branches, heldout)"
$PY $RO --embodiment b1 --cf_dir $CW/b1_branches_heldout --ckpt $NAME=$CK/b1.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
echo "--- counterfactual read-out c08 zero-shot (physics branches, held-out body, hexapod projector)"
$PY $RO --embodiment hexapod --cf_dir $CW/c08_branches_heldout --ckpt $NAME=$CK/hex.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
} 2>&1 | tee $OUT/readout.txt | tee -a $OUT/summary.txt
echo "EVAL_SUITE_DONE $NAME" | tee -a $OUT/summary.txt
