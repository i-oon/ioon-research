#!/usr/bin/env bash
# Baseline: the standard suite on the current models (before retraining), under the F295 conventions.
cd "$(dirname "$0")/../.."
bash scripts/run/eval_suite.sh fmd_s0 hexonly wm/runs/fmd_beh24_s0/best.pt
bash scripts/run/eval_suite.sh fmd_s1 hexonly wm/runs/fmd_beh24_s1/best.pt
bash scripts/run/eval_suite.sh jointD3_nosim joint wm/runs/jointD3_nosim_s0/best.pt
bash scripts/run/eval_suite.sh jointD3_sim joint wm/runs/jointD3_sim_s0/best.pt
echo EVAL_SUITE_CURRENT_DONE
