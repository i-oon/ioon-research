#!/usr/bin/env bash
# STRIDE-MATCHED FROUDE LABELS (2026-10-15; F348, Ajan Blink week 20). One variable vs the base model
# round1_branches_s0_gray_imread: the same arm B data, recipe and 30.5k steps, only the hexapod label window changes
# (whole strides per command segment instead of 1 s; B1 already spans 2 strides and is unchanged).
# Evaluation on BOTH label definitions, for both models, so the comparison is on identical targets:
#   standard 1 s labels   : eval_suite.sh NAME joint                 (base model already evaluated)
#   stride-matched labels : WM_STRIDE_LABELS=1 eval_suite.sh NAME_evalstride joint  (both models)
# Run on the GPU PC from the repo root:  nohup bash scripts/run/stride_labels_run.sh > results/wm/logs/stride_labels_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
grep -q "def stride_window" wm/data/embodiment.py || { echo "STOP: git pull first"; exit 1; }
[ -s wm/data/stride_periods_hexapod.json ] || { echo "STOP: wm/data/stride_periods_hexapod.json missing"; exit 1; }
RECIPE="--grayscale True --aug_brightness 0.0 --aug_contrast 0.4 --aug_blur 2.6 --aug_noise 0.03 --aug_prob 0.5
 --aug_bright_mult 3.0 --aug_clean 0.25 --lambda_imread 1.0 --imread_k 4"
N=round1_branches_s0_gray_imread_stride
RUN_TAG=_gray_imread_stride EXTRA="$RECIPE --stride_labels True" bash scripts/run/round1_counterfactual.sh B || exit 1
sleep 60
while ps -eo args | grep -v grep | grep -q -- "--name $N"; do sleep 300; done
[ -s wm/runs/$N/best.pt ] || { echo "STOP: no best.pt"; exit 1; }
bash scripts/run/eval_suite.sh $N joint wm/runs/$N/best.pt
WM_STRIDE_LABELS=1 bash scripts/run/eval_suite.sh ${N}_evalstride joint wm/runs/$N/best.pt
B=round1_branches_s0_gray_imread
[ -s wm/runs/$B/best.pt ] && WM_STRIDE_LABELS=1 bash scripts/run/eval_suite.sh ${B}_evalstride joint wm/runs/$B/best.pt
for n in $N ${N}_evalstride ${B}_evalstride; do echo "=== $n"; grep -E "^---|w=21|r fwd|real cf|FTM cf" results/eval/$n/summary.txt; done
echo STRIDE_LABELS_RUN_DONE
