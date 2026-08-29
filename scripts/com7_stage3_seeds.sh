#!/usr/bin/env bash
# Three seeds per arm of the stage-3 ablation, on the corrected B1 set.
#
# **Why seeds and not another budget.** The two arms have been trained once each at two budgets and
# the ordering flipped between them: MSE's forward speed error went 9% at 12k steps to 58% at 15k,
# and its turning sign went 4/4 correct to 0/4 (F116). A single run per cell cannot tell "the
# contrastive term helps" from "training is unstable", and every earlier number in this project's
# turning story turned out to be an artefact of something unmeasured.
#
# `--seed` controls batch order, negative sampling and init. Before it existed every run of one
# configuration was bit-identical, so repeats carried no information.
#
#   bash scripts/com7_stage3_seeds.sh            # from the repository root, on com7
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
# the 24-clip budget the original stage 3 used, listed here so this sheet needs no checkpoint
# to read it from -- it is 24 filenames, not a reason to copy 384 MB between machines
CLIPS="
  b1_ep1.npz b1_ep1001.npz b1_ep1002.npz b1_ep101.npz
  b1_ep102.npz b1_ep1101.npz b1_ep1102.npz b1_ep1201.npz
  b1_ep1202.npz b1_ep1301.npz b1_ep1302.npz b1_ep2.npz
  b1_ep2001.npz b1_ep2002.npz b1_ep201.npz b1_ep202.npz
  b1_ep2101.npz b1_ep2102.npz b1_ep2201.npz b1_ep2202.npz
  b1_ep2301.npz b1_ep2302.npz b1_ep301.npz b1_ep302.npz
"

for SEED in 0 1 2; do
  for ARM in nce mse; do
    LAM=$([ "$ARM" = nce ] && echo 1.0 || echo 0.0)
    OUT=wm/runs/beh12_hexonly/stage3_b1_${ARM}_v2_s${SEED}.pt
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== $ARM seed $SEED ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.adapt3 \
      --ckpt wm/runs/beh12_hexonly/adapted_b1_v2.pt \
      --projector wm/runs/beh12_hexonly/projector_b1_adapted_v2.pt \
      --data data/beh12_b1_flat --embodiment b1 --train_clips $CLIPS \
      --steps 15000 --lambda_nce $LAM --batch 8 --seed $SEED \
      --cache results/wm/cache/b1_v2.pt --out "$OUT"
  done
done
echo "done -- six checkpoints; close the loop on each and report mean and spread, never one run"
