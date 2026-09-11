#!/usr/bin/env bash
# Precommitted 2 lift ratios x 4 seeds; keep every outcome.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
COLLECTOR=sim/collect/collect_b1_coppelia_stance_swing_babble.py
OUT=results/wm/dataset/b1_babble/coppelia_stance_swing_pilot
mkdir -p "$OUT"

for ratio in 1.5 2.0; do
  for seed in 0 1 2 3; do
    tag="calf${ratio}_seed${seed}"
    "$PY" "$COLLECTOR" --steps 160 --warmup 25 --noise 0.05 \
      --generic-freq-min 1.0 --generic-freq-max 3.0 \
      --generic-amp-min 0.1 --generic-amp-max 0.8 \
      --generic-calf-ratio "$ratio" --seed "$seed" --ego --ego-seed "$seed" \
      --out "$OUT/${tag}.npz" --video "$OUT/${tag}.mp4"
  done
done
