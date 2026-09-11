#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
COLLECTOR=sim/collect/collect_b1_coppelia_stance_swing_babble.py
OUT=results/wm/dataset/b1_babble/coppelia_lift_frequency_grid
mkdir -p "$OUT"

for freq in 1.5 1.75 2.0; do
  for amp in 0.14 0.18 0.22; do
    for ratio in 2.0 2.5; do
      tag="f${freq}_a${amp}_c${ratio}"
      "$PY" "$COLLECTOR" --steps 160 --warmup 25 --noise 0.05 \
        --generic-frequency "$freq" --generic-amplitude "$amp" \
        --generic-calf-ratio "$ratio" --seed 0 --ego --ego-seed 0 \
        --out "$OUT/${tag}.npz" --video "$OUT/${tag}.mp4"
    done
  done
done
