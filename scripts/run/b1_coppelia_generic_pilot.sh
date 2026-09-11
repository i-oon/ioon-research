#!/usr/bin/env bash
# Four-seed precommitted claim-honest pilot. Keep every rollout; do not branch on its outcome.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
COLLECTOR=sim/collect/collect_b1_coppelia_generic_trot_babble.py
OUT=results/wm/dataset/b1_babble/coppelia_quadruped_trot_pilot

mkdir -p "$OUT"
for seed in 0 1 2 3; do
  "$PY" "$COLLECTOR" \
    --steps 160 --warmup 25 --noise 0.05 \
    --generic-freq-min 1.0 --generic-freq-max 3.0 \
    --generic-amp-min 0.1 --generic-amp-max 0.8 \
    --seed "$seed" --ego --ego-seed "$seed" \
    --out "$OUT/generic_seed${seed}.npz" \
    --video "$OUT/generic_seed${seed}.mp4"
done
