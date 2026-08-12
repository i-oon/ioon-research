#!/usr/bin/env bash
set -euo pipefail

cd /home/aria/ioon-research

RUN_NAME="${1:-stage2_clean_adv}"
LOG="results/wm/${RUN_NAME}.log"

mkdir -p "wm/runs/${RUN_NAME}/summary" results/wm

.venv/bin/python3 -m wm.train \
  --name "${RUN_NAME}" \
  --sources hexapod=data/ik_walk_8body b1=data/b1_framed \
  --heldout_bodies c06f10t06 c08f09t09 c10f10t06 \
  --clips_per_body hexapod=5 \
  --action_lag 1 \
  --lambda_adv 0.1 \
  --lambda_cross 0.0 \
  --cross_augment true \
  --within_body_std true \
  --md_head mlp \
  --balance_embodiments true \
  --ftm_embodiment_channel false \
  --center_embeddings false \
  --epochs 60 \
  --batch_size 8 \
  --lr 0.0001 \
  --seed 0 \
  --checkpoint_every 2 \
  2>&1 | tee "${LOG}"

