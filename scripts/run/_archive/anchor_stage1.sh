#!/usr/bin/env bash
# Does anchoring Stage 1 put the B1's video latents into the hexapod's z? (plan 2026-09-29)
# Base: fmd_beh24_s0 (hexapod pretrain, Froude head shapes z, no joint-command decoder).
# Stage 1 as the standard pipeline (LoRA r2, 3 stratified B1 clips, hinge 0.5, 1000 steps), four arms:
#   none | froude-head anchor | similarity anchor | both
# Measured on the adapted ITM: body-ID probe, cross-body R2, k-NN mixing, cross-body retrieval
# (shared_latent_figure.py), plus Stage 1's own rollout ratio (prediction must not get worse).
# Then the queued D physics loop re-run.
#   nohup bash scripts/run/anchor_stage1.sh > results/wm/logs/anchor_stage1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
BASE=wm/runs/fmd_beh24_s0/best.pt
D=wm/runs/fmd_beh24_s0/anchor
mkdir -p $D
M=()
for arm in "none:0:0" "froude:1.0:0" "sim:0:1.0" "both:1.0:1.0"; do
  IFS=: read name wf ws <<< "$arm"
  echo "=== stage 1, anchor $name (froude $wf, sim $ws) $(date)"
  $PY -m wm.adapt --ckpt $BASE --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 \
      --clips 3 --stratify --lambda_hinge 0.5 --hinge_margin 0.1 \
      --anchor_froude $wf --anchor_sim $ws --out $D/adapted_$name.pt 2>&1 | grep -v -i warn | tail -9
  M+=(--model "anchor_$name=$D/adapted_$name.pt")
done
echo "=== shared latent $(date)"
$PY scripts/figures/shared_latent_figure.py --device cuda "${M[@]}" --out results/deck/shared_latent_anchor 2>&1 \
  | grep -v -i warn | tail -6
echo ANCHOR_STAGE1_DONE
bash scripts/run/fmd_physics.sh > results/wm/logs/fmd_physics_rerun2.log 2>&1
echo FMD_PHYSICS_RERUN_DONE
