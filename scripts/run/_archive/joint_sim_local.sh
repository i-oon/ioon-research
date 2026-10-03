#!/usr/bin/env bash
# Joint hexapod + B1 pretraining with the Froude-similarity loss (plan 2026-09-29).
# Same settings as FS (dt0_beh24: beh24, stride 5, hinge / readout / rollout, Froude head shapes z)
# plus both bodies (48 + 48 clean-train clips) and --lambda_sim 0.05. The matched control
# (identical, --lambda_sim 0) is not run yet; the earlier joint baseline (beh12_ego, stride 1) differs
# in data, stride, losses and split.
# Then: shared-latent numbers incl. cross-body retrieval, the standard evaluation (B1 pipeline, c08
# zero-shot, read-out checks), and the c10 same-body test.
#   nohup bash scripts/run/joint_sim_local.sh > results/wm/logs/joint_sim_local.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
N=joint_sim_beh24_s0
echo "=== train $N $(date)"
$PY -m wm.train \
  --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain b1=data/egocentric/beh24_b1_ego_flat_cleantrain \
  --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval b1=data/egocentric/beh24_b1_ego_flat_cleanval \
  --lambda_body 0.5 --detach_body_z False --lambda_hinge 0.5 --lambda_readout 1.0 --lambda_rollout 1.0 \
  --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 50 --checkpoint_every 2 \
  --resume auto --lambda_sim 0.05 --seed 0 --name $N >> results/wm/logs/$N.log 2>&1 \
  || { echo "TRAIN FAILED"; exit 1; }
echo "=== shared latent $(date)"
$PY scripts/figures/shared_latent_figure.py --device cuda \
  --model "joint_sim=wm/runs/$N/best.pt" --model "joint_beh12_ego=wm/runs/beh12_ego/best.pt" \
  --model "single_FS=wm/runs/dt0_beh24_s0/best.pt" --out results/deck/shared_latent_joint_sim 2>&1 | grep -v -i warn | tail -5
echo "=== standard evaluation $(date)"
bash scripts/run/detach_eval_local.sh $N
echo "=== c10 same body $(date)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
  --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
  --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
  --cache results/wm/cache/anova_hex_beh24val.pt \
  --ckpt "JS_c10=wm/runs/$N/c08_zeroshot/ckpt_lib_zeroshot.pt" --windows 0 11 | grep -E "w=|bounds"
echo JOINT_SIM_DONE
