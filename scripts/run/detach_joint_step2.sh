#!/usr/bin/env bash
# Step 2 of the detach test (plan 2026-09-28): with hexapod AND B1 pretrained together, does letting
# the shared body head's Froude gradient reach z (detach off) make z genuinely shared across bodies?
# Strict test: a read-out fit on one body and evaluated on the other (M10), plus k-NN mixing (M11),
# linear-probe body ID (M9), and the eta^2 grid (M5) for the action-sensitivity side of the trade-off.
#
#   DATA=beh24  hexapod beh24 cleantrain + B1 beh24 cleantrain (both exist)
#   DATA=babble hexapod switch babble + B1 switch babble (collect first, needs CoppeliaSim:
#               .venv/bin/python3 scripts/dataset/collect_babble_b1.py --clips 48 --seed 7 \
#                   --out data/egocentric/babble_b1_switch ; flat dir of its npz as babble_b1_switch_flat)
# Choose DATA from step 1's result. 2 seeds x detach on/off = 4 runs, ~11 h each (two bodies' data).
#   DATA=beh24 nohup bash scripts/run/detach_joint_step2.sh > results/wm/logs/detach_joint_step2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:?set DATA=beh24 or DATA=babble}
if [ "$DATA" = beh24 ]; then
  SRC="hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain b1=data/egocentric/beh24_b1_ego_flat_cleantrain"
else
  SRC="hexapod=data/egocentric/babble_c10f10t10_switch_flat b1=data/egocentric/babble_b1_switch_flat"
fi
for s in $SRC data/egocentric/beh24_c10f10t10_ego_flat_cleanval; do
  p=${s#*=}; [ -e "$p" ] || { echo "MISSING $p"; exit 1; }; done
PY=.venv/bin/python3
COMMON="--val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5
 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5
 --balance_embodiments True --epochs 50 --checkpoint_every 2 --resume auto"
M=()
for det in True False; do for seed in 0 1; do
  N=joint_${DATA}_dt$([ $det = True ] && echo 1 || echo 0)_s$seed
  echo "=== $N $(date)"
  $PY -m wm.train $COMMON --sources $SRC --detach_body_z $det --seed $seed --name $N \
      >> results/wm/logs/$N.log 2>&1 || { echo "TRAIN FAILED $N"; continue; }
  $PY scripts/diagnostics/objective_experiments/rollout_state_action_anova.py --z_source true \
      --embodiment hexapod --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
      --cache results/wm/cache/anova_hex_beh24val.pt --ckpt "$N=wm/runs/$N/best.pt" | tail -6
  M+=(--model "$N=wm/runs/$N/best.pt")
done; done
$PY scripts/figures/shared_latent_figure.py "${M[@]}" --out results/deck/shared_latent_joint_$DATA
echo DETACH_STEP2_DONE
