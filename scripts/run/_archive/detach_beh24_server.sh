#!/usr/bin/env bash
# Server job: step 1's beh24 x detach-off cell, TRAINING ONLY (2 seeds). Evaluation runs on the
# workstation afterwards (its caches are keyed by the workstation's absolute paths), with
#   bash scripts/run/detach_eval_local.sh dt0_beh24_s0 dt0_beh24_s1
# Identical to A / A2 (beh24_stride5_cleansplit / _A2) except --detach_body_z False:
# stride 5 (frame_stride 5; action_chunk 0 = follows the stride -> 5-command chunks), action_lag 1,
# same losses, 50 epochs.
# Needs: data/egocentric/beh24_c10f10t10_ego_flat_cleantrain, ..._cleanval (copy with rsync -L:
# they are symlinks into beh24_c10f10t10_ego_flat), the V-JEPA2 weights in the HF cache, .venv.
#   nohup bash scripts/run/detach_beh24_server.sh > results/wm/logs/detach_beh24_server.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
for p in data/egocentric/beh24_c10f10t10_ego_flat_cleantrain data/egocentric/beh24_c10f10t10_ego_flat_cleanval; do
  [ "$(ls $p/*.npz 2>/dev/null | head -1)" ] && [ -s "$(ls $p/*.npz | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
mkdir -p results/wm/logs
PY=.venv/bin/python3
COMMON="--val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5
 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5
 --epochs 50 --checkpoint_every 2 --resume auto"
for seed in 0 1; do
  N=dt0_beh24_s$seed
  echo "=== $N $(date)"
  $PY -m wm.train $COMMON --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
      --detach_body_z False --seed $seed --name $N >> results/wm/logs/$N.log 2>&1 || echo "TRAIN FAILED $N"
  grep -q "detach_body_z: false" wm/runs/$N/config.yaml && echo "  config ok: detach_body_z false" \
      || echo "  WARNING: config does not record detach_body_z false"
done
echo DETACH_BEH24_SERVER_DONE
