#!/usr/bin/env bash
# Server job (plan 2026-09-29): "motion decoder -> Froude instead of joint actions".
# Identical to dt0_beh24 (beh24 cleantrain, stride 5 = 5-command chunks, same losses, 50 epochs,
# detach_body_z False) except --lambda_motion 0: the per-body joint-command motion decoder no longer
# trains z; the shared z-only Froude head (undetached) is the only motion target.
# Seeds 0 and 1 run in parallel, one per GPU. TRAINING ONLY: evaluate on the workstation with
#   bash scripts/run/detach_eval_local.sh fmd_beh24_s0 fmd_beh24_s1
# Files to bring back per run: wm/runs/<name>/best.pt, config.yaml (and results/wm/logs/<name>.log).
# Needs on the server: updated wm/ code (detach_body_z flag), data/egocentric/beh24_c10f10t10_ego_flat_clean{train,val}.
#   bash scripts/run/froude_md_server.sh        (starts both in the background and returns)
set -uo pipefail
cd "$(dirname "$0")/../.."
for p in data/egocentric/beh24_c10f10t10_ego_flat_cleantrain data/egocentric/beh24_c10f10t10_ego_flat_cleanval; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
grep -q detach_body_z wm/config.py || { echo "wm/config.py lacks detach_body_z: copy the updated wm/ code first"; exit 1; }
mkdir -p results/wm/logs
for s in 0 1; do
  N=fmd_beh24_s$s
  CUDA_VISIBLE_DEVICES=$s nohup .venv/bin/python3 -m wm.train \
    --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5 \
    --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 \
    --epochs 50 --checkpoint_every 2 --resume auto \
    --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
    --detach_body_z False --lambda_motion 0 --seed $s --name $N > results/wm/logs/$N.log 2>&1 &
  echo "started $N on GPU $s (pid $!)"
done
echo "check: nvidia-smi ; tail -3 results/wm/logs/fmd_beh24_s0.log"
echo "after ~1 min: grep -E 'detach_body_z|lambda_motion' wm/runs/fmd_beh24_s*/config.yaml   (false / 0.0)"
