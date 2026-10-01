#!/usr/bin/env bash
# Local (2026-09-30 evening): jointD3 similarity arm (B1 on v3) with strong training-view augmentation
# (crop 0.5-1, brightness/contrast +-0.4, blur <=2 px, saturation +-0.5, hue +-20 deg, noise <=3%).
# Pilot for render robustness: evaluate on the old B1 renders and on v3.











set -uo pipefail
cd "$(dirname "$0")/../.."
for p in data/egocentric/beh24_c10f10t10_ego_flat_cleantrain data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
         data/egocentric_v3/beh24_b1_ego_flat_cleantrain data/egocentric_v3/beh24_b1_ego_flat_cleanval; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
grep -q lambda_sim wm/config.py || { echo "wm/config.py lacks lambda_sim: copy the updated wm/ code first"; exit 1; }
mkdir -p results/wm/logs
i=0
for arm in "jointD3aug_sim_s0:0.05"; do
  N=${arm%%:*}; W=${arm#*:}
  .venv/bin/python3 -m wm.train \
    --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain b1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain \
    --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval b1=data/egocentric_v3/beh24_b1_ego_flat_cleanval \
    --lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0 \
    --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 50 \
    --checkpoint_every 2 --resume auto --lambda_sim $W --seed 0 --aug_min_scale 0.5 --aug_brightness 0.4 --aug_contrast 0.4 --aug_blur 2.0 --aug_saturation 0.5 --aug_hue 20 --aug_noise 0.03 --name $N > results/wm/logs/$N.log 2>&1
  echo "started $N (lambda_sim $W) on GPU $i (pid $!)"
  i=$((i + 1))
done
echo "after ~2 min: grep -E 'lambda_sim|lambda_motion|detach_body_z' wm/runs/jointD3_*/config.yaml"
echo "              grep '^epoch' results/wm/logs/jointD3_sim_s0.log | tail -1   (should show 'sim ... x-body pairs')"
