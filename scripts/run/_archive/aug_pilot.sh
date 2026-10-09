#!/usr/bin/env bash
# NOTE 2026-10-03: data paths are older than data/counterfactual_walks; switch to data/counterfactual_walks (doc/DATA.md) before running.
# Augmentation pilot (2026-09-30): does Egocentric VSM-style view randomisation make the model robust to a
# rendering change without hurting clean performance? jointD3 similarity recipe (c10 + B1 v3, Froude only,
# lambda_sim 0.05, S0), shortened to 17 epochs (~1/3), three arms run ONE AT A TIME:
#   augnorm  current augmentation (crop 85-100%, brightness/contrast +-20%)           -- baseline at equal length
#   augsafe  crop 30-100%, brightness x1/3-3 (log), contrast +-40%, blur kernel <=15 (sigma 2.6),
#            saturation +-0.5, hue +-20 deg, noise <=3%
#   augvsm   crop 10-100%, brightness x0.1-10 (log), blur kernel <=41 (sigma 6.5), rest as augsafe
# In augsafe / augvsm every type is applied with probability 0.5 and 25% of samples stay fully clean.
# Same augmentation on both frames of a pair (wm/data/augment.py). No rotation or flip.
#   bash scripts/run/aug_pilot.sh            (all three)     bash scripts/run/aug_pilot.sh augsafe   (one)
set -uo pipefail
cd "$(dirname "$0")/../.."
grep -q aug_clean wm/config.py || { echo "wm/config.py lacks aug_clean"; exit 1; }
mkdir -p results/wm/logs
COMMON="--sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain b1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
 --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval b1=data/egocentric_v3/beh24_b1_ego_flat_cleanval
 --lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0
 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 17
 --checkpoint_every 2 --resume auto --lambda_sim 0.05 --seed 0"
STRONG="--aug_contrast 0.4 --aug_brightness 0 --aug_saturation 0.5 --aug_hue 20 --aug_noise 0.03 --aug_prob 0.5 --aug_clean 0.25"
declare -A ARM=(
  [augnorm]=""
  [augsafe]="$STRONG --aug_min_scale 0.3 --aug_bright_mult 3 --aug_blur 2.6"
  [augvsm]="$STRONG --aug_min_scale 0.1 --aug_bright_mult 10 --aug_blur 6.5"
)
for a in ${@:-augnorm augsafe augvsm}; do
  N=pilot_${a}_s0
  echo "=== $N $(date)"
  .venv/bin/python3 -m wm.train $COMMON ${ARM[$a]} --name $N > results/wm/logs/$N.log 2>&1
  tail -2 results/wm/logs/$N.log
done
echo AUG_PILOT_DONE
