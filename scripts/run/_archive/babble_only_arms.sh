#!/usr/bin/env bash
# Does random/varied action data teach dynamics? (plan 2026-09-27). Stride-5 pretrains on 48 c10 clips
# each, differing only in how actions vary; 2 seeds per arm:
#   1 defined   beh24 cleantrain, one behaviour per clip   (exists: beh24_stride5_cleansplit, _A2)
#   2 switch    c10 babble, new drive every 15-25 frames   (babsw_stride5_s0/_s1)
#   3 rapid     c10 babble, new drive every 5 frames        (babrp_stride5_s0/_s1)
# Each goes through switch_pretrain_arms.sh's evaluation (B1 pipeline + state/action shares, c08
# zero-shot, counterfactual read-out, 5-step prediction), then the physics closed loops.
# Needs CoppeliaSim on port 23000. One GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
CEN=data/egocentric/beh24_c10f10t10_ego_flat/hexapod_ep10004.npz
RP=data/egocentric/babble_c10f10t10_rapid
if [ ! -d ${RP}_flat ]; then
  $PY scripts/dataset/collect_babble_hex.py --clips 48 --seed 5 --seg 5 5 --ramp 2 \
      --morph c10f10t10=medauroidea_c10f10t10.ttt --centre_from $CEN --out $RP || exit 1
  mkdir -p ${RP}_flat
  for d in $RP/clip*; do n=$(basename $d); ln -sf ../babble_c10f10t10_rapid/$n/c10f10t10_babble_ep0.npz ${RP}_flat/hexapod_babble_$n.npz; done
  $PY scripts/figures/render_babble_review.py --babble $RP --out results/deck/babble_review_c10_rapid
fi
echo RAPID_DATA_READY
COMMON="--val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5
 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5
 --epochs 50 --checkpoint_every 2 --resume auto"
eval "$(sed -n '/^arm() {/,/^}/p' scripts/run/switch_pretrain_arms.sh)"
SW=hexapod=data/egocentric/babble_c10f10t10_switch_flat
RR=hexapod=${RP}_flat
arm babsw_stride5_s0 --sources $SW
arm babrp_stride5_s0 --sources $RR
arm babsw_stride5_s1 --sources $SW --seed 1
arm babrp_stride5_s1 --sources $RR --seed 1
echo BABBLE_ARMS_TRAINED
OUT=results/wm/closed_loop/physics
declare -A G=([turn_s0.29]=hexapod_ep1202 [turn_s0.56]=hexapod_ep1303 [side_L_lvl0]=hexapod_ep2000
              [side_R_lvl1]=hexapod_ep2301 [speed_c7.1]=hexapod_ep103 [speed_c8.8]=hexapod_ep302)
for M in SW0:babsw_stride5_s0 RP0:babrp_stride5_s0 SW1:babsw_stride5_s1 RP1:babrp_stride5_s1; do
  tag=${M%%:*}; R=wm/runs/${M#*:}
  [ -f $R/b1_lora_c3/ckpt_lib_s4.pt ] || { echo "SKIP $tag (no checkpoint)"; continue; }
  for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
    goal=data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/${G[$c]}.npz
    for mech in direct rollout; do
      for ph in "" "--phase_match"; do
        r=$($PY sim/control/close_loop_hexapod_froude.py --mechanism $mech --window 21 $ph \
            --ckpt $R/c08_zeroshot/ckpt_lib_zeroshot.pt --goal $goal \
            --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat --morph c08f09t09=medauroidea_c08f09t09.ttt \
            --out $OUT/$tag/c08 2>&1 | grep -E "closed loop|Error|FELL" | tail -1)
        echo "c08 $tag $mech ${ph:-plain} $c | $r" | tee -a $OUT/summary.txt
      done
      r=$($PY sim/control/close_loop_b1_physics_froude.py --mechanism $mech --window 21 \
          --ckpt $R/b1_lora_c3/ckpt_lib_s4.pt --goal $goal --out $OUT/$tag/b1 2>&1 \
          | grep -E "closed loop|Error" | tail -1)
      echo "b1 $tag $mech policy $c | $r" | tee -a $OUT/summary.txt
    done
  done
done
echo BABBLE_ARMS_DONE
