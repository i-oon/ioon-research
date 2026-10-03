#!/usr/bin/env bash
# Physics closed loops for the detach-off models (step 1), identical to babble_only_arms.sh's loop.
# Needs CoppeliaSim on port 23000 and the trained runs under wm/runs/ (copy back from the server:
#   rsync -a ioon@SERVER:ioon-research/wm/runs/dt0_* wm/runs/ ).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
OUT=results/wm/closed_loop/physics
declare -A G=([turn_s0.29]=hexapod_ep1202 [turn_s0.56]=hexapod_ep1303 [side_L_lvl0]=hexapod_ep2000
              [side_R_lvl1]=hexapod_ep2301 [speed_c7.1]=hexapod_ep103 [speed_c8.8]=hexapod_ep302)
for M in FM0:fmd_beh24_s0 FM1:fmd_beh24_s1; do
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
echo FMD_PHYSICS_DONE
