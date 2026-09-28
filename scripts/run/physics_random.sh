#!/usr/bin/env bash
# Physics reference: a random library candidate at every decision (model-independent), c08 plain /
# phase-matched and B1. Runs after physics_loops.sh.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
until grep -q PHYSICS_LOOPS_DONE results/wm/logs/physics_loops.log 2>/dev/null; do sleep 300; done
OUT=results/wm/closed_loop/physics
R=wm/runs/beh24_stride5_cleansplit
declare -A G=([turn_s0.29]=hexapod_ep1202 [turn_s0.56]=hexapod_ep1303 [side_L_lvl0]=hexapod_ep2000
              [side_R_lvl1]=hexapod_ep2301 [speed_c7.1]=hexapod_ep103 [speed_c8.8]=hexapod_ep302)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  goal=data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/${G[$c]}.npz
  for ph in "" "--phase_match"; do
    r=$($PY sim/control/close_loop_hexapod_froude.py --mechanism random $ph --ckpt $R/c08_zeroshot/ckpt_lib_zeroshot.pt \
        --goal $goal --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat --morph c08f09t09=medauroidea_c08f09t09.ttt \
        --out $OUT/random/c08 2>&1 | grep -E "closed loop|Error" | tail -1)
    echo "c08 random random ${ph:-plain} $c | $r" | tee -a $OUT/summary.txt
  done
  r=$($PY sim/control/close_loop_b1_physics_froude.py --mechanism random --ckpt $R/b1_lora_c3/ckpt_lib_s4.pt \
      --goal $goal --out $OUT/random/b1 2>&1 | grep -E "closed loop|Error" | tail -1)
  echo "b1 random random policy $c | $r" | tee -a $OUT/summary.txt
done
echo PHYSICS_RANDOM_DONE
