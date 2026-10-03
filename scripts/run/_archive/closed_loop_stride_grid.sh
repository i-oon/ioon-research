#!/usr/bin/env bash
# Rendered closed loop (90-deg ego camera -> V-JEPA2 -> planner every 2 steps, kinematic posing,
# --level_body default) for stride 1 vs stride 5 (F261), out-of-sample Stage 2/4 fits. Same six goals
# and grading as selection_eval.py. Needs CoppeliaSim on port 23000.
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/wm/closed_loop/stride_live}
declare -A CK=([stride1]=wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_s4.pt
               [stride5]=wm/runs/beh24_stride5_cleansplit/b1_lora_c3/ckpt_lib_s4.pt)
declare -A G=([turn_s0.29]=hexapod_ep1202 [turn_s0.56]=hexapod_ep1303 [side_L_lvl0]=hexapod_ep2000
              [side_R_lvl1]=hexapod_ep2301 [speed_c7.1]=hexapod_ep103 [speed_c8.8]=hexapod_ep302)
mkdir -p $OUT
run() { local arm=$1 w=$2 MECH=$3
  for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
    r=$(.venv/bin/python3 sim/control/close_loop_direct_froude.py --mechanism $MECH \
      --goal_source physics --goal_timevarying --window $w --ckpt ${CK[$arm]} \
      --demo data/egocentric/beh12_b1_ego_flat_cleantrain/b1_ep2001.npz \
      --goal data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/${G[$c]}.npz \
      --goal_embodiment hexapod --candidates_dir data/egocentric/beh12_b1_ego_flat_cleantrain \
      --horizon 2 --replan_every 2 --warm_start 0 --travel 0 --steps 66 \
      --out $OUT/${MECH}_${arm}_w$w 2>&1 | grep -E "time-varying goal|view check|Error")
    echo "$MECH $arm w=$w $c $r" | tee -a $OUT/grid.txt
  done
}
for arm in stride1 stride5; do run $arm 0 rollout; run $arm 11 rollout; run $arm 11 direct; done
echo GRID_DONE
