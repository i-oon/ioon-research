#!/usr/bin/env bash
# c10 (pretraining body, reference) physics closed loop, physics_rr_c08.sh's twin: same joint c10+B1 branches pretraining
# with random rooms + hexapod projector, same 6 rr held-out c10 goals, same window / decision cadence. Candidates = the 48
# rr c10 TRAINING clips (the held-out goals are never candidates); live ego view in the goal clip's own random room.
# Needs one own CoppeliaSim on $PORT (never 23000).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25600}
OUT=results/wm/closed_loop_rr/physics/joint_rr
CK=results/eval/round1_branches_s0_rr/ckpt/hex.pt
CAND=data/counterfactual_walks/rr_c10_clips_train
declare -A G=([turn_s0.29]=40101 [turn_s0.56]=40110 [side_L_lvl0]=40163 [side_R_lvl1]=40212 [speed_c7.1]=40012 [speed_c8.8]=40030)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  e=${G[$c]}; goal=data/counterfactual_walks/rr_c10_clips_heldout/hexapod_ep$e.npz
  for mech in direct rollout random; do
    o=$OUT/c10; [ $mech = random ] && o=$OUT/random/c10
    r=$(nice -n 10 $PY sim/control/close_loop_hexapod_froude.py --mechanism $mech --window 21 --ckpt $CK \
        --goal $goal --candidates_dir $CAND --morph c10f10t10=medauroidea_c10f10t10.ttt --rr_room $goal \
        --port $PORT --out $o 2>&1 | tee -a $OUT/loop_c10.log | grep -E "closed loop|Error|FELL|MAE|corr|rr room" | tr '\n' ' ')
    echo "c10 joint_rr $mech $c | $r" | tee -a $OUT/summary_c10.txt
  done
done
echo PHYSICS_RR_C10_DONE
