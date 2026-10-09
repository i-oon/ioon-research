#!/usr/bin/env bash
# B1 physics closed loop (fmd_physics.sh / physics_random.sh logic) with the current model and rr data:
# joint c10+B1 branches pretraining, random rooms (round1_branches_s0_rr), rr held-out B1 candidates, rr held-out c10
# goals; live ego view in the goal's own random room (paired rr B1 clip b1_ep5xxxx, same room seed / size / offset).
# Needs one own CoppeliaSim on $PORT (never 23000).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25500}
OUT=results/wm/closed_loop_rr/physics/joint_rr
CK=results/eval/round1_branches_s0_rr/ckpt/b1.pt
CAND=data/counterfactual_walks/rr_b1_clips_heldout
declare -A G=([turn_s0.29]=40101 [turn_s0.56]=40110 [side_L_lvl0]=40163 [side_R_lvl1]=40212 [speed_c7.1]=40012 [speed_c8.8]=40030)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  e=${G[$c]}; goal=data/counterfactual_walks/rr_c10_clips_heldout/hexapod_ep$e.npz
  room=$CAND/b1_ep$((e + 10000)).npz
  for mech in direct rollout random; do
    o=$OUT/b1; [ $mech = random ] && o=$OUT/random/b1
    r=$(nice -n 10 $PY sim/control/close_loop_b1_physics_froude.py --mechanism $mech --window 21 --ckpt $CK \
        --goal $goal --candidates_dir $CAND --rr_room $room --thirdperson --port $PORT --out $o 2>&1 \
        | tee -a $OUT/loop.log | grep -E "closed loop|Error|FELL|MAE|corr" | tr '\n' ' ')
    echo "b1 joint_rr $mech policy $c | $r" | tee -a $OUT/summary.txt
  done
done
echo PHYSICS_RR_DONE
