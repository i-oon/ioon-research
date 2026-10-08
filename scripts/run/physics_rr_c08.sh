#!/usr/bin/env bash
# c08 (held-out hexapod, never trained) physics closed loop, physics_rr_b1.sh's twin: joint c10+B1 branches
# pretraining with random rooms + hexapod projector (round1_branches_s0_rr/ckpt/hex.pt), rr held-out c08 candidates,
# rr held-out c10 goals; live ego view in the goal's own random room (paired rr c08 clip hexapod_ep6xxxx, same room
# seed / size / offset). Needs one own CoppeliaSim on $PORT (never 23000).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25600}
OUT=results/wm/closed_loop_rr/physics/joint_rr
CK=results/eval/round1_branches_s0_rr/ckpt/hex.pt
CAND=data/counterfactual_walks/rr_c08_clips_heldout
declare -A G=([turn_s0.29]=40101 [turn_s0.56]=40110 [side_L_lvl0]=40163 [side_R_lvl1]=40212 [speed_c7.1]=40012 [speed_c8.8]=40030)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  e=${G[$c]}; goal=data/counterfactual_walks/rr_c10_clips_heldout/hexapod_ep$e.npz
  room=$CAND/hexapod_ep$((e + 20000)).npz
  for mech in direct rollout random; do
    o=$OUT/c08; [ $mech = random ] && o=$OUT/random/c08
    r=$(nice -n 10 $PY sim/control/close_loop_hexapod_froude.py --mechanism $mech --window 21 --ckpt $CK \
        --goal $goal --candidates_dir $CAND --morph c08f09t09=medauroidea_c08f09t09.ttt --rr_room $room \
        --port $PORT --out $o 2>&1 | tee -a $OUT/loop_c08.log | grep -E "closed loop|Error|FELL|MAE|corr|rr room" | tr '\n' ' ')
    echo "c08 joint_rr $mech $c | $r" | tee -a $OUT/summary.txt
  done
done
echo PHYSICS_RR_C08_DONE
