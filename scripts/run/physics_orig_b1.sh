#!/usr/bin/env bash
# B1 physics closed loop in the ORIGINAL rooms (sized to the body, B1 17.65 m, the room seed of each clip):
# round1_branches_s0 (joint c10 + B1, with branches, rooms sized to the body) + its B1 projector, 24 tuned held-out B1
# candidates (b1_clips_heldout), goals = 6 held-out hexapod clips (c10_clips_heldout); live ego view in the goal's paired
# B1 clip's original room (--orig_room). Twin of physics_rr_b1.sh; only the model and rooms differ. Own CoppeliaSim on $PORT.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25500}
OUT=${OUT:-results/wm/closed_loop_orig/physics/joint_branches}
CK=${CK:-results/eval/round1_branches_s0/ckpt/b1.pt}
CAND=data/counterfactual_walks/b1_clips_heldout
declare -A G=([turn_s0.29]=40101 [turn_s0.56]=40110 [side_L_lvl0]=40163 [side_R_lvl1]=40212 [speed_c7.1]=40012 [speed_c8.8]=40030)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  e=${G[$c]}; goal=data/counterfactual_walks/c10_clips_heldout/hexapod_ep$e.npz
  room=$CAND/b1_ep$((e + 10000)).npz
  for mech in ${MECHS:-direct rollout}; do
    o=$OUT/b1; [ $mech = random ] && o=$OUT/random/b1
    r=$(nice -n 10 $PY sim/control/close_loop_b1_physics_froude.py --mechanism $mech --window 21 --ckpt $CK \
        --goal $goal --candidates_dir $CAND --rr_room $room --orig_room --thirdperson --port $PORT --device ${DEVICE:-cuda} --out $o 2>&1 \
        | tee -a $OUT/loop.log | grep -E "closed loop|Error|FELL|MAE|corr" | tr '\n' ' ')
    echo "b1 joint_branches_orig $mech policy $c | $r" | tee -a $OUT/summary.txt
  done
done
echo PHYSICS_ORIG_B1_DONE
