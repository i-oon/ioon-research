#!/usr/bin/env bash
# Hexapod physics closed loop in the ORIGINAL rooms (sized to the body, 8 m, each clip's own room seed), with the F323
# fix (--cpg_clock: one continuous CPG gait clock, 4-frame recipe cross-fade at a switch, as the branches).
# Model round1_branches_s0 (joint c10 + B1, with branches, rooms sized to the body) + its hexapod projector.
#   BODY=c10: candidates = the 48 c10 TRAINING clips (c10_clips_train); live room = the goal clip's own room
#   BODY=c08: candidates = the 24 c08 held-out clips (never trained, zero-shot); live room = the paired c08 clip's room
# Goals = the same 6 held-out hexapod clips (c10_clips_heldout) as physics_rr_*.sh / physics_orig_b1.sh. Same loop
# script, window 21, decision every 2 frames, 33 decisions. One own CoppeliaSim on $PORT (never 23000).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25702}; BODY=${BODY:-c10}
OUT=${OUT:-results/wm/closed_loop_orig/physics/joint_branches}
CK=${CK:-results/eval/round1_branches_s0/ckpt/hex.pt}
CW=data/counterfactual_walks
if [ $BODY = c10 ]; then CAND=$CW/c10_clips_train; MORPH=c10f10t10=medauroidea_c10f10t10.ttt; DR=0
else CAND=$CW/c08_clips_heldout; MORPH=c08f09t09=medauroidea_c08f09t09.ttt; DR=20000; fi
declare -A G=([turn_s0.29]=40101 [turn_s0.56]=40110 [side_L_lvl0]=40163 [side_R_lvl1]=40212 [speed_c7.1]=40012 [speed_c8.8]=40030)
for c in turn_s0.29 turn_s0.56 side_L_lvl0 side_R_lvl1 speed_c7.1 speed_c8.8; do
  e=${G[$c]}; goal=$CW/c10_clips_heldout/hexapod_ep$e.npz
  if [ $BODY = c10 ]; then room=$goal; else room=$CAND/hexapod_ep$((e + DR)).npz; fi
  for mech in ${MECHS:-direct rollout}; do
    o=$OUT/$BODY; [ $mech = random ] && o=$OUT/random/$BODY
    r=$(nice -n 10 $PY sim/control/close_loop_hexapod_froude.py --mechanism $mech --window 21 --ckpt $CK \
        --goal $goal --candidates_dir $CAND --morph $MORPH --orig_room $room --cpg_clock \
        --port $PORT --device ${DEVICE:-cuda} --out $o 2>&1 | tee -a $OUT/loop_$BODY.log \
        | grep -E "closed loop|Error|FELL|MAE|corr|cpg clock|original room" | tr '\n' ' ')
    echo "$BODY joint_branches_orig $mech $c | $r" | tee -a $OUT/summary_$BODY.txt
  done
done
echo PHYSICS_ORIG_${BODY}_DONE
