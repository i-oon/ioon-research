#!/usr/bin/env bash
# F336 confirmation test: start state = standing / walking straight / already turning, before the planner takes over.
# model _gray_imread, c10, goals turn 0.56 + turn 0.29, direct + rollout, 5 hand-over times (gait phase / heading vary),
# graded from 10 frames after the hand-over. Everything else as the physics loops.
cd /home/fibo07/ioon/ioon-research
CK=results/eval/round1_branches_s0_gray_imread/ckpt/hex.pt; G=data/counterfactual_walks/c10_clips_heldout
O=results/check/loop_prewalk/c10; S=$O/summary.txt
run() {  # goal_ep mech prewalk frames
  if [ -n "$3" ]; then PW="--prewalk $3 --prewalk_frames $4"; else PW=""; fi
  r=$(nice -n 10 .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism $2 --window 21 --ckpt $CK \
      --goal $G/hexapod_ep$1.npz --candidates_dir data/counterfactual_walks/c10_clips_train \
      --morph c10f10t10=medauroidea_c10f10t10.ttt --orig_room $G/hexapod_ep$1.npz --cpg_clock $PW --grade_from 10 \
      --port 25762 --out $O 2>&1 | grep -oE "mean L2 error [0-9.]+|Error.*|ego view check.*" | tr '\n' ' ')
  echo "$1 $2 ${3:-stand} ${4:-0} | $r" | tee -a $S
}
for ep in 40110 40101; do
  for m in rollout direct; do
    run $ep $m "" ""
    for P in 20 24 28 32 36; do run $ep $m speed_c7.1 $P; run $ep $m turn_s0.56 $P; done
  done
done
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE6_DONE
