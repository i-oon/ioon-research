#!/usr/bin/env bash
# root cause of the hexapod physics gap (F335): decide every 10 / 20 frames instead of 2 (grading stays every 2 frames,
# --grade_every); model _gray_imread; c10 and c08; direct + rollout; everything else as queue4.
cd /home/fibo07/ioon/ioon-research
N=round1_branches_s0_gray_imread
for R in 10 20; do
  for B in c10 c08; do
    PORT=25762 BODY=$B REPLAN=$R CK=results/eval/$N/ckpt/hex.pt OUT=results/wm/closed_loop_orig/physics/${N}_r$R bash scripts/run/physics_orig_hex.sh
    sleep 120
  done
done
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE5_DONE
