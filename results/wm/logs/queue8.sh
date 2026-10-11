#!/usr/bin/env bash
# timing audit tests (2026-10-10): rollout reads 4 imagined steps (window 20 = the imread_k it was trained on) vs 5 (21)
cd /home/fibo07/ioon/ioon-research
CK=results/eval/round1_branches_s0_gray_imread/ckpt/hex.pt
O=results/check/rollout_phase
for RW in 20 21; do
  .venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py --conditions all --embodiment hexapod --candidates_dir data/counterfactual_walks/c10_clips_train --goal_dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --roll_window $RW --modes roll_live --phase_align --ckpt imread=$CK 2>&1 | grep -E "w=|oracle|Error|Traceback" > $O/c10_imread_rw$RW.txt
  sleep 60
done
.venv/bin/python3 scripts/dataset/render_shift_heldout.py launch --ports 25762
G=data/counterfactual_walks/c10_clips_heldout; S=results/check/loop_prewalk/c10/summary_rw20.txt; rm -f $S
for P in 20 24 28 32 36; do
  r=$(nice -n 10 .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism rollout --window 21 --roll_window 20 --ckpt $CK --goal $G/hexapod_ep40110.npz --candidates_dir data/counterfactual_walks/c10_clips_train --morph c10f10t10=medauroidea_c10f10t10.ttt --orig_room $G/hexapod_ep40110.npz --cpg_clock --prewalk speed_c7.1 --prewalk_frames $P --grade_from 10 --port 25762 --out results/check/loop_prewalk/c10 2>&1 | grep -oE "mean L2 error [0-9.]+|Error.*" | tr '\n' ' ')
  echo "40110 rollout rw20 speed_c7.1 $P | $r" | tee -a $S
done
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE8_DONE
