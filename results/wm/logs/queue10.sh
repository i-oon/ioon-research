#!/usr/bin/env bash
# goal alignment ablation (F339, Codex step 1): rollout compares with the mean goal label over the imagined window (--goal_align imagined); no weight change, no calibration. Uncalibrated baseline: queue4 / queue6.
cd /home/fibo07/ioon/ioon-research
N=round1_branches_s0_gray_imread; CKD=results/eval/$N/ckpt; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
.venv/bin/python3 scripts/dataset/render_shift_heldout.py launch --ports 25760 25762
# strong / mild turn by start state, rollout (uncalibrated baseline: summary.txt)
G=data/counterfactual_walks/c10_clips_heldout; S=results/check/loop_prewalk/c10/summary_ga.txt; rm -f $S
for ep in 40110 40101; do for pw in speed_c7.1 turn_s0.56; do for P in 20 24 28 32 36; do
  r=$(nice -n 10 .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism rollout --window 21 --ckpt $CKD/hex.pt --goal $G/hexapod_ep$ep.npz --candidates_dir data/counterfactual_walks/c10_clips_train --morph c10f10t10=medauroidea_c10f10t10.ttt --orig_room $G/hexapod_ep$ep.npz --cpg_clock --prewalk $pw --prewalk_frames $P --grade_from 10 --goal_align imagined --port 25762 --out results/check/loop_prewalk/c10 2>&1 | grep -oE "mean L2 error [0-9.]+|Error.*" | tr '\n' ' ')
  echo "$ep rollout_ga $pw $P | $r" | tee -a $S
done; done; done
# all goals, rollout only (baseline: queue4 round1_branches_s0_gray_imread)
O=results/wm/closed_loop_orig/physics/${N}_ga
PORT=25762 BODY=c10 MECHS=rollout GA=imagined CK=$CKD/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
PORT=25762 BODY=c08 MECHS=rollout GA=imagined CK=$CKD/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
PORT=25760 MECHS=rollout GA=imagined CK=$CKD/b1.pt OUT=$O bash scripts/run/physics_orig_b1.sh
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE10_DONE
