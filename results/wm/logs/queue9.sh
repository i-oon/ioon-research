#!/usr/bin/env bash
# de-shrink calibration test (F340): fit per-channel calibration of the rollout read on VALIDATION branches, then physics
# before/after (uncalibrated numbers: queue4 / queue6, same scripts and settings). Model _gray_imread.
cd /home/fibo07/ioon/ioon-research
N=round1_branches_s0_gray_imread; CKD=results/eval/$N/ckpt; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
C=results/check/rollout_calib; mkdir -p $C; CAL=$C/calib_$N.json; rm -f $CAL
.venv/bin/python3 $RO --embodiment hexapod --cf_dir data/counterfactual_walks/c10_branches_val --ckpt B=$CKD/hex.pt --pairs 11 --dump $C/val_c10.npz > $C/val_c10.txt 2>&1
.venv/bin/python3 $RO --embodiment b1 --cf_dir data/counterfactual_walks/b1_branches_val --ckpt B=$CKD/b1.pt --pairs 11 --dump $C/val_b1.npz > $C/val_b1.txt 2>&1
.venv/bin/python3 scripts/diagnostics/objective_experiments/fit_rollout_calib.py $C/val_c10.npz hexapod $CAL
.venv/bin/python3 scripts/diagnostics/objective_experiments/fit_rollout_calib.py $C/val_b1.npz b1 $CAL
.venv/bin/python3 scripts/dataset/render_shift_heldout.py launch --ports 25760 25762
# strong / mild turn by start state, rollout, calibrated (as queue6)
G=data/counterfactual_walks/c10_clips_heldout; S=results/check/loop_prewalk/c10/summary_cal.txt; rm -f $S
for ep in 40110 40101; do for pw in speed_c7.1 turn_s0.56; do for P in 20 24 28 32 36; do
  r=$(nice -n 10 .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism rollout --window 21 --ckpt $CKD/hex.pt --goal $G/hexapod_ep$ep.npz --candidates_dir data/counterfactual_walks/c10_clips_train --morph c10f10t10=medauroidea_c10f10t10.ttt --orig_room $G/hexapod_ep$ep.npz --cpg_clock --prewalk $pw --prewalk_frames $P --grade_from 10 --calib $CAL --port 25762 --out results/check/loop_prewalk/c10 2>&1 | grep -oE "mean L2 error [0-9.]+|Error.*" | tr '\n' ' ')
  echo "$ep rollout_cal $pw $P | $r" | tee -a $S
done; done; done
# all goals, rollout only, calibrated (as queue4)
O=results/wm/closed_loop_orig/physics/${N}_cal
PORT=25762 BODY=c10 MECHS=rollout CALIB=$CAL CK=$CKD/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
PORT=25762 BODY=c08 MECHS=rollout CALIB=$CAL CK=$CKD/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
PORT=25760 MECHS=rollout CALIB=$CAL CK=$CKD/b1.pt OUT=$O bash scripts/run/physics_orig_b1.sh
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE9_DONE
