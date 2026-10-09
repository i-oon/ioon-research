#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
CK=results/eval/round1_branches_s0/ckpt/both_rollout.pt
O=results/wm/closed_loop_orig/physics/joint_branches_rollproj; mkdir -p $O
.venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py --conditions all --embodiment hexapod --candidates_dir data/counterfactual_walks/c10_clips_train --goal_dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --ckpt R=$CK 2>&1 | grep -E "w=|bounds|oracle" > $O/offline_c10.txt
sleep 300
PORT=25702 BODY=c10 CK=$CK OUT=$O bash scripts/run/physics_orig_hex.sh
sleep 300
PORT=25702 BODY=c08 CK=$CK OUT=$O bash scripts/run/physics_orig_hex.sh
sleep 300
PORT=25700 CK=$CK OUT=$O bash scripts/run/physics_orig_b1.sh
echo QUEUE_DONE
