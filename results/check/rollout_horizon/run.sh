#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
for H in 4 8; do
.venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py --conditions all --embodiment hexapod --candidates_dir data/counterfactual_walks/c10_clips_train --goal_dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --horizon $H --modes direct roll_live --ckpt B=results/eval/round1_branches_s0/ckpt/hex.pt 2>&1 | grep -E "w=|oracle" > results/check/rollout_horizon/c10_h$H.txt
sleep 120
done
