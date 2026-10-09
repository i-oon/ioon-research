#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
for v in "21 0" "21 2" "41 0" "41 4" "61 6"; do set -- $v
.venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py --conditions all --embodiment hexapod --candidates_dir data/counterfactual_walks/c10_clips_train --goal_dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --modes roll_live --roll_window $1 --read_skip $2 --ckpt B=results/eval/round1_branches_s0/ckpt/hex.pt 2>&1 | grep -E "w=|oracle|Error|Traceback" > results/check/rollout_skip/c10_w$1_skip$2.txt
sleep 60
done
echo DONE
