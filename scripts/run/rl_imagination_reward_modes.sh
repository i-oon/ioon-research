#!/bin/bash
# Slide 16's imagination-RL recipe (H=12, symlog + return norm, MC anchor every 200) under four reward
# definitions, same start/goal clip, same budget. Common comparable number printed at the end of each run.
#   direct_const    original test: constant steady-state Froude goal, reward never touches the FTM
#   direct_tv       same readout, per-timestep goal from the clip
#   rollout_froude  Froude read from the FTM-imagined transition (ITM -> body_head), per-timestep goal
#   embed           distance between the FTM-imagined next embedding and the goal clip's own
cd "$(dirname "$0")/../.."
for M in direct_const direct_tv rollout_froude embed; do
  echo "=== $M ==="
  .venv/bin/python3 scripts/diagnostics/objective_experiments/rl_imagination_isolation_v2.py \
    --mode $M --horizon 12 --iters 2000 --batch 4 --stabilize --anchor_every 200 --mc_horizon 20 \
    --anchor_steps 10 --eval_every 50 \
    --out results/wm/closed_loop/rl_modes_$M.npz --save_actor results/wm/closed_loop/rl_modes_$M.pt
done
echo ALL DONE
