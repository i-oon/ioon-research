#!/usr/bin/env bash
# after queue_after_imread_eval: per-goal offline selection (c10, live rollout, gait-phase matched) + per-channel read slopes
cd /home/fibo07/ioon/ioon-research
while kill -0 1625302 2>/dev/null; do sleep 60; done
sleep 300
O=results/check/rollout_phase
for N in round1_branches_s0_gray round1_branches_s0_gray_imread; do
  .venv/bin/python3 scripts/diagnostics/objective_experiments/selection_eval.py --conditions all --embodiment hexapod --candidates_dir data/counterfactual_walks/c10_clips_train --goal_dir data/counterfactual_walks/c10_clips_heldout --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --modes direct roll_live --phase_align --ckpt $N=results/eval/$N/ckpt/hex.pt 2>&1 | grep -E "w=|oracle|Error|Traceback" > $O/c10_phase_$N.txt
  sleep 120
done
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
for N in round1_branches_s0_gray round1_branches_s0_gray_imread; do
  .venv/bin/python3 $RO --embodiment hexapod --cf_dir data/counterfactual_walks/c10_branches_heldout --ckpt B=results/eval/$N/ckpt/hex.pt --pairs 11 --dump results/check/rollout_channels/c10_$N.npz > results/check/rollout_channels/c10_$N.txt 2>&1
  sleep 60
  .venv/bin/python3 $RO --embodiment b1 --cf_dir data/counterfactual_walks/b1_branches_heldout --ckpt B=results/eval/$N/ckpt/b1.pt --pairs 11 --dump results/check/rollout_channels/b1_$N.npz > results/check/rollout_channels/b1_$N.txt 2>&1
  sleep 60
done
.venv/bin/python3 scripts/diagnostics/objective_experiments/readout_channels.py results/check/rollout_channels/c10_round1_branches_s0_gray*.npz results/check/rollout_channels/b1_round1_branches_s0_gray*.npz > results/check/rollout_channels/analysis_gray_vs_imread.txt 2>&1
echo QUEUE2_DONE
