#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py; CW=data/counterfactual_walks; CK=results/eval/round1_branches_s0/ckpt
.venv/bin/python3 $RO --embodiment hexapod --cf_dir $CW/c10_branches_heldout --ckpt B=$CK/hex.pt --pairs 1 11 --dump results/check/rollout_channels/c10.npz > results/check/rollout_channels/c10.txt 2>&1
.venv/bin/python3 $RO --embodiment b1 --cf_dir $CW/b1_branches_heldout --ckpt B=$CK/b1.pt --pairs 1 11 --dump results/check/rollout_channels/b1.npz > results/check/rollout_channels/b1.txt 2>&1
echo DONE
