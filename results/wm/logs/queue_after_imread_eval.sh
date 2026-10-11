#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 1620399 2>/dev/null; do sleep 60; done
sleep 300
.venv/bin/python3 /tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad/atten2_imread.py 2>&1 | grep -E "^c10|^b1" > results/check/rollout_channels/atten_gray_imread.txt
sleep 120
.venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_predcheck.py --ckpt gray=results/eval/round1_branches_s0_gray/ckpt/b1.pt 2>&1 | grep -v -i warn | tail -25
sleep 120
.venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_predcheck.py --ckpt gray_imread=results/eval/round1_branches_s0_gray_imread/ckpt/b1.pt 2>&1 | grep -v -i warn | tail -25
echo QUEUE_DONE
