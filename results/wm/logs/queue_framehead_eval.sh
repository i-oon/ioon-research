#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 2160529 2>/dev/null; do sleep 60; done
echo "train ended $(date)"; grep -E "^epoch|best val" results/wm/logs/round2_framehead_s0.log | cut -c1-120
sleep 600   # cool-down
bash scripts/run/eval_suite.sh round2_framehead_s0 joint wm/runs/round2_framehead_s0/best.pt > results/wm/logs/eval_round2_framehead_s0.log 2>&1 &
sleep 5; P=$(ps -eo pid,cmd | grep "[b]ash scripts/run/eval_suite.sh round2_framehead_s0" | awk '{print $1}' | head -1)
bash scripts/tools/gpu_guard.sh $P >> results/wm/logs/gpu_guard_framehead_eval.log 2>&1
echo "eval ended $(date)"; cat results/eval/round2_framehead_s0/summary.txt
