#!/usr/bin/env bash
# rerun the wall prediction check with the fixed Spearman (ties -> n/a), unsafe counts and the command-only controls
cd /home/fibo07/ioon/ioon-research
while pgrep -f "bash results/wm/logs/queue2.sh" >/dev/null; do sleep 60; done
sleep 300
for N in gray gray_imread; do
  F=round1_branches_s0_$N
  .venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_predcheck.py --ckpt $N=results/eval/$F/ckpt/b1.pt 2>&1 | grep -v -i "warn\|Loading weights" | tail -32
  sleep 120
done
echo QUEUE3_DONE
