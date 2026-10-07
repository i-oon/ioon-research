#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 805456 2>/dev/null; do sleep 60; done
echo "chain ended $(date)"; sleep 600
P=$(bash scripts/tools/launch_guarded.sh b1_align results/wm/logs/b1_babble_align.log bash scripts/run/b1_babble_align.sh wm/runs/round1_hexonly_s0_rr/best.pt 44 88)
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "align ended $(date)"
