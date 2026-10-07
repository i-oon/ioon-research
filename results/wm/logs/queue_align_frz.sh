#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 1200424 2>/dev/null; do sleep 60; done
sleep 600
P=$(bash scripts/tools/launch_guarded.sh b1_align_frz results/wm/logs/b1_babble_align_frz.log env FREEZE_FTM=1 OUT=results/eval/b1_babble_align_frz bash scripts/run/b1_babble_align.sh wm/runs/round1_hexonly_s0_rr/best.pt 44 88)
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "align frozen ended $(date)"
