#!/usr/bin/env bash
# after hexapod-only (random rooms) finishes: cool-down -> B1 babbling sweep -> cool-down -> joint training (random rooms)
cd /home/fibo07/ioon/ioon-research
while kill -0 121311 2>/dev/null; do sleep 60; done
echo "hexonly ended $(date)"; grep -E "^epoch|best val" results/wm/logs/round1_hexonly_s0_rr.log | cut -c1-120
sleep 600
P=$(bash scripts/tools/launch_guarded.sh b1_sweep results/wm/logs/b1_babble_sweep.log bash scripts/run/b1_babble_sweep.sh wm/runs/round1_hexonly_s0_rr/best.pt)
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "sweep ended $(date)"
sleep 600
PFX=rr_ RUN_TAG=_rr bash scripts/run/round1_counterfactual.sh B < /dev/null > results/wm/logs/joint_rr_launch.log 2>&1
echo "joint started $(date)"; cat results/wm/logs/joint_rr_launch.log
