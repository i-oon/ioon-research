#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
while kill -0 2297739 2>/dev/null; do sleep 60; done
echo "sweep ended $(date)"; sleep 600
P=$(bash scripts/tools/launch_guarded.sh b1_variants results/wm/logs/b1_babble_variants.log bash scripts/run/b1_babble_variants.sh wm/runs/round1_hexonly_s0_rr/best.pt)
while kill -0 $P 2>/dev/null; do sleep 60; done
echo "variants ended $(date)"; sleep 600
PFX=rr_ RUN_TAG=_rr bash scripts/run/round1_counterfactual.sh B < /dev/null > results/wm/logs/joint_rr_launch.log 2>&1
echo "joint started $(date)"; cat results/wm/logs/joint_rr_launch.log
