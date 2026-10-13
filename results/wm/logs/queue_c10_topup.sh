#!/usr/bin/env bash
# c10 wall TOP-UP (user OK 2026-10-13): fill the band x approach-type cells that are short of B1 (fairness rule: every
# split's cell counts = B1's), with the live == replay sync and start placement by approach type; excess is trimmed
# afterwards by whole episodes. Numbers: new blocks, disjoint from the full run (train 12000-, val 12500-, test 12800-;
# corner episodes at block + 200). 6 own CoppeliaSim instances (cap 6), no GPU.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_c10_wall_data.py
PORTS="25780 25782 25784 25786 25788 25790"
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25784 25786 25788 25790; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in $PORTS; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
$PY $S physics --split val   --n 8  --group oblique --n_corner 6 --ep0 12500 --look0 12500 --ports $PORTS
$PY $S physics --split test  --n 12 --group oblique --n_corner 3 --ep0 12800 --look0 12800 --ports $PORTS
$PY $S physics --split test  --n 6  --group head-on             --ep0 12850 --look0 12850 --ports $PORTS
$PY $S physics --split val   --n 2  --group head-on             --ep0 12550 --look0 12550 --ports $PORTS
$PY $S physics --split train --n 42 --group oblique             --ep0 12000 --look0 12000 --ports $PORTS
$PY $S report --splits train val test
ps aux | awk '/rpcPort=2578[0-9]|rpcPort=2579[0-9]/ && !/awk/ {print $2}' | xargs -r kill -9; ps aux | awk '/pythonLauncher/ && !/awk/ {print $2}' | xargs -r kill 2>/dev/null
echo QUEUE_C10_TOPUP_DONE
