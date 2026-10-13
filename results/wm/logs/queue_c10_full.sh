#!/usr/bin/env bash
# c10 wall data, FULL collection (user OK 2026-10-13), B1-matched by start states (~20 % corners).
# 6 own CoppeliaSim instances (cap 6), no GPU. Episodes / looks disjoint across splits.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_c10_wall_data.py
PORTS="25780 25782 25784 25786 25788 25790"
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25784 25786 25788 25790; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in $PORTS; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
$PY $S physics --split test  --n 15 --n_corner 6  --ep0 10800 --look0 10800 --ports $PORTS
$PY $S physics --split val   --n 8  --n_corner 3  --ep0 10500 --look0 10500 --ports $PORTS
$PY $S physics --split train --n 42 --n_corner 16 --ep0 10000 --look0 10000 --ports $PORTS
$PY $S select --split train
$PY $S report --splits train val test
for sp in test val train; do $PY $S render --split $sp --ports $PORTS; done
ps aux | awk '/rpcPort=2578[0-9]|rpcPort=2579[0-9]/ && !/awk/ {print $2}' | xargs -r kill -9; ps aux | awk '/pythonLauncher/ && !/awk/ {print $2}' | xargs -r kill 2>/dev/null
echo QUEUE_C10_FULL_DONE
