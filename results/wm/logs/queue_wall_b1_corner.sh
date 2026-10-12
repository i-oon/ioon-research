#!/usr/bin/env bash
# B1 wall data: CORNER episodes added (2026-10-12; corners were 7/135 train, 1/25 val, 6/51 test start states).
# 20 s corner episodes (start 2.5-4 m from both walls, facing into the corner; branch states only where both walls
# are close), ~2 corner states each. New episode / look numbers: splits stay disjoint. Target ~20 % corner states.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_b1_wall_data.py
nice -n 10 $PY $S physics --split train --n 13 --ep0 8030 --look0 8030 --workers 5 --corner
nice -n 10 $PY $S physics --split val   --n 3  --ep0 8530 --look0 8530 --workers 3 --corner
nice -n 10 $PY $S physics --split test  --n 4  --ep0 8830 --look0 8830 --workers 4 --corner
$PY $S select --split train
$PY $S report --splits train val test
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25780 25782 25784 25786; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in 25780 25782 25784 25786; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
for sp in test val train; do nice -n 10 $PY $S render --split $sp --ports 25780 25782 25784 25786; done
ps aux | awk '/rpcPort=2578[0-9]/ && !/awk/ {print $2}' | xargs -r kill -9; ps aux | awk '/pythonLauncher/ && !/awk/ {print $2}' | xargs -r kill 2>/dev/null
echo QUEUE_WALL_B1_CORNER_DONE
