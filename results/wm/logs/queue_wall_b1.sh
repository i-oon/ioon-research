#!/usr/bin/env bash
# B1 wall data, full collection (DATA_PLAN 13; user go 2026-10-12): train 22 / val 4 / test 8 episodes, far band 5 s,
# train = stratified 14 of 28 commands per state. Splits disjoint in episode and room look. CPU physics (5 workers) then
# render on 4 own CoppeliaSim instances (cap 6), then the per-split report.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_b1_wall_data.py
nice -n 10 $PY $S physics --split train --n 22 --ep0 8000 --look0 8000 --workers 5
nice -n 10 $PY $S physics --split val   --n 4  --ep0 8500 --look0 8500 --workers 4
nice -n 10 $PY $S physics --split test  --n 8  --ep0 8800 --look0 8800 --workers 5
$PY $S select --split train
$PY $S report --splits train val test
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25780 25782 25784 25786; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in 25780 25782 25784 25786; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
for sp in test val train; do nice -n 10 $PY $S render --split $sp --ports 25780 25782 25784 25786; done
ps aux | awk '/rpcPort=2578[0-9]/ && !/awk/ {print $2}' | xargs -r kill -9; ps aux | awk '/pythonLauncher/ && !/awk/ {print $2}' | xargs -r kill 2>/dev/null
echo QUEUE_WALL_B1_DONE
