#!/usr/bin/env bash
# BALANCE round 4 (2026-10-14): the 12 c10 start states still short after round 1 (balance_wall_data.py dry run):
# still short after round 2: train middle head-on L 3, val middle corner L 1, test middle head-on R 1 (2x spare).
# 5 own CoppeliaSim (cap 6). Numbers: train 14040-, val 14510 (corner 14710), test 14820- ; corners + 200.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; C=scripts/dataset/collect_c10_wall_data.py
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
PORTS="25780 25782 25784 25786"
for P in $PORTS; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in $PORTS; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
( $PY $C physics --split train --n 2 --group head-on --side L --ep0 14070 --look0 14070 --ports 25780 ) > results/wm/logs/bal4_p1.log 2>&1 &
( $PY $C physics --split train --n 2 --group head-on --side L --ep0 14072 --look0 14072 --ports 25782 ) > results/wm/logs/bal4_p2.log 2>&1 &
( $PY $C physics --split train --n 2 --group head-on --side L --ep0 14074 --look0 14074 --ports 25784 ) > results/wm/logs/bal4_p3.log 2>&1 &
( $PY $C physics --split test --n 2 --group head-on --side R --ep0 14850 --look0 14850 --ports 25786 ) > results/wm/logs/bal4_p4.log 2>&1 &
wait
ps aux | awk '/rpcPort=2578[0-8]/ && !/awk/ {print $2}' | xargs -r kill -9
.venv/bin/python3 scripts/dataset/balance_wall_data.py --body c10 2>&1 | tail -8
echo QUEUE_BALANCE4_DONE
