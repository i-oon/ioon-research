#!/usr/bin/env bash
# c10 wall top-up PILOT: 1 oblique + 1 head-on episode, checks the approach-group placement and the live == replay sync.
# 2 own CoppeliaSim instances (cap 6), no GPU.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; S=scripts/dataset/collect_c10_wall_data.py
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25780 25782; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in 25780 25782; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
$PY $S physics --split pilot_topup --n 1 --group oblique --ep0 9500 --look0 9500 --ports 25780 &
$PY $S physics --split pilot_topup --n 1 --group head-on --ep0 9501 --look0 9501 --ports 25782 &
wait
$PY $S report --splits pilot_topup
echo PILOT_TOPUP_DONE
