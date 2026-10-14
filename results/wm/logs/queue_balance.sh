#!/usr/bin/env bash
# BALANCE top-up (user OK 2026-10-14; DATA_PLAN 13 fairness rule): collect only the short cells (band x approach x side)
# so B1 and c10 reach B1's current cell counts, left / right even; excess trimmed afterwards by whole episodes.
# c10: 4 own CoppeliaSim (25780-25786), one job chain per instance. B1: MuJoCo physics, 2 CPU workers. Cap: 6 sims, load ~12.
# Numbers: c10 train 14000-, val 14500-, test 14800- (corners + 200); B1 train 8100-, val 8600-, test 8900-.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; C=scripts/dataset/collect_c10_wall_data.py; B=scripts/dataset/collect_b1_wall_data.py
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for P in 25780 25782 25784 25786; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in 25780 25782 25784 25786; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
( $PY $C physics --split train --n 4 --group head-on --side L --ep0 14000 --look0 14000 --ports 25780 ) > results/wm/logs/bal_c10_p1.log 2>&1 &
( $PY $C physics --split train --n 0 --n_corner 3 --side R --ep0 14030 --look0 14030 --ports 25782
  $PY $C physics --split val --n 0 --n_corner 1 --side L --ep0 14500 --look0 14500 --ports 25782 ) > results/wm/logs/bal_c10_p2.log 2>&1 &
( $PY $C physics --split test --n 3 --group head-on --side L --ep0 14800 --look0 14800 --ports 25784 ) > results/wm/logs/bal_c10_p3.log 2>&1 &
( $PY $C physics --split train --n 2 --group oblique --side L --ep0 14010 --look0 14010 --ports 25786
  $PY $C physics --split train --n 1 --group head-on --side R --ep0 14020 --look0 14020 --ports 25786
  $PY $C physics --split test --n 1 --group head-on --side R --ep0 14810 --look0 14810 --ports 25786 ) > results/wm/logs/bal_c10_p4.log 2>&1 &
( $PY $B physics --split train --n 5 --group head-on --side R --ep0 8100 --look0 8100 --workers 2
  $PY $B physics --split train --n 2 --group oblique --side R --ep0 8110 --look0 8110 --workers 2
  $PY $B physics --split train --n 1 --group oblique --side L --ep0 8120 --look0 8120 --workers 2
  $PY $B physics --split train --n 1 --corner --ep0 8130 --look0 8130 --workers 2
  $PY $B physics --split val --n 2 --group head-on --side L --ep0 8600 --look0 8600 --workers 2
  $PY $B physics --split val --n 1 --group oblique --side R --ep0 8610 --look0 8610 --workers 2
  $PY $B physics --split test --n 3 --group oblique --side R --ep0 8900 --look0 8900 --workers 2 ) > results/wm/logs/bal_b1.log 2>&1 &
wait
ps aux | awk '/rpcPort=2578[0-6]/ && !/awk/ {print $2}' | xargs -r kill -9
echo QUEUE_BALANCE_DONE
