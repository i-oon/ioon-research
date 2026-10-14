#!/usr/bin/env bash
# FINALISE the wall data and run D1-D3 (user OK 2026-10-14). Stops unless BOTH bodies' balance dry runs are EXACT.
# 6 own CoppeliaSim for rendering (3 per body), then one GPU job (prediction check). Cap: 6 sims, one GPU job.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; BAL=scripts/dataset/balance_wall_data.py
B=scripts/dataset/collect_b1_wall_data.py; C=scripts/dataset/collect_c10_wall_data.py
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
for b in b1 c10; do $PY $BAL --body $b > results/wm/logs/final_dry_$b.log 2>&1; tail -1 results/wm/logs/final_dry_$b.log; done
grep -q "BALANCE EXACT" results/wm/logs/final_dry_b1.log && grep -q "BALANCE EXACT" results/wm/logs/final_dry_c10.log \
  || { echo "STOP: balance not exact (see final_dry_*.log)"; exit 1; }
for b in b1 c10; do $PY $BAL --body $b --apply 2>&1 | tail -10; done
$PY $B select --split train 2>&1 | tail -2
$PY $C select --split train 2>&1 | tail -2
PORTS="25780 25782 25784 25786 25788 25790"
for P in $PORTS; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
for P in $PORTS; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done
( for s in test val train; do $PY $B render --split $s --ports 25780 25782 25784; done ) > results/wm/logs/final_render_b1.log 2>&1 &
( for s in test val train; do $PY $C render --split $s --ports 25786 25788 25790; done ) > results/wm/logs/final_render_c10.log 2>&1 &
wait
ps aux | awk '/rpcPort=2578[0-9]|rpcPort=2579[0-9]/ && !/awk/ {print $2}' | xargs -r kill -9
grep -h "render\] done\|errors" results/wm/logs/final_render_*.log | tail -6
$PY scripts/dataset/check_wall_data.py --body b1 --render > results/wm/logs/final_gate_b1.log 2>&1
$PY scripts/dataset/check_wall_data.py --body c10 --render --ref_body b1 > results/wm/logs/final_gate_c10.log 2>&1
grep -h "GATES\|cell counts\|  train:\|  val:\|  test:" results/wm/logs/final_gate_*.log
grep -q "GATES ALL PASS" results/wm/logs/final_gate_b1.log && grep -q "GATES ALL PASS" results/wm/logs/final_gate_c10.log \
  || { echo "STOP: gates do not all pass -- no prediction check"; exit 1; }
$PY scripts/diagnostics/wall_pilot/wall_predcheck.py --body b1 > results/wm/logs/final_predcheck_b1.log 2>&1
$PY scripts/diagnostics/wall_pilot/wall_predcheck.py --body c10 > results/wm/logs/final_predcheck_c10.log 2>&1
grep -h "PASS\|FAIL" results/wm/logs/final_predcheck_*.log
echo QUEUE_FINALIZE_DONE
