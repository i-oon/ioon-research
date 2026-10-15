#!/usr/bin/env bash
# OVERNIGHT 2026-10-15 (user asleep). One GPU job at a time, <= 6 own CoppeliaSim, every step stops the chain on failure.
#  A. finish the wall renders (a CoppeliaSim instance hung at 23:55): render with a 20-min timeout per body / split,
#     fresh simulators per attempt, up to 6 attempts; the renderer skips files already written
#  B. gates on both bodies (renders included) -- STOP unless both ALL PASS
#  C. D1-D3 control: frozen current model (round1_branches_s0_gray_imread), B1 then c10
#  D. vision-goal physics (end-to-end goal), B1 / c10 / c08, same settings as the slide's runs
#  E. wall retrain, arm W (round3_wall_s0_gray_imread: current recipe + wall branches, original objectives, 30.5k steps)
#  F. eval_suite joint on it, then G. the same locked D1-D3 on it (before / after)
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; B=scripts/dataset/collect_b1_wall_data.py; C=scripts/dataset/collect_c10_wall_data.py
SP=/tmp/claude-1000/-home-fibo07-ioon-ioon-research/dcfeaa45-e45c-4e8c-b0af-6d41d80d8c56/scratchpad
L=results/wm/logs
PORTS="25780 25782 25784 25786 25788 25790"
sims_up() { for P in $PORTS; do (cd ~/CoppeliaSim && setsid bash -c "tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort=$P -GzmqRemoteApi.cntPort=$((P+1))" > $SP/csim$P.log 2>&1 &); sleep 8; done
  for P in $PORTS; do for i in $(seq 40); do $PY -c "from coppeliasim_zmqremoteapi_client import RemoteAPIClient as R;R('localhost',port=$P).require('sim').getSimulationState()" 2>/dev/null && break; sleep 3; done; done; }
sims_down() { for p in $(ps -eo pid,args | awk '/rpcPort=2578[0-9]|rpcPort=2579[0-9]/ && !/awk/ {print $1}'); do kill -9 $p; done; sleep 3; }
echo "=== A. renders $(date)"
for attempt in 1 2 3 4 5 6; do
  sims_down; sims_up; left=0
  for sp in test val train; do
    ( timeout 1200 $PY $B render --split $sp --ports 25780 25782 25784 ) > $L/night_render_b1_$sp.log 2>&1 &
    ( timeout 1200 $PY $C render --split $sp --ports 25786 25788 25790 ) > $L/night_render_c10_$sp.log 2>&1 &
    wait
    for b in b1 c10; do n=$(grep -m1 -oE "^[0-9]+ files to render" $L/night_render_${b}_$sp.log | cut -d" " -f1)
      d=$(grep -c "render\] done" $L/night_render_${b}_$sp.log); echo "  attempt $attempt $b $sp: to render ${n:-?}, finished $d"
      [ "${n:-1}" != 0 ] && [ "$d" = 0 ] && left=1; done
  done
  [ $left = 0 ] && break
done
sims_down
echo "=== B. gates $(date)"
$PY scripts/dataset/check_wall_data.py --body b1 --render > $L/night_gate_b1.log 2>&1
$PY scripts/dataset/check_wall_data.py --body c10 --render --ref_body b1 > $L/night_gate_c10.log 2>&1
grep -h "GATES\|  7 rendered\|cell counts\|  train:\|  val:\|  test:" $L/night_gate_*.log
grep -q "GATES ALL PASS" $L/night_gate_b1.log && grep -q "GATES ALL PASS" $L/night_gate_c10.log || { echo "STOP: gates"; exit 1; }
echo "=== C. D1-D3 control (frozen current model) $(date)"
for b in b1 c10; do $PY scripts/diagnostics/wall_pilot/wall_predcheck.py --body $b > $L/night_predcheck_$b.log 2>&1; grep -h "PASS\|FAIL" $L/night_predcheck_$b.log; done
echo "=== D. vision-goal physics $(date)"
sims_up
E=results/eval/round1_branches_s0_gray_imread/ckpt; O=results/wm/closed_loop_orig/physics/round1_branches_s0_gray_imread_ga_vision
GS=vision GA=imagined CK=$E/b1.pt OUT=$O PORT=25790 bash scripts/run/physics_orig_b1.sh > $L/night_vision_b1.log 2>&1
for b in c10 c08; do GS=vision GA=imagined CK=$E/hex.pt OUT=$O BODY=$b PORT=25788 bash scripts/run/physics_orig_hex.sh > $L/night_vision_$b.log 2>&1; done
sims_down
grep -h "mean L2 error" $L/night_vision_*.log | wc -l
echo "=== E. wall retrain $(date)"
RECIPE="--grayscale True --aug_brightness 0.0 --aug_contrast 0.4 --aug_blur 2.6 --aug_noise 0.03 --aug_prob 0.5 --aug_bright_mult 3.0 --aug_clean 0.25 --lambda_imread 1.0 --imread_k 4"
N=round3_wall_s0_gray_imread
RUN_TAG=_gray_imread EXTRA="$RECIPE" bash scripts/run/round1_counterfactual.sh W
sleep 60
while ps -eo args | grep -v grep | grep -q -- "--name $N"; do sleep 300; done
[ -s wm/runs/$N/best.pt ] || { echo "STOP: no best.pt for $N"; exit 1; }
grep -E "^epoch" $L/$N.log | tail -3
echo "=== F. eval_suite $(date)"
bash scripts/run/eval_suite.sh $N joint wm/runs/$N/best.pt > $L/night_eval_$N.log 2>&1
cat results/eval/$N/summary.txt | grep -E "^---|w=21"
echo "=== G. D1-D3 on the wall model $(date)"
for b in b1 c10; do $PY scripts/diagnostics/wall_pilot/wall_predcheck.py --body $b --ckpt $N > $L/night_predcheck_wall_$b.log 2>&1; grep -h "PASS\|FAIL" $L/night_predcheck_wall_$b.log; done
echo "QUEUE_OVERNIGHT_DONE $(date)"
