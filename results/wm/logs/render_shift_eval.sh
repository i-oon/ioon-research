#!/usr/bin/env bash
# Step 2 of the plan: are round-1 results an artifact of room scaling? Same models, heldout re-rendered with room size
# random 8-26.5 m (shared by all bodies), vs the original renders of the same 24 branch groups.
cd /home/fibo07/ioon/ioon-research
CW=data/counterfactual_walks; PY=.venv/bin/python3
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
RS="LIB=$CW/rs_b1_clips_heldout HEXT=$CW/rs_c10_clips_heldout C08=$CW/rs_c08_clips_heldout CFH=$CW/rs_c10_branches_heldout CFB=$CW/rs_b1_branches_heldout CFC=$CW/rs_c08_branches_heldout"
orig24() {  # name, hex ckpt, b1 ckpt
  echo "=== $1: read-out on ORIGINAL renders, same 24 groups"
  for b in hexapod:c10:$2 b1:b1:$3 hexapod:c08:$2; do
    IFS=: read e d ck <<< "$b"
    echo "--- $d"; $PY $RO --embodiment $e --cf_dir $CW/orig24_${d}_branches_heldout --ckpt $1=$ck --pairs 1 11 2>&1 | grep -A3 "^==="
  done
}
env $RS bash scripts/run/eval_suite.sh round1_branches_s0_rs joint wm/runs/round1_branches_s0/best.pt > results/wm/logs/eval_round1_branches_s0_rs.log 2>&1
sleep 300
orig24 round1_branches_s0 results/eval/round1_branches_s0/ckpt/hex.pt results/eval/round1_branches_s0/ckpt/b1.pt > results/eval/round1_branches_s0_rs/orig24_readout.txt 2>&1
sleep 300
env $RS bash scripts/run/eval_suite.sh round1_hexonly_s0_rs hexonly wm/runs/round1_hexonly_s0/best.pt > results/wm/logs/eval_round1_hexonly_s0_rs.log 2>&1
sleep 300
orig24 round1_hexonly_s0 results/eval/round1_hexonly_s0/ckpt/hex.pt results/eval/round1_hexonly_s0/ckpt/b1.pt > results/eval/round1_hexonly_s0_rs/orig24_readout.txt 2>&1
echo RS_EVAL_DONE
