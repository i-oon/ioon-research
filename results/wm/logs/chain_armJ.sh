#!/usr/bin/env bash
cd /home/fibo07/ioon/ioon-research
CW=data/counterfactual_walks
env LIB=$CW/rr_b1_clips_heldout HEXT=$CW/rr_c10_clips_heldout C08=$CW/rr_c08_clips_heldout \
    CFH=$CW/rr_c10_branches_heldout CFB=$CW/rr_b1_branches_heldout CFC=$CW/rr_c08_branches_heldout \
    HEX=$CW/rr_c10_clips_train B1=$CW/rr_b1_clips_train \
    bash scripts/run/eval_suite.sh round2_align_s0_rr joint wm/runs/round2_align_s0_rr/best.pt
echo ARMJ_EVAL_DONE
