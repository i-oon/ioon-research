#!/usr/bin/env bash
# wall prediction check v2 (mirrored state set): physics (CPU) -> render (4 own CoppeliaSim) -> sheet -> check both models (GPU)
cd /home/fibo07/ioon/ioon-research
S=scripts/diagnostics/wall_pilot/b1_wall_predcheck_data_v2.py
.venv/bin/python3 $S physics --workers 5 2>&1 | grep -v -i warn | tail -3
.venv/bin/python3 scripts/dataset/render_shift_heldout.py launch --ports 25740 25742 25744 25746
.venv/bin/python3 $S render --ports 25740 25742 25744 25746 2>&1 | grep -E "render\]|futures|ERROR" | tail -3
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
.venv/bin/python3 $S sheet
for N in gray gray_imread; do
  .venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_predcheck.py --set v2 --ckpt $N=results/eval/round1_branches_s0_$N/ckpt/b1.pt 2>&1 | grep -v -i "warn\|Loading weights" | tail -32
  sleep 120
done
echo QUEUE7_DONE
