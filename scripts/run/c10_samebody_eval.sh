#!/usr/bin/env bash
# Same-body selection test (2026-09-28): is the rollout's weakness independent of any body change?
# Goals: the six c10f10t10 held-out goal clips. Candidates: c10f10t10's own beh24 val library (24 clips,
# never trained on; the hexapod projector in c08_zeroshot/ was fit on beh24 cleantrain, so the library is
# out-of-sample for it). No cross-embodiment, no leg-length change. Metric: NS (w=0 and w=11).
#   bash scripts/run/c10_samebody_eval.sh 2>&1 | tee results/wm/logs/c10_samebody_eval.log
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
CK=()
for M in A:beh24_stride5_cleansplit A2:beh24_stride5_A2 SW0:babsw_stride5_s0 SW1:babsw_stride5_s1 \
         RP0:babrp_stride5_s0 RP1:babrp_stride5_s1 DT0:dt0_beh24_s0 DT1:dt0_beh24_s1; do
  f=wm/runs/${M#*:}/c08_zeroshot/ckpt_lib_zeroshot.pt
  [ -f $f ] && CK+=(--ckpt "${M%%:*}_c10=$f") || echo "skip ${M%%:*} (no $f)"
done
$PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
    --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
    --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
    --cache results/wm/cache/anova_hex_beh24val.pt "${CK[@]}" --windows 0 11 | grep -E "w=|bounds"
