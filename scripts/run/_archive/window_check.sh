#!/usr/bin/env bash
# Read-out window 11 vs 21 frames (0.55 s vs ~1 s, the goal's own timescale) on the current models,
# with the edge-correct Froude labels. Evaluation only.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
CK=(--ckpt hexad_S0=wm/runs/fmd_beh24_s0/b1_v3_new/ckpt.pt --ckpt hexad_S1=wm/runs/fmd_beh24_s1/b1_v3_new/ckpt.pt
    --ckpt joint_nosim=wm/runs/jointD3_nosim_s0/stage2only/ckpt_stage2only.pt
    --ckpt joint_sim=wm/runs/jointD3_sim_s0/stage2only/ckpt_stage2only.pt)
echo "=== B1"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir data/egocentric_v3/beh12_b1_ego_flat_cleantrain \
  --cache results/wm/cache/selection_eval_cands_v3.pt --windows 11 21 "${CK[@]}" | grep -E "w=|bounds"
echo "=== c10"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
  --candidates_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanval --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout \
  --cache results/wm/cache/anova_hex_beh24val.pt --windows 11 21 \
  --ckpt D_S0=wm/runs/fmd_beh24_s0/c08_zeroshot/ckpt_lib_zeroshot.pt --ckpt D_S1=wm/runs/fmd_beh24_s1/c08_zeroshot/ckpt_lib_zeroshot.pt \
  --ckpt joint_nosim=wm/runs/jointD3_nosim_s0/stage2only/ckpt_stage2only.pt | grep -E "w=|bounds"
echo WINDOW_CHECK_DONE
