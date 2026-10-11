#!/usr/bin/env bash
# Re-measure every current B1 result on the F293-corrected B1 v3 data (poses + re-rendered frames;
# library 23 clips without the unverifiable b1_ep2002). Sequential, one GPU job at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
echo "=== hexapod-only Froude-only + current adaptation"
OUTTAG=fmd_v3_f293 bash scripts/run/hexonly_v3_new.sh fmd_beh24_s0 fmd_beh24_s1
echo "=== joint-command decoder + current adaptation"
OUTTAG=dt0_v3_f293 bash scripts/run/hexonly_v3_new.sh dt0_beh24_s0 dt0_beh24_s1
echo "=== jointD3 (pretrained on pre-fix labels), Stage 2 refit on fixed data"
rm -rf wm/runs/jointD3_sim_s0/stage2only wm/runs/jointD3_nosim_s0/stage2only
bash scripts/run/eval_joint_models_v3.sh jointD3_sim_s0 jointD3_nosim_s0
echo "=== vision-read goal, B1"
$PY scripts/diagnostics/objective_experiments/selection_eval.py \
  --candidates_dir data/egocentric_v3/beh12_b1_ego_flat_cleantrain --cache results/wm/cache/selection_eval_cands_v3.pt \
  --windows 21 --goal_source vision \
  --ckpt hexad_S0=wm/runs/fmd_beh24_s0/b1_v3_new/ckpt.pt --ckpt hexad_S1=wm/runs/fmd_beh24_s1/b1_v3_new/ckpt.pt \
  --ckpt joint_nosim=wm/runs/jointD3_nosim_s0/stage2only/ckpt_stage2only.pt \
  --ckpt joint_sim=wm/runs/jointD3_sim_s0/stage2only/ckpt_stage2only.pt 2>&1 | grep -E "w=11|bounds"
echo REEVAL_F293_DONE
