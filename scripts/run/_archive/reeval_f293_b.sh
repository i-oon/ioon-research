#!/usr/bin/env bash
# Second part of the F293 re-measurement: hexapod-only unadapted shared latent with the corrected B1,
# the old adaptation recipe on corrected data (page 2 comparison), then the B1 counterfactual read-out.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
echo "=== shared latent, hexapod-only pretrained ITM, B1 not adapted"
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" \
  --model "hexonly_s0=wm/runs/fmd_beh24_s0/best.pt" --model "hexonly_s1=wm/runs/fmd_beh24_s1/best.pt" \
  --out results/deck/shared_latent_hexonly_unadapted_f293 2>&1 | grep -v -i warn | tail -3
echo "=== old adaptation recipe on corrected data (S0)"
bash scripts/run/hexonly_v3.sh 2>&1 | grep -E "w=|bounds|retr|fmd"
echo "=== B1 counterfactual read-out (current adaptation, corrected data)"
rm -f results/wm/cache/counterfactual_readout_b1v3.pt
$PY scripts/diagnostics/objective_experiments/counterfactual_readout_b1.py \
  --ckpt S0=wm/runs/fmd_beh24_s0/b1_v3_new/ckpt.pt --ckpt S1=wm/runs/fmd_beh24_s1/b1_v3_new/ckpt.pt 2>&1 | grep -v -i "warn\|Loading"
echo REEVAL_F293_B_DONE
