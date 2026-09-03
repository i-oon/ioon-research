#!/usr/bin/env bash
#
# F157. Does the lag-3 task genuinely need the action once a model is trained on it?
#
#   bash scripts/com7_pretrain_lag3.sh          # on com7, under tmux, about ninety minutes
#
# **This is not a candidate model and nothing should be tuned against it.** F156 measured
# `null/real` peaking at 1.078 on the insect and 1.053 on the B1 at lag 3, but measured it with an
# ITM and a forward model that were both fitted at lag 1 -- every lag-3 row there is
# off-distribution for them, so 1.078 is a lower bound and cannot be trusted to reject lag 3.
# Training briefly at lag 3 removes that confound and nothing else.
#
#   frame_stride 3    the peak on both robots and in every behaviour family (F156)
#   action_chunk 0    "follow frame_stride". **Widening the stride without this is measurably
#                     wrong**: `z` is asked to summarise k steps while `L_motion` still scores it
#                     against one, which took validation motion from 0.218 to 0.928 (F88)
#   lambda_hinge 0    hinge OFF, so there is no separation pressure and no explosion risk. F154's
#                     divergence came from a hinge acting where the prediction loss could not see
#   lambda_readout 0  same reason
#   lambda_recon 1.0  back to the default. The 3.0 of F153 existed only to counter the hinge
#   epochs 10         enough to fit the new target, far short of a model worth keeping
#
# **The decision this run exists to make, fixed before it runs:**
#
#   null/real rises clearly above 1.078   a target worth weighting exists -> add the hinge and
#                                         pretrain fully
#   null/real stays near 1.08             the limit is the representation, not the objective. No
#                                         frameskip reaches it; the next move is the encoder or
#                                         the prediction target itself, and **not another
#                                         objective term**. Report that plainly
#
# Read per body and per family. Insect turning is the sharpest cell: 51.4% at lag 1 (F155), 92.9%
# at lag 3 off-distribution (F156).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
RUN=wm/runs/beh12_lag3_nohinge

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "=== short lag-3 pretrain, hinge off  $(date '+%F %T')"
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_lag3_nohinge \
    --sources hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2 \
    --frame_stride 3 --action_chunk 0 \
    --lambda_recon 1.0 --lambda_hinge 0.0 --lambda_readout 0.0 \
    --epochs 10
fi

echo
echo "=== does the action matter now?  the one number this run exists for"
$PY -u scripts/diagnostics/objective_experiments/action_necessity.py --ckpt "$RUN/best.pt" \
  --data data/allocentric/beh12_c08f09t09_flat --embodiment hexapod --lags 1 2 3 5
$PY -u scripts/diagnostics/objective_experiments/action_necessity.py --ckpt "$RUN/best.pt" \
  --data data/allocentric/beh12_b1_flat --embodiment b1 --lags 1 2 3 5

echo
echo "=== and did anything else break?  a stride change is not free (F88)"
$PY -u scripts/diagnostics/shared_body_target/body_head_calibration.py --ckpt "$RUN/best.pt" \
  --data hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat

echo
echo "=== done  $(date '+%F %T')   send back this whole log"
