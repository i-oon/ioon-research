#!/usr/bin/env bash
#
# F160. Does the body term additionally starve `z` of action detail, on top of the encoder limit?
#
#   bash scripts/com7_lambda_body0_control.sh          # on com7, under tmux, about ninety minutes
#
# **This is not a test of whether the body term causes the action-insensitivity.** F159 already
# rules that out for the lead finding: action recoverability is measured from **frozen V-JEPA2
# embeddings**, which `lambda_body` never touches, so the pose-is-the-command result is
# encoder-level and stands whatever this run says. The narrower question here is whether our own
# coordinate objective **compounds** the encoder limit by shaping `z` into three coarse body-motion
# numbers and leaving no room for joint-level detail. That is real, unmeasured, and the first thing
# a reviewer will ask.
#
# **The comparison is against `beh12_lag3_nohinge`, and every other setting is held identical:**
#
#   frame_stride 3, action_chunk 0     same as the baseline
#   lambda_hinge 0, lambda_readout 0   same
#   lambda_recon 1.0                   same
#   epochs 10                          same
#   body_dim 3, body_channels 0 1 2    **kept, so the architecture is identical**
#   lambda_body 0.0                    **the single variable**
#
# Keeping `body_dim` and zeroing only the weight matters: the modules, the latent width and the
# saved buffers stay the same, so the only difference between the two checkpoints is the gradient.
#
# **The measurement path, pinned before the run (F140's lesson).** `action_necessity.py` builds the
# ITM and the FTM and nothing else -- it never constructs or loads the body head or the motion
# decoder -- and `gather` reads only `embedding_offsets` from the checkpoint. So the path is
# byte-identical to the baseline's:
#
#   null   `ITM(e_t, e_t)`, the same definition as F151, F155, F157
#   lag    3, this checkpoint's own stride, as F156 requires
#   split  the same held-out body for the insect, the same clips for the B1
#
# **Read against `beh12_lag3_nohinge` measured at lag 3, not against any lag-1 number.**
#
#   null/real stays near 1.03   the body term is innocent and the insensitivity is purely
#                               encoder-level -- which **strengthens** the paper, because our own
#                               objective is not the cause
#   null/real rises             the body term was stripping action detail from `z`. F159 still
#                               stands, but our pipeline compounded the encoder limit, and what we
#                               claim about `z` specifically has to change
#
# Either outcome is worth having. We currently cannot answer "did your coordinate objective cause
# this?" with a measurement.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
RUN=wm/runs/beh12_lag3_nobody

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "=== lambda_body 0 control  $(date '+%F %T')"
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_lag3_nobody \
    --sources hexapod=data/beh12_c10f10t10_flat b1=data/beh12_b1_flat \
    --lambda_body 0.0 --body_dim 3 --body_channels 0 1 2 \
    --frame_stride 3 --action_chunk 0 \
    --lambda_recon 1.0 --lambda_hinge 0.0 --lambda_readout 0.0 \
    --epochs 10
fi

echo
echo "=== null/real, the same path as the baseline, at this checkpoint's own stride"
$PY -u scripts/diagnostics/action_necessity.py --ckpt "$RUN/best.pt" \
  --data data/beh12_c08f09t09_flat --embodiment hexapod --lags 1 2 3 5
$PY -u scripts/diagnostics/action_necessity.py --ckpt "$RUN/best.pt" \
  --data data/beh12_b1_flat --embodiment b1 --lags 1 2 3 5

echo
echo "=== and the baseline again, so both tables are in one log and cannot be mismatched"
if [ -f wm/runs/beh12_lag3_nohinge/best.pt ]; then
  $PY -u scripts/diagnostics/action_necessity.py --ckpt wm/runs/beh12_lag3_nohinge/best.pt \
    --data data/beh12_c08f09t09_flat --embodiment hexapod --lags 3
  $PY -u scripts/diagnostics/action_necessity.py --ckpt wm/runs/beh12_lag3_nohinge/best.pt \
    --data data/beh12_b1_flat --embodiment b1 --lags 3
fi

echo
echo "=== done  $(date '+%F %T')   send back this whole log"
