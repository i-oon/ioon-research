#!/usr/bin/env bash
#
# The ActSWM rebuild: pretrain from scratch with rollout separation and a frozen action readout.
#
#   bash scripts/com7_pretrain_actswm.sh          # on com7, under tmux, about five hours
#
# **Every setting here is backed by a measurement, and three of ActSWM's were deliberately not
# copied.**
#
#   margin 0.1, not 0.3   at 0.3 the term overshoots, switches itself off and collapses --
#                         separation read 0.019, 0.137, 0.496, 0.008 with its gradient dying to
#                         0.00006 (F151). At 0.1 it rises and holds on both bodies (F152)
#   K = 3, not 12         the rolled prediction crosses "worse than a frozen frame" by five steps
#                         on this architecture (F140, F150). Hinging separation past that trains
#                         on noise
#   H = 1, not 32         `wm/train.py` conditions the forward model on a single frame. 32 is a
#                         different architecture, not a hyperparameter
#   lambda_sig unused     SigReg is LeWM-specific; this is V-JEPA2 and the term is not guessed in
#
#   lambda_recon 3.0      raised, not ActSWM's 1.0: with the prediction weight tripled the short
#                         run lost half as much accuracy (F152, run B)
#
# **The null is `ITM(e_t, e_t)`** -- the latent of "nothing happened". F148's standing stance is an
# *action* and pretraining has no projector to map it into `z`; a hinge on `proj(stance)` puts zero
# gradient into `z`, measured (F151). The stance null still applies to anything scored through the
# projector. **Never compare `/mean-z` across the two stages**: the same checkpoint reads about 1.0
# against one and 0.86-0.96 against the other.
#
# **Read the result per body, never pooled.** The two robots have different horizon shapes: the
# insect's sensitivity improves with horizon and the B1's is flat (F150), and the B1 is the one that
# lost accuracy fastest in every short run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
RUN=wm/runs/beh12_actswm

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "=== pretrain with the ActSWM objective  $(date '+%F %T')"
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_actswm \
    --sources hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2 \
    --lambda_recon 3.0 --lambda_hinge 0.5 --lambda_readout 1.0 \
    --hinge_margin 0.1 --hinge_K 3 --readout_hidden 512
fi

echo
echo "=== per-body readout: sensitivity and prediction, on a body neither run trained on"
for SPEC in "hexapod data/allocentric/beh12_c08f09t09_flat results/wm/cache/fid_hexapod.pt" \
            "b1 data/allocentric/beh12_b1_flat results/wm/cache/b1_body3.pt"; do
  set -- $SPEC
  echo "--- $1"
  $PY -u scripts/diagnostics/rollout_fidelity.py --ckpt "$RUN/best.pt" \
    --data "$2" --embodiment "$1" --cache "$3" --mean_z --family_mean --horizons 1 2 3 5
done

echo
echo "=== the shared coordinate, both robots, straight off the pretrain"
$PY -u scripts/diagnostics/body_head_calibration.py --ckpt "$RUN/best.pt" \
  --data hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat

echo
echo "=== gradient balance against the F149 baseline"
for SPEC in "hexapod data/allocentric/beh12_c10f10t10_flat results/wm/cache/hex_c10.pt" \
            "b1 data/allocentric/beh12_b1_flat results/wm/cache/b1_body3.pt"; do
  set -- $SPEC
  $PY -u scripts/diagnostics/loss_gradient_balance.py --ckpt "$RUN/best.pt" \
    --dir "$2" --embodiment "$1" --cache "$3" --batch 24
done

echo
echo "=== done  $(date '+%F %T')   send back $RUN/ and this whole log"
