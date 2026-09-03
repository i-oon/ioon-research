#!/usr/bin/env bash
#
# Pretrain with **both robots** and a three-channel shared body target, then rebuild stages 1-3.
#
#   bash scripts/com7_pretrain_body3.sh          # on com7, under tmux
#
# **Why both robots, which is the change F129 forced.** The live pretrain `beh12_hexonly` is
# hexapod-only, and neither adaptation stage touches the motion decoder -- so the shared body head
# has never seen a B1 latent. Measured: it reads forward speed on the hexapod at correlation +0.99
# and compression 1.0x, and returns the dataset mean for every B1 behaviour (F129). Widening that
# head while the target robot is still absent from pretraining would produce a three-channel head
# that has still never seen a B1, and a negative result would say nothing.
#
# **Why three channels.** `body_channels 0 1 2` is forward, lateral and yaw -- the full body pose
# delta, dimensionless, observed from outside and needing no kinematic model. One channel is not
# enough to plan with: F128 scored candidates on forward speed alone and could only separate
# "sideways or not".
#
# **The pass bar, fixed before the run so it cannot be moved afterwards.**
#   * per-channel calibration on **both** robots, predicted against measured, compression <~1.5x
#   * `roll/goal` above 28% on **mismatched** cross-embodiment goals
# Scored against the demonstration, a rule passes without reading its goal at all (F123, F127), so
# the mismatched column is the only admissible number.
#
# **If forward calibrates and yaw does not, stop and report.** That is F83's channel competition --
# adding yaw cost forward 68% and bought yaw at +0.37 +/- 0.27 -- reproduced on corrected data, and
# it makes the bottleneck the pretraining objective rather than the width of the head. Do not tune.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
. scripts/b1_stage3_clips.sh
RUN=wm/runs/beh12_hex-b1_body3

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

echo
echo "=== pretrain, both embodiments, 3-channel shared target  $(date '+%F %T')"
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_hex-b1_body3 \
    --sources hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2
fi

echo
echo "=== calibration A: straight off the pretrain, both robots  $(date '+%F %T')"
# **Measured here and again after stage 1, because adaptation moves the latent the head reads.**
# On the current hexapod-only pretrain the head reads the B1 at corr +0.76 / 2.2x before adaptation
# and +0.23 / 3.2x after it -- `wm/adapt.py` says why in its own docstring, "stage 1 moved what `z`
# means", and no stage adapts the head to follow. If that repeats here, the pass bar can be met at
# A and lost at B, and only running both shows it.
$PY scripts/diagnostics/shared_body_target/body_head_calibration.py --ckpt "$RUN/best.pt" \
  --data hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat

echo
echo "=== stage 1: adapt ITM and forward model to the B1  $(date '+%F %T')"
[ -f "$RUN/adapted_b1.pt" ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.adapt \
  --ckpt "$RUN/best.pt" --data data/allocentric/beh12_b1_flat --embodiment b1 \
  --clips 9 --stratify --train_clips $CLIPS --out "$RUN/adapted_b1.pt"

echo
echo "=== stage 2: fit the action projector against the adapted ITM  $(date '+%F %T')"
[ -f "$RUN/projector_b1_adapted.pt" ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
  -m wm.fit_projector --ckpt "$RUN/adapted_b1.pt" \
  --hex_dir data/allocentric/beh12_c10f10t10_flat --b1_dir data/allocentric/beh12_b1_flat --exclude $HOLDOUT \
  --cache results/wm/cache/beh12_embeddings_body3.pt --out "$RUN/projector_b1_adapted.pt"

echo
echo "=== stage 3: contrastive, one seed to start  $(date '+%F %T')"
[ -f "$RUN/stage3_b1_nce_s0.pt" ] || PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u \
  -m wm.adapt3 --ckpt "$RUN/adapted_b1.pt" --projector "$RUN/projector_b1_adapted.pt" \
  --data data/allocentric/beh12_b1_flat --embodiment b1 --train_clips $CLIPS \
  --steps 15000 --lambda_nce 1.0 --batch 8 --seed 0 \
  --cache results/wm/cache/b1.pt --out "$RUN/stage3_b1_nce_s0.pt"

echo
echo "=== calibration B: after stage 3, on the checkpoint the planner would use  $(date '+%F %T')"
$PY scripts/diagnostics/shared_body_target/body_head_calibration.py --ckpt "$RUN/stage3_b1_nce_s0.pt" \
  --data hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat

echo
echo "=== done  $(date '+%F %T')"
echo "send back $RUN/{best.pt,config.yaml,adapted_b1.pt,projector_b1_adapted.pt,stage3_b1_nce_s0.pt}"
echo "then here:  calibration on both robots first, scoring only after it passes"

# ---------------------------------------------------------------------------------------
# ADDENDUM 2026-08-30, after F131.
#
# The single-channel coordinate now works: fitted on the latents the planner actually supplies,
# selection follows the goal at 86% same-robot (7% on the demonstration, below chance) and clears
# chance cross-embodiment at 35-38% against 28%. **So this run is buying the extra channels, not
# the basic capability**, and its result should be read against those numbers rather than against
# the broken ones F128 reported.
#
# After the stages finish, fit the head on the latents it will be shown and score with the control:
#
#   .venv/bin/python3 -m wm.fit_body_head --ckpt $RUN/stage3_b1_nce_s0.pt \
#       --data data/allocentric/beh12_b1_flat --latent projector \
#       --also hexapod=data/allocentric/beh12_c10f10t10_flat \
#       --out $RUN/stage3_b1_nce_s0_bodyfit_proj.pt
#
#   .venv/bin/python3 scripts/diagnostics/planning/score_by_body_motion.py \
#       --ckpt $RUN/stage3_b1_nce_s0_bodyfit_proj.pt --projector $RUN/stage3_b1_nce_s0.pt \
#       --data data/allocentric/beh12_b1_flat --goal_dir data/allocentric/beh12_c08f09t09_flat --goal_embodiment hexapod
#
# **Only the mismatched-versus-goal column decides, and the bar is now 42%, not 38%** (F132). That
# is what forward speed alone reaches in the honest condition -- frames and rollout on both sides,
# nothing from a recorded trajectory -- at horizon 10, rising from 35% at horizon 3.
#
# **The realistic outcome, written down before the run so it cannot be reinterpreted afterwards.**
# F83 measured the channels competing: adding yaw cost forward 68% and bought yaw at +0.37 +/- 0.27.
# If that repeats, forward stays near 42% and yaw comes in low. **That is still a result worth
# having** -- "the shared coordinate extends to forward speed robustly, and yaw and lateral need a
# pretraining objective that does not trade channels against each other" is bounded, measured, and
# names its own next problem. It is stronger than an unbounded claim, and it is not a failure of
# this run.
