#!/usr/bin/env bash
# Round 1: does training on counterfactual branches (same state, different command) fix the rollout?
# Joint c10 hexapod + B1 pretraining on data/counterfactual_walks (doc/DATA.md), recipe = the current pipeline
# (Froude head shapes z, no joint-command decoder, no similarity loss), seed 0. Two arms, equal gradient-step budget:
#   A  clips      c10 + B1 clips_train only                          45 epochs x  672 steps = 30,240 steps
#   B  branches   the same clips + c10 / B1 branches_train           3 epochs x 10,176 steps = 30,528 steps
# (batch 8, balanced embodiments; B's epoch = (2,688 + 38,016) pairs per body / 8 x 2 bodies.)
# Both arms select best.pt on the SAME validation set (clips_val of both bodies), so selection is comparable.
# Evaluate each with scripts/run/eval_suite.sh NAME joint wm/runs/NAME/best.pt.
#   bash scripts/run/round1_counterfactual.sh A        # e.g. on the server (needs only the clips dirs)
#   bash scripts/run/round1_counterfactual.sh B        # locally (needs the branch dirs too)
#   bash scripts/run/round1_counterfactual.sh H        # hexapod only: c10 clips + branches, 6 epochs x 5,088 steps
#                                                      # = 30,528 steps (same budget); then eval_suite.sh NAME hexonly
#   GPU: CUDA_VISIBLE_DEVICES=0 bash ...
#   second seed: SEED=1 bash ... B   (run name round1_branches_s1)
#   random-room data: PFX=rr_ RUN_TAG=_rr bash ... H   (run round1_hexonly_s0_rr on data/counterfactual_walks/rr_*)
#   bash scripts/run/round1_counterfactual.sh F        # round 2: arm B + --body_sees_frame True (Froude head reads
#                                                      # head(e_t, z), LAC-WM's MD(x_t, z)); run round2_framehead_s$SEED;
#                                                      # then eval_suite.sh NAME joint. Same data dirs as B.
#   EXTRA="--body_frame_on_branches_only True" ... F  # extra wm.train flags appended (e.g. frame-head loss on branch pairs only)
#   a longer arm B is a NEW run (new --name, fresh cosine schedule); --resume only continues an unfinished run
set -uo pipefail
cd "$(dirname "$0")/../.."
ARM=${1:?usage: round1_counterfactual.sh A|B|H|F}
SEED=${SEED:-0}
PFX=${PFX:-}              # data prefix: "" = current renders; rr_ = random room size + start (rendering only)
RUN_TAG=${RUN_TAG:-}      # appended to the run name, e.g. _rr
CW=data/counterfactual_walks
PY=.venv/bin/python3
$PY tests/test_froude_labels.py || { echo "label tests FAIL: wrong code here"; exit 1; }
need=("$CW/${PFX}c10_clips_train" "$CW/${PFX}c10_clips_val" "$CW/${PFX}b1_clips_train" "$CW/${PFX}b1_clips_val")
[ "$ARM" = B -o "$ARM" = F ] && need+=("$CW/${PFX}c10_branches_train" "$CW/${PFX}b1_branches_train")
[ "$ARM" = H ] && need=("$CW/${PFX}c10_clips_train" "$CW/${PFX}c10_clips_val" "$CW/${PFX}c10_branches_train")
for p in "${need[@]}"; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
COMMON="--lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0
 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --lambda_sim 0 --seed $SEED
 --resume auto"
VAL="--val_sources hexapod=$CW/${PFX}c10_clips_val b1=$CW/${PFX}b1_clips_val"
mkdir -p results/wm/logs
case $ARM in
  A) NAME=round1_clips_s$SEED$RUN_TAG
     ARGS="--sources hexapod=$CW/${PFX}c10_clips_train b1=$CW/${PFX}b1_clips_train --epochs 45 --checkpoint_every 3" ;;
  B) NAME=round1_branches_s$SEED$RUN_TAG
     ARGS="--sources hexapod=$CW/${PFX}c10_clips_train hexapod=$CW/${PFX}c10_branches_train b1=$CW/${PFX}b1_clips_train
           b1=$CW/${PFX}b1_branches_train --epochs 3 --checkpoint_every 1" ;;
  F) NAME=round2_framehead_s$SEED$RUN_TAG        # = arm B, Froude head conditioned on the current frame
     ARGS="--sources hexapod=$CW/${PFX}c10_clips_train hexapod=$CW/${PFX}c10_branches_train b1=$CW/${PFX}b1_clips_train
           b1=$CW/${PFX}b1_branches_train --epochs 3 --checkpoint_every 1 --body_sees_frame True" ;;
  H) NAME=round1_hexonly_s$SEED$RUN_TAG          # hexapod only (B1 never seen); B1 is adapted afterwards
     ARGS="--sources hexapod=$CW/${PFX}c10_clips_train hexapod=$CW/${PFX}c10_branches_train --epochs 6 --checkpoint_every 1"
     VAL="--val_sources hexapod=$CW/${PFX}c10_clips_val" ;;
  *) echo "arm must be A, B, H or F"; exit 1 ;;
esac
setsid nohup $PY -m wm.train $COMMON $VAL $ARGS ${EXTRA:-} --name $NAME >> results/wm/logs/$NAME.log 2>&1 &
TRAIN_PID=$!
echo "started $NAME (pid $TRAIN_PID); log results/wm/logs/$NAME.log"
# GPU temperature guard on the training process itself (pause > 83 C, resume < 75 C); NO_GUARD=1 to skip
[ -z "${NO_GUARD:-}" ] && setsid nohup bash scripts/tools/gpu_guard.sh $TRAIN_PID > results/wm/logs/gpu_guard_$NAME.log 2>&1 < /dev/null &
echo "gpu_guard attached to $TRAIN_PID (results/wm/logs/gpu_guard_$NAME.log)"
echo "check after a few minutes: grep -E '^train|^epoch' results/wm/logs/$NAME.log | tail -3"
