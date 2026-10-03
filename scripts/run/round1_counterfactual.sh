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
#   GPU: CUDA_VISIBLE_DEVICES=0 bash ...
#   a longer arm B is a NEW run (new --name, fresh cosine schedule); --resume only continues an unfinished run
set -uo pipefail
cd "$(dirname "$0")/../.."
ARM=${1:?usage: round1_counterfactual.sh A|B}
CW=data/counterfactual_walks
PY=.venv/bin/python3
$PY tests/test_froude_labels.py || { echo "label tests FAIL: wrong code here"; exit 1; }
need=("$CW/c10_clips_train" "$CW/c10_clips_val" "$CW/b1_clips_train" "$CW/b1_clips_val")
[ "$ARM" = B ] && need+=("$CW/c10_branches_train" "$CW/b1_branches_train")
for p in "${need[@]}"; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
COMMON="--lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0
 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --lambda_sim 0 --seed 0
 --resume auto --val_sources hexapod=$CW/c10_clips_val b1=$CW/b1_clips_val"
mkdir -p results/wm/logs
case $ARM in
  A) NAME=round1_clips_s0
     ARGS="--sources hexapod=$CW/c10_clips_train b1=$CW/b1_clips_train --epochs 45 --checkpoint_every 3" ;;
  B) NAME=round1_branches_s0
     ARGS="--sources hexapod=$CW/c10_clips_train hexapod=$CW/c10_branches_train b1=$CW/b1_clips_train
           b1=$CW/b1_branches_train --epochs 3 --checkpoint_every 1" ;;
  *) echo "arm must be A or B"; exit 1 ;;
esac
setsid nohup $PY -m wm.train $COMMON $ARGS --name $NAME >> results/wm/logs/$NAME.log 2>&1 &
echo "started $NAME (pid $!); log results/wm/logs/$NAME.log"
echo "check after a few minutes: grep -E '^train|^epoch' results/wm/logs/$NAME.log | tail -3"
