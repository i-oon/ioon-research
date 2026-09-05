#!/usr/bin/env bash
#
# Same StateHead recipe as com7_state_head.sh, retrained on 5x more clips of the SAME 12
# behaviours -- the data-sparsity test, not an architecture change.
#
#   bash scripts/run/com7_state_head_more.sh          # on com7, under tmux
#
# **Why this run exists.** F186/F187 (condition_confusion.py) found the state head ranks well
# above noise on well-separated candidates (68% win, 3.95x signal/floor) but is far from the
# ~100% a scorer with no error of its own should get there. The errors concentrate almost
# entirely on confusing ADJACENT SPEEDS (speed_c7.1 vs c8.8, etc.) -- a real but closely-spaced
# distinction that had only 4 clips to learn from. Two architecture-side fixes were tried and
# both landed flat on the action-lever metric (multi-step rollout supervision; wider z_tokens /
# more ftm_blocks, "deep injection") -- neither moved the number they were meant to move. This
# is the one lever not yet tested: more examples of the SAME behaviours, nothing new added to
# the action space.
#
# **What changed in the data, and what did not.** `beh12_c10f10t10_more_ego_flat`: 240 clips (20
# per condition, up from 4), same 12 conditions, same recipe, --separability-verified to match
# the reference on every family including a turn-sign bug caught and fixed mid-collection (a
# fresh CoppeliaSim session reproduced the SAME wrong sign as the first one -- not session noise,
# fixed with --spin_sign -1, verified against the reference to match). b1's data is UNCHANGED:
# its collector is MuJoCo, which is deterministic -- re-running a command gives the same clip
# back, not new data (see memory b1-mujoco-deterministic.md) -- so this test is hexapod-only by
# construction, and the batch sampler will oversample b1 more than usual as a result.
#
# **The read.** This is not a ranking-test pass/fail like the first run -- it is a before/after on
# F187's own diagnostic:
#   condition_confusion.py accuracy/cosine/magnitude-gap clearly improve on the enlarged set
#     -> the wall was data sparsity, and the fix is more collection, not more architecture
#   no meaningful change
#     -> data was not the limit either; the two flat architecture results plus this now leave
#        the frozen-encoder ceiling (cosine_ceiling.py's 0.690) as the remaining explanation
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python3
RUN=wm/runs/beh12_state_more
BASE=wm/runs/beh12_ego/teacher_ego.pt
HEX=data/egocentric/beh12_c10f10t10_more_ego_flat
B1=data/egocentric/beh12_b1_ego_flat

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

echo
echo "=== train: + StateHead, lambda_state, warm-started from teacher_ego.pt, 5x data  $(date '+%F %T')"
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_state_more \
    --sources hexapod="$HEX" b1="$B1" \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2 \
    --lambda_state 1.5 \
    --init_ckpt "$BASE" \
    --epochs 50
fi

echo
echo "=== fit a fresh projector against the retrained checkpoint  $(date '+%F %T')"
# egocentric dirs, explicit -- fit_projector's own defaults are allocentric and silently feed
# the ITM frames it never trained on if left unset (caught the hard way on the first state-head
# run: rollout gap read fine on the mismatched fit and had to be redone).
if [ -f "$RUN/projector_more.pt" ]; then echo "skip $RUN/projector_more.pt"; else
  $PY -u -m wm.fit_projector --ckpt "$RUN/best_state.pt" \
    --hex_dir "$HEX" --b1_dir "$B1" \
    --out "$RUN/projector_more.pt"
fi
echo "  read the rollout-gap ratio above against the first run's 0.339 (hexapod) / 0.206 (b1)"
echo "  before trusting anything downstream of it"

echo
echo "=== assemble one file load_teacher can read  $(date '+%F %T')"
if [ -f "$RUN/teacher_more.pt" ]; then echo "skip $RUN/teacher_more.pt"; else
  $PY -u -m wm.assemble_teacher \
    --base "$RUN/best_state.pt" \
    --projector "$RUN/projector_more.pt" \
    --out "$RUN/teacher_more.pt"
fi

echo
echo "=== F187's own diagnostic, before/after. CoppeliaSim must be up, GUI, one instance  $(date '+%F %T')"
echo "  medauroidea_c08f09t09.ttt on port 23000 -- see doc/SIM_GUIDE.md"
echo "  this does not launch CoppeliaSim itself: start it, confirm the connection, THEN continue"
read -p "CoppeliaSim ready on port 23000 with medauroidea_c08f09t09.ttt? [y/N] " ok
[ "$ok" = "y" ] || { echo "stopping -- start the sim first"; exit 1; }

$PY -u scripts/diagnostics/planning/condition_confusion.py \
  --teacher "$RUN/teacher_more.pt" \
  --train_data "$HEX"

echo
echo "=== the ranking test too, same protocol as the first run, for the direct comparison  $(date '+%F %T')"
$PY -u scripts/diagnostics/planning/rank_fine_three_ways.py \
  --teacher "$RUN/teacher_more.pt" \
  --states 40 \
  --goal_clip data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz \
  --repeat_control 4 --candidates conditions

echo
echo "=== done  $(date '+%F %T')"
echo "send back $RUN/{best_state.pt,teacher_more.pt,config.yaml} and both scripts' printed output"
echo "read F186/F187 (doc/FINDINGS.md) before reading these numbers -- this run only means"
echo "something in comparison to the 68% / 3.95x / adjacent-speed-confusion baseline they measured"
