#!/usr/bin/env bash
#
# Cross-embodiment closed loop: the goal frames are a **stick insect's**, the robot driven is the
# **quadruped**, under both stage-3 objectives.
#
#   bash scripts/loop_b1_hexgoal_ab.sh
#
# **This is the run that tests the direction fix.** In the same-robot loop the goal and the
# candidate library are both B1, so a reversed turn is reversed on both sides and cancels. Here the
# goal is an insect turning one way and the candidates are B1 clips turning another -- which is why
# every cross-embodiment turning result before the F117 re-collection compared a left turn against a
# right turn and could not have succeeded. All three sets now turn positive (F118), so the question
# is finally askable.
#
# **Goals come from `c08f09t09`, the held-out insect body** -- never pretrained on, never in any
# projector or adaptation set. Candidates stay B1 clips because only those are executable, so only
# the goal crosses embodiments.
#
# **`--demo` is held fixed at one forward B1 clip and is not varied**, because it supplies the start
# state and the ten warm-start commands: varying it with the goal would change two things at once
# (F109, F110). It is a validation clip, so nothing in the chain trained on it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
# **`--commit` is not a free parameter and 1 is the worst value measured.** F103: holding the
# chosen behaviour three steps is the only setting that has ever cleared the speed criterion on
# this robot (2/3 against 0/3), because every switch interrupts the stride -- F102 priced that at
# half the turning and four fifths of the lateral travel. Runs are tagged with it so two commit
# settings cannot be compared without saying so.
COMMIT=${COMMIT:-3}
# **`--warm_start` is an intervention, not a convenience.** F109 changed only the warm-start clip
# and every planned yaw moved with it, while varying the goal across a thirty-fold range of
# commanded yaw moved it almost nothing -- and the turning family numbers crossed the chance line
# in both directions, which is why F107's turn result was withdrawn. F109 also showed the ten steps
# are not load-bearing: at `--warm_start 0` all four goals survived 65 of 65. So `WARM=0` is the
# setting to trust and `WARM=10` exists only to reproduce the earlier runs.
WARM=${WARM:-10}
# **`CENTER=1` translates the insect goal into the B1's mean appearance before scoring.** F123
# measured that offline: raw, the planner selects the exact condition *below* chance (4.2% against
# 8.3%); shifted, it reaches 23.2% and clears the same-robot baseline on behaviour too. Off by
# default so the earlier runs stay reproducible; the setting is in every output name.
CENTER=${CENTER:-0}
CFLAG=""; TAG=""
[ "$CENTER" = 1 ] && { CFLAG="--center_goal"; TAG="ctr"; }
RUN=wm/runs/beh12_hexonly
DEMO=data/allocentric/beh12_b1_flat/b1_ep3.npz

# two per family; the sideways pair is both signs, since behaviour-family accuracy cannot see
# direction and a run that strafes the wrong way scores identically to one that does not (F109)
GOALS="hexapod_ep303 hexapod_ep3 hexapod_ep1203 hexapod_ep1303 hexapod_ep2103 hexapod_ep2303"

for ARM in nce mse; do
  CK=$RUN/stage3_b1_${ARM}_s0.pt
  for G in $GOALS; do
    OUT=results/wm/closed_loop/b1_hexgoal_${ARM}_s0_c${COMMIT}w${WARM}${TAG}_${G}
    [ -d "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== $ARM $G"
    $PY -u sim/control/close_loop_b1_physics.py \
      --ckpt "$CK" --projector "$CK" --demo "$DEMO" \
      --goal data/allocentric/beh12_c08f09t09_flat/${G}.npz --goal_embodiment hexapod \
      --commit "$COMMIT" --warm_start "$WARM" $CFLAG \
      --out "$OUT"
  done
done
