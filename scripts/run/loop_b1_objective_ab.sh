#!/usr/bin/env bash
#
# Close the loop on the B1 under both stage-3 objectives, on the same six goal clips.
#
#   bash scripts/loop_b1_objective_ab.sh
#
# **The A/B is the point, so nothing but `--ckpt` differs between the two arms.** F119 measured the
# contrastive arm selecting at 54.8% against MSE's 21.6% offline; this asks whether that survives
# into physics, which F100 says it need not -- forward selection has cleared chance before under a
# model whose `/mean-z` was 0.977, i.e. one that was not reading the action at all.
#
# **Goals are the `...3` clips**: the twelve validation clips, never in the candidate library and
# never in stage 3's training set, so the loop is asked about clips nothing in this chain has seen.
#
# **Spread comes from goals, not repeats** (F105): rerunning one B1 configuration returns the
# identical number, so six goals under each arm is the only spread available here.
#
# `--warm_start` stays at its default 10. It replays the goal clip's own commands, which is an
# intervention (F110) -- held identical across both arms rather than removed, so the comparison is
# clean even though each arm's absolute number carries it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
# **`--commit` is not a free parameter and 1 is the worst value measured.** F103: holding the
# chosen behaviour three steps is the only setting that has ever cleared the speed criterion on
# this robot (2/3 against 0/3), because every switch interrupts the stride -- F102 priced that at
# half the turning and four fifths of the lateral travel. Runs are tagged with it so two commit
# settings cannot be compared without saying so.
COMMIT=${COMMIT:-3}
RUN=wm/runs/beh12_hexonly

# two per family, and both signs of the sideways pair, because behaviour-family accuracy cannot
# see direction and a run that strafes the wrong way scores identically to one that does not (F109)
GOALS="b1_ep3 b1_ep303 b1_ep1203 b1_ep1303 b1_ep2103 b1_ep2303"

for ARM in nce mse; do
  CK=$RUN/stage3_b1_${ARM}_s0.pt
  for G in $GOALS; do
    OUT=results/wm/closed_loop/b1_${ARM}_s0_c${COMMIT}_${G}
    [ -d "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== $ARM $G"
    $PY -u sim/control/close_loop_b1_physics.py \
      --ckpt "$CK" --projector "$CK" \
      --demo data/allocentric/beh12_b1_flat/${G}.npz --commit "$COMMIT" --out "$OUT"
  done
done

echo
echo "score with:"
echo "  $PY scripts/diagnostics/planning/score_closed_loop.py results/wm/closed_loop/b1_{nce,mse}_s0_*/*.npz --demo_dir data/allocentric/beh12_b1_flat"
