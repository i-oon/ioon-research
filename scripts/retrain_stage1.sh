#!/usr/bin/env bash
#
# Retrain the five Stage 1 runs the slide deck cites.
#
# Their checkpoints were lost when wm/runs was deleted; the configs survived in
# results/wm/RUNS.md and every dataset is still on disk, so all five are reproducible.
#
#   bash scripts/retrain_stage1.sh              run all five, in order
#   bash scripts/retrain_stage1.sh tib_cross    run one by name
#
# Data comes from three directories built by scripts/build_stage1_dirs.py, which links only
# clips where the body walked (signed forward travel >= 0.30 m, lateral drift < 0.20 m). Run
# that script first; the preflight below refuses to start otherwise.
#
# ---------------------------------------------------------------------------------------
# DEVIATION 1: action_lag 1
#
#   m3d_cross and m3d_bracketed were trained before `action_lag` existed, so they ran at the
#   legacy value 0 -- the setting where the collector's frame ordering leaks the answer into the
#   decoder's own input (FINDINGS.md F29). These are retrained at action_lag 1, the corrected
#   semantics. Reproducing a known bug to keep a slide unchanged is the worse trade, and F31
#   measured the difference as small: deleting the transition from z costs 1.19x at lag 0 and
#   1.36x at lag 1.
#
#   tib_cross and bracket_cross were already action_lag 1, so their settings reproduce exactly.
#   tib_ctrl was never run -- slide 3 names it as tib_cross's control and no such checkpoint
#   exists. It is included here so the claim becomes true.
#
# DEVIATION 2: only bodies and clips that walk
#
#   The originals were not clean. Auditing every clip they trained on:
#
#     m3d_*          c10f10t06 and c06f10t06 fail 30/30 clips each -- 40% of the training data
#                    was a body veering 0.36-0.43 m off course (FINDINGS.md F42)
#     tib_*, bracket held-out c10f10t06 fails 30/30, so slides 8 and 9 measured extrapolation
#                    onto a body that does not walk straight
#
#   Replaced:
#     m3d_*    drop the two veering bodies, add c06f06t06 -- sound, 30/30 usable, never used
#              before. Four training bodies, 111 clips. c08f09t09 stays inside their convex
#              hull on all three axes and off every pairwise line, so it remains a composition
#              test rather than an interpolation.
#     tib_*    hold out c10f10t08 instead: femur/tibia 1.04, dead zone 11.8 mm, 20/20 usable.
#              Still a femur/tibia extrapolation, since every training body sits at 0.83.
#     bracket  same held-out body and the same 20 clips of it, so the pair is scored on
#              identical frames.
#
#   WHAT THIS COSTS THE m3d CLAIM. Inside ik_walk_8body every body with femur != tibia is one
#   of the broken ones, so the clean training set is entirely femur == tibia, and so is
#   c08f09t09. m3d therefore tests composition along coxa and overall scale but no longer
#   probes femur/tibia decoupling at all. That test now rests solely on tib_* and bracket_*,
#   which draw from ik_walk_decoupled.
#
#   VOLUME MATCHING. Slide 9's claim is that coverage helped *at matched data volume* -- without
#   that, a better score is just more data. It takes two directories, because the same body
#   cannot hold 24 clips for one run and 16 for the other:
#
#     data/ik_walk_cov_narrow   4 bodies x 24 clips = 96     tib_cross, tib_ctrl
#     data/ik_walk_cov_wide     6 bodies x 16 clips = 96     bracket_cross
#
#   96 rather than 120 because c06f10t10 has only 25 usable clips, and matching has to level
#   down to the scarcest body. The wide run's four shared bodies are a subset of the narrow
#   run's own selection, not an independent draw, so neither can land a luckier sample.
#
#   Cost: every number on slides 5-9 has to be re-measured. They had to be anyway, because of
#   action_lag. Gain: no figure in the deck rests on a robot that veers.
# ---------------------------------------------------------------------------------------
#
# Runs sequentially: one GPU, and each pair is only meaningful if both halves finish.
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python3
LOGS=results/wm
mkdir -p "$LOGS"

# ~3.7 GB for each 50-epoch run, ~2.2 GB for each 10-epoch one, including the 1.1 GB last.pt
# that carries optimiser state for --resume. Refuse to start rather than fill the disk at epoch 40.
NEEDED_GB=16
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$NEEDED_GB" ]; then
  echo "only ${free_gb} GB free, need about ${NEEDED_GB} GB for all five runs." >&2
  echo "clear space or run them one at a time by name." >&2
  exit 1
fi

# Fail here rather than after two hours of m3d training, and catch a checkout where the
# symlink directories were never built or their targets are missing.
preflight() {
  local dir=$1 expected=$2 found dangling
  # `|| true` on both: a missing directory makes find exit nonzero, and under `set -e` with
  # pipefail the failing assignment would kill the script before it could say why
  found=$(find "data/$dir" -name '*.npz' 2>/dev/null | wc -l) || true
  if [ "$found" -ne "$expected" ]; then
    echo "data/$dir holds $found clips, expected $expected." >&2
    echo "run: $PY scripts/build_stage1_dirs.py" >&2
    exit 1
  fi
  # a dangling symlink still matches -name, so the count above cannot catch it
  dangling=$(find "data/$dir" -name '*.npz' -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l) || true
  if [ "$dangling" -ne 0 ]; then
    echo "data/$dir has $dangling symlinks whose targets are missing." >&2
    echo "the source datasets are not where scripts/build_stage1_dirs.py left them." >&2
    exit 1
  fi
}
preflight ik_walk_m3d_clean 140    # 111 training + 29 held out
preflight ik_walk_cov_narrow 116   #  96 training + 20 held out
preflight ik_walk_cov_wide 116     #  96 training + 20 held out

common=(--action_lag 1 --seed 0)

run_m3d_cross() {
  $PY -m wm.train --data_dir data/ik_walk_m3d_clean \
    --train_morphs c10f10t10 c06f10t10 c10f06t06 c06f06t06 \
    --heldout_morph c08f09t09 \
    --lambda_cross 0.5 "${common[@]}" \
    --epochs 50 --checkpoint_every 10 \
    --name m3d_cross 2>&1 | tee "$LOGS/m3d_cross.log"
}

# identical to m3d_cross but for lambda_cross: this is the matched control slide 5 rests on
run_m3d_bracketed() {
  $PY -m wm.train --data_dir data/ik_walk_m3d_clean \
    --train_morphs c10f10t10 c06f10t10 c10f06t06 c06f06t06 \
    --heldout_morph c08f09t09 \
    --lambda_cross 0.0 "${common[@]}" \
    --epochs 50 --checkpoint_every 10 \
    --name m3d_bracketed 2>&1 | tee "$LOGS/m3d_bracketed.log"
}

# four bodies all at femur/tibia 0.83, held out one where they differ: slide 8's limit
run_tib_cross() {
  $PY -m wm.train --data_dir data/ik_walk_cov_narrow \
    --train_morphs c10f10t10 c06f10t10 c10f06t06 c08f09t09 \
    --heldout_morph c10f10t08 \
    --lambda_cross 0.5 "${common[@]}" \
    --epochs 10 --checkpoint_every 5 \
    --name tib_cross 2>&1 | tee "$LOGS/tib_cross.log"
}

# the control slide 3 names but which was never run
run_tib_ctrl() {
  $PY -m wm.train --data_dir data/ik_walk_cov_narrow \
    --train_morphs c10f10t10 c06f10t10 c10f06t06 c08f09t09 \
    --heldout_morph c10f10t08 \
    --lambda_cross 0.0 "${common[@]}" \
    --epochs 10 --checkpoint_every 5 \
    --name tib_ctrl 2>&1 | tee "$LOGS/tib_ctrl.log"
}

# same held-out body and the same 20 clips of it as tib_cross, two more bodies where femur and
# tibia differ: slide 9's coverage experiment is tib_cross against this, so the pair has to match
run_bracket_cross() {
  $PY -m wm.train --data_dir data/ik_walk_cov_wide \
    --train_morphs c10f10t10 c06f10t10 c10f06t06 c08f09t09 c10f09t07 c10f08t06 \
    --heldout_morph c10f10t08 \
    --lambda_cross 0.5 "${common[@]}" \
    --epochs 10 --checkpoint_every 5 \
    --name bracket_cross 2>&1 | tee "$LOGS/bracket_cross.log"
}

ORDER=(m3d_cross m3d_bracketed tib_cross tib_ctrl bracket_cross)
targets=("${@:-${ORDER[@]}}")

for name in "${targets[@]}"; do
  # best.pt alone is not proof of completion -- it is rewritten at every improvement, so an
  # interrupted run leaves one behind and a skip on that basis would ship a half-trained model
  if [ -f "wm/runs/$name/COMPLETE" ]; then
    echo "=== $name already finished, skipping (delete wm/runs/$name to force a retrain)"
    continue
  fi
  if [ -e "wm/runs/$name" ]; then
    echo "=== $name exists but did not finish; delete wm/runs/$name, or resume it with" >&2
    echo "    $PY -m wm.train --resume auto --name $name ..." >&2
    exit 1
  fi
  echo "=== $name  $(date '+%F %T')"
  "run_$name"
  touch "wm/runs/$name/COMPLETE"
  echo "=== $name done  $(date '+%F %T')"
done

echo
echo "all requested runs finished. Copy back only what the measurements need:"
for name in "${targets[@]}"; do
  echo "  scp <host>:~/ioon-research/wm/runs/$name/{best.pt,config.yaml} wm/runs/$name/"
done
