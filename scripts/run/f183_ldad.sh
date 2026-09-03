#!/usr/bin/env bash
#
# F183: Delta-JEPA's LDAD on egocentric locomotion, the method never tried here.
#
#   bash scripts/f183_ldad.sh baseline     # here, no training -- what dz already carries
#   bash scripts/f183_ldad.sh train        # com7, two short arms, lambda 10 and 50
#   bash scripts/f183_ldad.sh measure      # the deciding split, on each arm
#
# **Two questions, and only the second decides anything.** Delta-JEPA trains the world model so the
# difference of consecutive state latents carries the action, and reports that this is what stops
# action-insensitive collapse. It explains F168 exactly: from the endpoints `[z_t, z_t+1]` a decoder
# reads the action off cues in `z_t+1` without modelling the transition; a difference cannot be read
# that way. **It was evaluated on manipulation and navigation, neither of which is periodic.**
#
#   GENERAL   does dz carry the action at all, once trained to? Expected to work -- F158 found the
#             PASSIVE residual was noise, and that is a fact about what is there, not about what an
#             objective would put there.
#   FINE      does it carry the action WITHIN one behaviour, at the 2.5% scale that F145, F179 and
#             F182 could not separate? **This is the deciding one and it is pre-registered as such.**
#             Coarse-only is the same wall relocated, not a solution.
#
# **`--lambda_ldad 50` is their best and 10 is the low end of their useful range.** The term is meant
# to dominate; a small weight reproduces the collapse it exists to prevent, so a sweep that only
# tries small values would fail for the wrong reason.
#
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
# **Keep the terminal output.** The first run of this script printed to stdout alone and the log was
# lost with the scrollback; `wm.train` writes its scalars to TensorBoard so nothing was
# unrecoverable, but a diagnostic's output is not written anywhere and would have been gone.
LOGDIR=results/f183
mkdir -p "$LOGDIR"
run() { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
HEX=data/egocentric/beh12_c10f10t10_ego_flat
HELD=data/egocentric/beh12_c08f09t09_ego_flat
B1=data/egocentric/beh12_b1_ego_flat

case "${1:-}" in

baseline)
  LOG="$LOGDIR/baseline.log"; : > "$LOG"
  # the untrained numbers the trained arms must beat; no GPU-hours, no retrain
  run $PY scripts/diagnostics/objective_experiments/delta_action_decoding.py --ckpt wm/runs/beh12_ego/teacher_ego.pt \
      --data "$HELD" --embodiment hexapod --cache results/wm/cache/ego_hex.pt
  ;;

train)
  # **Short, and everything except `--lambda_ldad` matches `beh12_ego` exactly**, so the arms are
  # comparable with F172-F179 and with each other rather than with a differently-configured run.
  for LAM in 10 50; do
    [ -f "wm/runs/beh12_ego_ldad$LAM/best.pt" ] && { echo "skip lambda $LAM"; continue; }
    LOG="$LOGDIR/train_lambda$LAM.log"; : > "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run $PY -u -m wm.train \
      --name "beh12_ego_ldad$LAM" \
      --sources hexapod="$HEX" b1="$B1" \
      --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2 \
      --lambda_ldad "$LAM" --epochs 15
  done
  ;;

measure)
  for LAM in 10 50; do
    LOG="$LOGDIR/measure_lambda$LAM.log"; : > "$LOG"
    echo "############ lambda_ldad $LAM ############" | tee -a "$LOG"
    # **A projector per arm, fitted against that arm.** A stage-1 pretrain carries none, and
    # borrowing one from another checkpoint compares two different latent spaces -- the F160 trap.
    # Without it the response ratio, one of the three numbers this gate reports, is missing.
    PROJ="wm/runs/beh12_ego_ldad$LAM/projector.pt"
    [ -f "$PROJ" ] || run $PY -m wm.fit_projector --ckpt "wm/runs/beh12_ego_ldad$LAM/best.pt" \
        --hex_dir "$HEX" --b1_dir "$B1" --cache "results/wm/cache/ldad$LAM.pt" --out "$PROJ"
    run $PY scripts/diagnostics/objective_experiments/delta_action_decoding.py \
        --ckpt "wm/runs/beh12_ego_ldad$LAM/best.pt" --projector "$PROJ" \
        --data "$HELD" --embodiment hexapod --cache results/wm/cache/ego_hex.pt
  done
  echo
  echo "PASS  the within-cond column on the dz row clears the baseline clearly. The 2.5% wall is"
  echo "      gone, and teacher-student and an imagined actor become worth revisiting."
  echo "FAIL  dz reconstructs overall and not within a condition. LDAD fixed collapse and not the"
  echo "      periodicity-driven limit -- which is a locomotion-specific result Delta-JEPA's own"
  echo "      evidence could not have found, and is the contribution rather than a defeat."
  ;;

gate)
  # **Reconstruction is not ranking, and this session's whole result is that they are different.**
  # LDAD moved a reconstruction number and a response ratio; neither is F179's 47%. Two things get
  # checked here and the first can invalidate the second: `null/real`, because both responses shrank
  # fivefold and a better ratio with a collapsed GATE C is worse rather than better, and then the
  # ranking itself, on the simulator, against 47% and the physics separation.
  for LAM in 10 50; do
    LOG="$LOGDIR/gate_lambda$LAM.log"; : > "$LOG"
    RUN=wm/runs/beh12_ego_ldad$LAM
    [ -f "$RUN/teacher.pt" ] || run $PY -m wm.assemble_teacher --base "$RUN/best.pt" \
        --projector "$RUN/projector.pt" --out "$RUN/teacher.pt"
    echo "---- GATE C: does it still USE the action? baseline 1.16 insect ----" | tee -a "$LOG"
    run $PY scripts/diagnostics/objective_experiments/action_necessity.py --ckpt "$RUN/best.pt" \
        --data "$HELD" --embodiment hexapod --lags 1 2 3 5 \
        --cache results/wm/cache/ego_hex.pt
  done
  echo
  echo "If null/real held, run the ranking test -- it needs CoppeliaSim and is the deciding one:"
  echo "  $PY scripts/diagnostics/planning/teacher_label_quality.py \\"
  echo "     --teacher wm/runs/beh12_ego_ldad50/teacher.pt \\"
  echo "     --student wm/runs/students/insect_bc_ego.pt --data $HELD \\"
  echo "     --goal_clip $HELD/hexapod_ep100.npz --cache results/wm/cache/ego_hex.pt \\"
  echo "     --scene medauroidea_c08f09t09.ttt --ego --ego_seed 0 --repeat_control 4"
  echo "  against F179: teacher closer 47%, a coin 50%, separation 2.5%."
  ;;

*) echo "usage: $0 {baseline|train|measure|gate}"; exit 2 ;;
esac

echo
echo "log kept in $LOGDIR -- the numbers are also in wm/runs/*/summary as TensorBoard scalars."
