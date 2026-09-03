#!/usr/bin/env bash
#
# F166. Meaningful phase-breaking: the robot leaves the rhythm **on purpose**.
#
# **F165's run of this script is void.** `--gait cpg` kept only `cmds.mean(0)` from the scheduled
# foot path and regenerated the stroke from `--cycles`, so eight of the twelve conditions carried no
# within-clip change while every log line said they did. `cpg_commands` now takes a per-frame pace
# and advances its phase by `cumsum(rate) - rate`, which reproduces `arange(frames)` exactly at rate
# 1, so nothing collected earlier moved. **The design below was correct and is unchanged; only the
# collector was broken.**
#
#   bash scripts/dataset/collect_intent.sh          # CoppeliaSim GUI, exactly ONE instance
#
# **Distinct from both earlier attempts, and that is the whole point.** F163's speed ramp retimed
# the whole foot path and preserved inter-leg phase, so the gap stayed at +0.024. F164's random
# `--cmd_noise` broke the phase and opened the gap to +0.198, but with **jitter**: a model trained
# on it learns to read noise, which is useless for control. Here the rhythm breaks because a
# *command* changed -- a stop, a speed break, a turn beginning -- so the thing the transition
# carries is an **intent** a controller could act on.
#
# **`--spin_schedule` is new and is what makes a turn an event.** With a constant `--spin` every
# frame of a turning clip says "turning clip" and no frame is the one where the turn begins; a probe
# can read the label off any frame and never has to predict the change. Onsets are placed at
# different fractions across conditions, and in both directions, so "later in the clip" and
# "turning" are not the same thing -- the reasoning F60 used for collecting ramps both ways.
#
# **This is also the exploratory set the parked bootstrap problem needs.** One collection, two uses.
#
# **The clean control already exists**: `data/allocentric/beh12_c10f10t10_sweepn00_flat`, from F164's sitting,
# which read 0.729 single-frame with a +0.102 gap. Sitting-to-sitting variation on that measurement
# is about 0.035 and 0.018 (F164), so a real effect has to clear that.
#
# **Read the `da` row of `intent_recoverability.py` and nothing else as the result.** A gap on the
# family label is a label, which we already have and already rejected. **Separability improving is
# not a result either** -- meaningful turns plausibly make behaviours easier to tell apart, and that
# is not what is being tested.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
MORPH=c10f10t10=medauroidea_c10f10t10.ttt
COMMON="--gait cpg --scale 0.65 --cam_dx -0.6 --behavior walk --port 23000 --episodes 0 --repeats 2 --travel 0.8"
SIDE="--amps 0.00 0.20 0.30 --ft_phase 0.5 --symmetric --spin_amp 0.25 --ik_iters 8"
# **`intent2`, because the first collection is kept.** It is the evidence that a discarded flag
# passes every other gate, and deleting it would leave only a claim about that.
RAW=data/allocentric/beh12_c10f10t10_intent2_raw
OUT=data/allocentric/beh12_c10f10t10_intent2_flat
CLEAN=data/allocentric/beh12_c10f10t10_sweepn00_flat

# name|extra flags.  Four per axis, because merge_behaviour_dirs requires the axes balanced.
CONDS=(
  "speed_stopmid|--cycles 7.1 --schedule '1@0.4 0@0.2 1@0.4'"
  "speed_slowfast|--cycles 7.1 --schedule '0.7@0.5 1.3@0.5'"
  "speed_fastslow|--cycles 7.1 --schedule '1.3@0.5 0.7@0.5'"
  "speed_breakrun|--cycles 7.1 --schedule '1@0.3 0@0.15 1.4@0.55'"
  "turn_onsetlate|--cycles 7.1 --spin_schedule '0@0.6 0.4@0.4'"
  "turn_onsetearly|--cycles 7.1 --spin_schedule '0@0.25 0.4@0.75'"
  "turn_pulse|--cycles 7.1 --spin_schedule '0@0.3 0.4@0.4 0@0.3'"
  "turn_reverse|--cycles 7.1 --spin_schedule '0.4@0.35 0@0.3 -0.4@0.35'"
  "side_L_stopmid|$SIDE --strafe -0.4 --spin 0.19 --schedule '1@0.4 0@0.2 1@0.4'"
  "side_L_slowfast|$SIDE --strafe -0.8 --spin 0.19 --schedule '0.7@0.5 1.3@0.5'"
  "side_R_stopmid|$SIDE --strafe 0.4 --spin -0.24 --schedule '1@0.4 0@0.2 1@0.4'"
  "side_R_slowfast|$SIDE --strafe 0.8 --spin -0.24 --schedule '0.7@0.5 1.3@0.5'"
)

if [ -d "$OUT" ]; then
  echo "skip collection, $OUT exists"
else
  for SPEC in "${CONDS[@]}"; do
    NAME="${SPEC%%|*}"; FLAGS="${SPEC#*|}"
    echo "=== $NAME   $FLAGS"
    # **`eval`, because a schedule is one argument containing spaces.** Without it bash splits
    # "1@0.4 0@0.2 1@0.4" into three words and argparse rejects the last two, which is exactly how
    # the first attempt failed. Every string here is a literal in this file.
    eval $PY sim/collect/collect_ik.py $COMMON --morphs "$MORPH" --out "$RAW/$NAME" $FLAGS
  done
  $PY scripts/dataset/merge_behaviour_dirs.py --src "$RAW" --out "$OUT" --embodiment hexapod
fi

echo
echo "############ GATE -- did the intervention reach the robot? ############"
echo "F165's first run failed here and nowhere else: --schedule was discarded by --gait cpg, and"
echo "the command lines, the log, walk_check, separability and the R2 tables all passed it."
echo "**Nothing below runs unless every condition changes within its clip.**"
$PY scripts/dataset/check_within_clip_intent.py --data "$OUT" --clean "$CLEAN"

echo
echo "############ SEPARABILITY -- context only, NOT the result ############"
$PY scripts/dataset/collect_beh12.py --separability "$OUT" || true

echo
echo "############ GAP, against F164's clean null: 0.729 single frame, +0.102 gap ############"
$PY scripts/diagnostics/objective_experiments/inverse_dynamics_r2.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt \
    --data "$OUT" --embodiment hexapod --cache results/wm/cache/intent2.pt --stride 2

echo
echo "############ INTENT -- the da row is the result ############"
$PY scripts/diagnostics/objective_experiments/intent_recoverability.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt \
    --data "$OUT" --embodiment hexapod --cache results/wm/cache/intent2.pt --stride 2
echo "--- the same three targets on F164's CLEAN arm, so the da row has its own null"
$PY scripts/diagnostics/objective_experiments/intent_recoverability.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt \
    --data "$CLEAN" --embodiment hexapod \
    --cache results/wm/cache/sweepn00.pt --stride 2

echo
echo "=== done  $(date '+%F %T')   send back this whole log"
