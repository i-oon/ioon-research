#!/usr/bin/env bash
#
# F163. Is the action redundant because of the ENCODER, or because the DATA is a clean gait?
#
#   bash scripts/dataset/collect_offrhythm.sh        # CoppeliaSim GUI, exactly ONE instance
#
# **The hypothesis nothing so far has tested.** F159 showed the insect's command is readable from a
# single frame at R2 0.779; F162 showed a motion transform of the *representation* neither breaks
# that nor survives it. **Neither touched the data.** On a steady gait the pose fixes the phase and
# the phase predicts the next frame, so the action is redundant **by rhythm**. Off the rhythm it
# should not be.
#
# **This changes the DATA, not the representation.** Blinding or corrupting `e_t` is the failed
# "cripple the frame" trap and is not what happens here: both arms are collected identically, the
# encoder is untouched, and the only difference is what the robot was commanded to do.
#
# **Two arms in one sitting**, so camera, scene and lighting cannot drift between them. That matters
# more than matching `data/allocentric/beh12_c10f10t10_flat`, which was collected on another day -- **the
# existing set is not the control; the clean arm below is.**
#
#   clean   --cmd_noise 0      the steady gait every set so far has used
#   noisy   --cmd_noise 0.05   temporally correlated noise, tau 5 steps, on the final joint command
#
# **The logged `a_t` is the perturbed command.** The noise is added after the heading and oscillator
# branches and before `actions.append`, so nothing downstream re-derives a clean command. If that
# were not true the whole measurement would be void.
#
# **Watch the videos before measuring anything.** Six defects in this project were found by looking
# and none by the tables passing at the time. A robot that has fallen over is not off-rhythm
# walking, and its numbers would mean nothing. `walk_check` prints a verdict per clip; read it.
#
# 12 conditions x 2 repeats x 2 arms. Expect a couple of hours.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
MORPH=c10f10t10=medauroidea_c10f10t10.ttt

for ARM in "clean 0.0" "noisy 0.05"; do
  set -- $ARM
  RAW="data/allocentric/beh12_c10f10t10_rhythm${1}_raw"
  OUT="data/allocentric/beh12_c10f10t10_rhythm${1}_flat"
  echo "=== $1 arm, cmd_noise $2"
  if [ -d "$OUT" ]; then echo "skip, $OUT exists"; continue; fi
  $PY scripts/dataset/collect_beh12.py --morph "$MORPH" --out "$RAW" --repeats 2 \
      --extra --cmd_noise "$2" --noise_tau 5
  $PY scripts/dataset/merge_behaviour_dirs.py --src "$RAW" --out "$OUT" --embodiment hexapod
done

echo
echo "=== separability, both arms -- the noisy arm must still resolve its twelve conditions"
for ARM in clean noisy; do
  echo "--- $ARM"
  $PY scripts/dataset/collect_beh12.py --separability "data/allocentric/beh12_c10f10t10_rhythm${ARM}_flat" || true
done

echo
echo "=== now look at them, then measure"
echo "  $PY scripts/dataset/preview_clips.py --data data/allocentric/beh12_c10f10t10_rhythmnoisy_flat"
echo "  $PY scripts/diagnostics/objective_experiments/inverse_dynamics_r2.py --ckpt wm/runs/beh12_hex-b1_body3/best.pt \\"
echo "      --data data/allocentric/beh12_c10f10t10_rhythm{clean,noisy}_flat --embodiment hexapod --cache <fresh>"
