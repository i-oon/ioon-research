#!/usr/bin/env bash
#
# Confirmation run: does the REAL counterfactual design produce separable futures?
#
#   bash scripts/dataset/confirm_counterfactual.sh      # CoppeliaSim GUI, exactly ONE instance
#
# **The 21x insect number does not decide this and must not be quoted as if it did.** That compared
# different behaviours from the same *spawn*, diverging at frame 0 with no shared momentum -- the
# easy version. **The design we would actually collect shares a prefix of `k` frames and branches
# from a shared pose**, so both futures start with identical velocity and contact state and the
# action has to overcome that first. **Expect a lower ratio here. This run is the one that counts.**
#
# **Three things are checked and all three must pass:**
#
#   1. the shared prefix reproduces  -- within the measured floor, about 7 mm at 15 frames
#   2. the branch diverges           -- clearly above that floor, on position AND heading
#   3. turning survives              -- **the weak case for both robots** (F136), and position alone
#                                       understates it, which is why heading is measured
#
# **Pre-registered pass mark: 3x noise on position *and* heading at every horizon, per arm.** A cell
# that clears position and fails heading is a failure -- it means the two futures end in the same
# place facing different ways, which is what turning *is*.
#
# **Every run here also renders.** Numbers decide pass or fail; the merged side-by-side is the
# visual check that the counterfactual is real, and it is produced without being asked for.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
OUT=data/allocentric/cf_confirm
VID=results/cf_confirm
BRANCH=33                       # half of a 66-frame clip
MORPH=c10f10t10=medauroidea_c10f10t10.ttt
COMMON="--gait cpg --scale 0.65 --cam_dx -0.6 --behavior walk --port 23000 --episodes 0 --travel 0.8"
mkdir -p "$OUT" "$VID"

# ---------------------------------------------------------------- insect
# **Every arm walks straight for the first half.** `0@0.5 X@0.5` and `0@1.0` issue identical
# commands until frame 33, so the prefix is shared by construction rather than by hope.
echo "=== insect: shared prefix to frame $BRANCH, then branch"
declare -A ARMS=(
  [forward]="--spin_schedule '0@1.0'"
  [forward_repeat]="--spin_schedule '0@1.0'"      # identical flags on purpose: this pair IS the noise floor
  [turn]="--spin_schedule '0@0.5 0.4@0.5'"
  [side]="--strafe_schedule '0@0.5 0.6@0.5'"
  [faster]="--schedule '1@0.5 1.4@0.5'"
)
for NAME in forward forward_repeat turn side faster; do
  D="$OUT/insect_$NAME"
  [ -d "$D" ] || eval $PY sim/collect/collect_ik.py $COMMON --repeats 1 --morphs "$MORPH" \
      --out "$D" ${ARMS[$NAME]}
done
# **Absolute target, and no error suppression.** `ln -sf` resolves a *relative* target from the
# link's own directory, so a repo-relative path here silently produced
# `data/allocentric/cf_confirm/data/allocentric/cf_confirm/...` and the failure was swallowed by `2>/dev/null || true`.
# The `-not -name manifest` matters too: each arm directory also holds a `manifest.npy`.
for N in forward forward_repeat turn side faster; do
  SRC=$(ls "$PWD/$OUT/insect_$N"/*.npz | grep -v manifest | head -1)
  [ -n "$SRC" ] || { echo "no clip in $OUT/insect_$N"; exit 1; }
  ln -sfn "$SRC" "$OUT/insect_$N.npz"
  [ -r "$OUT/insect_$N.npz" ] || { echo "link $OUT/insect_$N.npz does not resolve"; exit 1; }
done

echo
echo "--- insect divergence after the branch, position AND heading"
$PY scripts/diagnostics/egocentric_view/branch_divergence.py --embodiment hexapod --branch $BRANCH \
    --noise "$OUT/insect_forward.npz" "$OUT/insect_forward_repeat.npz" \
    --pair forward="$OUT/insect_forward.npz" turn="$OUT/insect_turn.npz" \
           side="$OUT/insect_side.npz" faster="$OUT/insect_faster.npz"

echo
echo "--- merged clips: shared prefix in sync, then the split"
for P in "turn forward-vs-turn" "side forward-vs-side" "faster forward-vs-faster"; do
  set -- $P
  $PY scripts/render/merge_counterfactual.py --a "$OUT/insect_forward.npz" \
      --b "$OUT/insect_$1.npz" --branch $BRANCH --label_a forward --label_b "$1" \
      --out "$VID/insect_$2.mp4"
done

# ---------------------------------------------------------------- B1
# **The B1 branches from a bit-identical saved state** (`mjSTATE_INTEGRATION`), so its prefix is not
# merely close and its noise floor is exactly zero -- measured, not assumed. It is rendered by
# kinematic replay from stored states, so CoppeliaSim's own non-determinism never enters.
echo
echo "=== B1: branch from a saved state (physics is local and already verified)"
$PY sim/collect/branch_b1_mujoco.py --prefix 60 --branch_steps 30 \
    --arms forward=0.4,0,0 turn=0.4,0,0.6 side=0,0.4,0 --out "$OUT"

echo
echo "--- B1 divergence after the branch, position AND heading"
$PY scripts/diagnostics/egocentric_view/branch_divergence.py --embodiment b1 --branch 60 \
    --noise "$OUT/b1_forward.npz" "$OUT/b1_forward_repeat.npz" \
    --pair forward="$OUT/b1_forward.npz" turn="$OUT/b1_turn.npz" side="$OUT/b1_side.npz" \
    --horizons 1 3 5 10 15 25

echo
echo "--- B1 replay: the frames, with the camera settings that are part of the data"
for N in forward turn side; do
  D="$OUT/b1_render_$N"
  [ -d "$D" ] || $PY sim/render/render_b1_replay.py --scene sim/env/b1_flat.ttt \
      --traj "$OUT/b1_$N.npz" --out "$D" --cam_fov 24 --spawn 0 0 --floor_scale 3
  ln -sf "$(ls $D/*.npz | head -1)" "$OUT/b1_frames_$N.npz"
done

echo
echo "--- B1 merged clips"
for P in "turn forward-vs-turn" "side forward-vs-side"; do
  set -- $P
  $PY scripts/render/merge_counterfactual.py --a "$OUT/b1_frames_forward.npz" \
      --b "$OUT/b1_frames_$1.npz" --branch 60 --label_a forward --label_b "$1" \
      --out "$VID/b1_$2.mp4"
done

echo
echo "=== done  $(date '+%F %T')   send back this log and the mp4s in $VID/"
