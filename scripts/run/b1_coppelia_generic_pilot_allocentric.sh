#!/usr/bin/env bash
# Replay the exact generic-pilot dynamics trajectories with only the camera changed.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
SRC=results/wm/dataset/b1_babble/coppelia_generic_pilot_v2
OUT="$SRC/allocentric"
mkdir -p "$OUT"

for seed in 0 1 2 3; do
  "$PY" sim/render/render_b1_replay.py \
    --scene sim/env/b1_flat_convex.ttt \
    --traj "$SRC/generic_seed${seed}.npz" --out "$OUT" \
    --align_yaw --cam_fov 24 --floor_scale 3 --fps 20
done

"$PY" sim/render/npz_to_video.py "$OUT"/generic_seed0.npz "$OUT"/generic_seed1.npz \
  "$OUT"/generic_seed2.npz "$OUT"/generic_seed3.npz \
  --out "$OUT/generic_all_four_allocentric.mp4" \
  --labels "seed 0 (fell);seed 1 (fell);seed 2 (upright);seed 3 (upright)" --fps 20
