#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
SRC=results/wm/dataset/b1_babble/coppelia_stance_swing_pilot
OUT="$SRC/allocentric"
mkdir -p "$OUT"

for ratio in 1.5 2.0; do
  "$PY" sim/render/render_b1_replay.py --scene sim/env/b1_flat_convex.ttt \
    --traj "$SRC/calf${ratio}_seed3.npz" --out "$OUT" \
    --align_yaw --cam_fov 24 --floor_scale 3 --fps 20
done

"$PY" sim/render/npz_to_video.py "$OUT/calf1.5_seed3.npz" "$OUT/calf2.0_seed3.npz" \
  --out "$OUT/seed3_calf_lift_comparison.mp4" \
  --labels "calf 1.5x, Froude 0.0121;calf 2.0x, Froude 0.0179" --fps 20
