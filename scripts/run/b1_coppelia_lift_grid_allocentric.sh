#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
SRC=results/wm/dataset/b1_babble/coppelia_lift_frequency_grid
OUT="$SRC/allocentric"
mkdir -p "$OUT"

for tag in \
  f1.5_a0.14_c2.0 f1.5_a0.14_c2.5 \
  f1.75_a0.14_c2.0 f1.75_a0.14_c2.5 \
  f2.0_a0.14_c2.0 f2.0_a0.14_c2.5 \
  f1.75_a0.18_c2.5; do
  "$PY" sim/render/render_b1_replay.py --scene sim/env/b1_flat_convex.ttt \
    --traj "$SRC/${tag}.npz" --out "$OUT" \
    --align_yaw --cam_fov 24 --floor_scale 3 --fps 20
done

"$PY" sim/render/npz_to_video.py \
  "$OUT/f1.5_a0.14_c2.0.npz" "$OUT/f1.5_a0.14_c2.5.npz" \
  "$OUT/f1.75_a0.14_c2.0.npz" "$OUT/f1.75_a0.14_c2.5.npz" \
  "$OUT/f2.0_a0.14_c2.0.npz" "$OUT/f2.0_a0.14_c2.5.npz" \
  --out "$OUT/stable_frequency_lift_grid.mp4" \
  --labels "1.5Hz calf2x;1.5Hz calf2.5x;1.75Hz calf2x;1.75Hz calf2.5x;2.0Hz calf2x;2.0Hz calf2.5x" \
  --fps 20

"$PY" sim/render/npz_to_video.py "$OUT/f1.75_a0.18_c2.5.npz" \
  --out_dir "$OUT" --fps 20
