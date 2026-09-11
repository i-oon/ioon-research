#!/usr/bin/env bash
# Reproduce the three accepted native CoppeliaSim-Bullet CPG primitives.
# Requires one GUI CoppeliaSim instance listening on port 23000.
#
# The behavior presets live in b1_coppelia_cpg_controller.py:
#   forward: diagonal thigh/calf trot
#   lateral: left-pair/right-pair stepping with mirrored hip axes
#   yaw: same-sign diagonal hip wave (the measured yaw-dominant convention)
# These are simple motion primitives for the babble side of Q21/F194. They are
# not an expert controller and do not by themselves authorize the 2x2x2 rerun.
set -euo pipefail

PY=.venv/bin/python3
CTRL=scripts/diagnostics/objective_experiments/b1_coppelia_cpg_controller.py
OUT=results/wm/dataset/b1_babble/coppelia_native_cpg

mkdir -p "$OUT"

$PY "$CTRL" --behavior forward --steps 160 \
  --out "$OUT/final_forward.npz" --video "$OUT/final_forward.mp4"
$PY "$CTRL" --behavior lateral --direction 1 --steps 160 \
  --out "$OUT/final_lateral.npz" --video "$OUT/final_lateral.mp4"
$PY "$CTRL" --behavior yaw --direction 1 --steps 160 \
  --out "$OUT/final_yaw.npz" --video "$OUT/final_yaw.mp4"
