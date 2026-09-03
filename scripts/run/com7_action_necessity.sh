#!/usr/bin/env bash
# F155. Does one-step prediction need the action at all?
#
# Decides the next move after F154: if `null/real` sits near 1.0 the hinge can never bite and the
# prediction task itself must be made harder; if it is comfortably above 1.0, F154 is only a
# `lambda_recon` balance problem. The rebuild checkpoint lives here, so this has to run here.
set -euo pipefail
cd "$(dirname "$0")/.."

for arm in "wm/runs/beh12_actswm/best.pt|the rebuild (F153/F154)" \
           "wm/runs/beh12_hex-b1_body3/best.pt|the pre-rebuild model (F138)"; do
  ckpt="${arm%%|*}"; label="${arm##*|}"
  [ -f "$ckpt" ] || { echo "== skip $label -- $ckpt absent"; continue; }
  echo "== $label"
  .venv/bin/python3 scripts/diagnostics/objective_experiments/action_necessity.py \
      --ckpt "$ckpt" --data data/allocentric/beh12_c08f09t09_flat --embodiment hexapod --lags 1 2 3 5
  .venv/bin/python3 scripts/diagnostics/objective_experiments/action_necessity.py \
      --ckpt "$ckpt" --data data/allocentric/beh12_b1_flat --embodiment b1 --lags 1 2 3 5
done
