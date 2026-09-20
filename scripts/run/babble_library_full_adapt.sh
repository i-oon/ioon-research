#!/usr/bin/env bash
# Deployment-faithful protocol for a new body that only has babble: stage 1 (ITM/FTM adapt), stage 2
# (projector) and stage 4 (Cross-Body Head, hexapod rehearsal) ALL fitted on the babble library, then
# direct action selection over that same library. Log: results/wm/dataset/b1_babble/library_full_adapt_log.txt
set -u
cd "$(dirname "$0")/../.."
BASE=wm/runs/beh12_hinge_cleansplit
OUT=results/wm/dataset/b1_babble/library_full_adapt_log.txt
: > "$OUT"
for LIB in b1_babble_v2_ego_flat b1_babble_coppelia_spring_ego_flat; do
  D=$BASE/b1_adapt_clean; A=$D/adapted_${LIB}.pt; PJ=$D/proj_full_${LIB}.pt; H=$D/head_full_${LIB}.pt
  echo "=== ADAPT $LIB ===" >> "$OUT"
  .venv/bin/python3 -m wm.adapt --ckpt $BASE/best.pt --data data/egocentric/$LIB --embodiment b1 \
    --clips 9 --steps 1000 --lambda_hinge 0.5 --hinge_margin 0.1 --out $A >> "$OUT" 2>&1
  .venv/bin/python3 -m wm.fit_projector --ckpt $A --hex_dir "" --b1_dir data/egocentric/$LIB \
    --cache results/wm/cache/proj_lib_${LIB}.pt --epochs 300 --out $PJ >> "$OUT" 2>&1
  .venv/bin/python3 -m wm.fit_body_head --ckpt $A --data data/egocentric/$LIB --embodiment b1 \
    --also hexapod=data/egocentric/beh12_c10f10t10_ego_flat_cleantrain --val_frac 0.15 --epochs 400 \
    --cache results/wm/cache/sweep_lib_${LIB}.pt --out $H >> "$OUT" 2>&1
  .venv/bin/python3 - "$H" "$PJ" "$D/ckpt_full_${LIB}.pt" << 'PY'
import sys, torch
from wm.models.action_projector import action_dims_from
h, pj, out = sys.argv[1:4]
ck = torch.load(h, map_location="cpu", weights_only=False); sv = torch.load(pj, map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv); torch.save(ck, out)
PY
  echo "=== LIBRARY_FULL $LIB ===" >> "$OUT"
  .venv/bin/python3 scripts/diagnostics/objective_experiments/babble_library_eval.py \
    --ckpt $D/ckpt_full_${LIB}.pt --candidates_dir data/egocentric/$LIB \
    --goal_dir data/egocentric/beh12_c10f10t10_ego_flat >> "$OUT" 2>&1
done
echo "ALL DONE" >> "$OUT"
