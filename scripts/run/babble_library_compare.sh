#!/usr/bin/env bash
# Compare candidate libraries for direct action selection (one projector fitted per library, the
# deployment protocol: a new body only has its own babble). Log: results/wm/dataset/b1_babble/library_compare_log.txt
set -u
cd "$(dirname "$0")/../.."
BASE=wm/runs/beh12_hinge_cleansplit/b1_adapt_clean
OUT=results/wm/dataset/b1_babble/library_compare_log.txt
: > "$OUT"
for LIB in beh12_b1_ego_flat_cleantrain b1_babble_v2_ego_flat b1_babble_coppelia_spring_ego_flat; do
  PJ=$BASE/projector_lib_${LIB}.pt
  if [ "$LIB" = "beh12_b1_ego_flat_cleantrain" ]; then cp $BASE/projector_clean.pt $PJ; else
    .venv/bin/python3 -m wm.fit_projector --ckpt $BASE/body_head_b1_hex_clean.pt --hex_dir "" \
      --b1_dir data/egocentric/$LIB --cache results/wm/cache/proj_lib_${LIB}.pt --epochs 300 --out $PJ >> "$OUT" 2>&1
  fi
  .venv/bin/python3 - "$BASE" "$PJ" "$LIB" << 'PY'
import sys, torch
from wm.models.action_projector import action_dims_from
base, pj, lib = sys.argv[1:4]
ck = torch.load(f"{base}/body_head_b1_hex_clean.pt", map_location="cpu", weights_only=False)
sv = torch.load(pj, map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{base}/ckpt_lib_{lib}.pt")
PY
  echo "=== LIBRARY $LIB ===" >> "$OUT"
  .venv/bin/python3 scripts/diagnostics/objective_experiments/babble_library_eval.py \
    --ckpt $BASE/ckpt_lib_${LIB}.pt --candidates_dir data/egocentric/$LIB \
    --goal_dir data/egocentric/beh12_c10f10t10_ego_flat >> "$OUT" 2>&1
done
echo "ALL DONE" >> "$OUT"
