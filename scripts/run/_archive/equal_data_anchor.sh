#!/usr/bin/env bash
# Local, sequential (one GPU job at a time). Plan 2026-09-29.
#  J   joint hexapod + B1 + similarity pretrain (joint_sim_beh24_s0), evaluated as a joint model:
#      Stage 2 only (B1 projector); ITM, FTM and Froude head straight from pretraining.
#  E*  B1 adaptation onto the hexapod-only pretrain fmd_beh24_s0 with the SAME B1 data at every stage
#      (beh24 B1 clean-train, 48 clips: Stage 1 adapts on 44 and reports on 4; Stages 2 and 4 use all 48):
#      E1  no anchor, Stages 1 + 2 + 4 (head refit)       -- the usual recipe, equal data
#      E2  Froude-head anchor, Stages 1 + 2, NO Stage 4   -- is the anchored space usable as it is?
#      E3  Froude-head anchor, Stages 1 + 2 + 4
# Measures: B1 selection (normalised score, library = beh12 B1 clean-train, goals = held-out hexapod
# clips) and cross-body retrieval of the adapted ITM.
#   nohup bash scripts/run/equal_data_anchor.sh > results/wm/logs/equal_data_anchor.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
merge() {  # base ckpt, projector file, out
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3]); print("merged ->", sys.argv[3])
PYEOF
}
CK=()

echo "=== J: joint model, Stage 2 only $(date)"
J=wm/runs/joint_sim_beh24_s0/stage2only; mkdir -p $J
$PY -m wm.fit_projector --ckpt wm/runs/joint_sim_beh24_s0/best.pt --hex_dir $HEX --b1_dir $B1 \
    --out $J/projector.pt 2>&1 | grep -v -i warn | tail -3
merge wm/runs/joint_sim_beh24_s0/best.pt $J/projector.pt $J/ckpt_stage2only.pt
CK+=(--ckpt "J_joint_stage2only=$J/ckpt_stage2only.pt")

BASE=wm/runs/fmd_beh24_s0/best.pt
E=wm/runs/fmd_beh24_s0/equal_data; mkdir -p $E
for arm in "noanchor:0" "anchor:1.0"; do
  IFS=: read name wf <<< "$arm"
  echo "=== Stage 1 ($name), 44 clips $(date)"
  $PY -m wm.adapt --ckpt $BASE --data $B1 --embodiment b1 --clips 44 --test_clips 4 --stratify \
      --lambda_hinge 0.5 --hinge_margin 0.1 --anchor_froude $wf --out $E/adapted_$name.pt 2>&1 \
      | grep -v -i warn | tail -7
  $PY -m wm.fit_projector --ckpt $E/adapted_$name.pt --hex_dir $HEX --b1_dir $B1 \
      --out $E/projector_$name.pt 2>&1 | grep -v -i warn | tail -2
  merge $E/adapted_$name.pt $E/projector_$name.pt $E/ckpt_${name}_noS4.pt
  $PY -m wm.fit_body_head --ckpt $E/adapted_$name.pt --data $B1 --embodiment b1 --also hexapod=$HEX \
      --out $E/head_$name.pt 2>&1 | grep -v -i warn | tail -3
  merge $E/head_$name.pt $E/projector_$name.pt $E/ckpt_${name}_S4.pt
done
CK+=(--ckpt "E1_noanchor_S4=$E/ckpt_noanchor_S4.pt" --ckpt "E2_anchor_noS4=$E/ckpt_anchor_noS4.pt" \
     --ckpt "E3_anchor_S4=$E/ckpt_anchor_S4.pt" --ckpt "E0_noanchor_noS4=$E/ckpt_noanchor_noS4.pt")

echo "=== B1 selection $(date)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py "${CK[@]}" --windows 21 | grep -E "w=|bounds"
echo "=== retrieval of the adapted ITMs $(date)"
$PY scripts/figures/shared_latent_figure.py --device cuda --model "equal_noanchor=$E/adapted_noanchor.pt" \
    --model "equal_anchor=$E/adapted_anchor.pt" --out results/deck/shared_latent_equal_data 2>&1 | grep -v -i warn | tail -4
echo EQUAL_DATA_ANCHOR_DONE
