#!/usr/bin/env bash
# Can adaptation absorb the B1 render change? (2026-09-30, after F285)
# Joint models pretrained on the OLD B1 renders, adapted to the B1's v3 (matched-render) frames with the
# same v3 data at every stage (beh24 B1 clean-train: Stage 1 on 44 + 4 clips, Stages 2 and 4 on 48):
#   A  usual: LoRA Stage 1 (no anchor) + projector + Froude-head refit (Stage 4)
#   B  anchored: LoRA Stage 1 with the frozen pretrained Froude head as anchor + projector, no refit
# Then B1 selection on the v3 library and shared latent with the B1 on v3.
#   bash scripts/run/finetune_v3.sh jointD_sim_s0 jointD_nosim_s0
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
merge() {
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
}
CK=(); M=()
for N in "$@"; do
  D=wm/runs/$N/finetune_v3; mkdir -p $D
  for arm in "A:0" "B:1.0"; do
    IFS=: read a wf <<< "$arm"
    echo "=== $N arm $a $(date)"
    $PY -m wm.adapt --ckpt wm/runs/$N/best.pt --data $B1 --embodiment b1 --clips 44 --test_clips 4 --stratify \
        --lambda_hinge 0.5 --hinge_margin 0.1 --anchor_froude $wf --out $D/adapted_$a.pt 2>&1 | grep -v -i warn | tail -3
    $PY -m wm.fit_projector --ckpt $D/adapted_$a.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt \
        --out $D/projector_$a.pt 2>&1 | grep -v -i warn | tail -1
    if [ $a = A ]; then
      $PY -m wm.fit_body_head --ckpt $D/adapted_$a.pt --data $B1 --embodiment b1 --also hexapod=$HEX \
          --cache results/wm/cache/fitbody_v3.pt --out $D/head_$a.pt 2>&1 | grep -v -i warn | tail -1
      merge $D/head_$a.pt $D/projector_$a.pt $D/ckpt_$a.pt
    else
      merge $D/adapted_$a.pt $D/projector_$a.pt $D/ckpt_$a.pt
    fi
    CK+=(--ckpt "${N}_${a}=$D/ckpt_$a.pt"); M+=(--model "${N}_${a}=$D/adapted_$a.pt")
  done
done
echo "=== B1 selection, v3 library"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir $LIB \
    --cache results/wm/cache/selection_eval_cands_v3.pt "${CK[@]}" --windows 11 | grep -E "w=|bounds"
echo "=== shared latent, B1 on v3"
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" \
    "${M[@]}" --out results/deck/shared_latent_finetune_v3 2>&1 | grep -v -i warn | tail -5
echo FINETUNE_V3_DONE
