#!/usr/bin/env bash
# Hexapod-only pretrain with the B1 adapted on v3 frames using the CURRENT adaptation pipeline
# (Stage 1: LoRA rank 8, 3000 steps, pretraining losses incl. the Froude loss through the frozen head;
# Stage 2: projector; no head refit). Replaces the old-recipe hexonly_v3.sh row on the weekly update.
#   bash scripts/run/hexonly_v3_new.sh fmd_beh24_s0 fmd_beh24_s1
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
CK=(); M=()
for N in "$@"; do
  D=wm/runs/$N/b1_v3_new; mkdir -p $D
  echo "=== $N $(date)"
  $PY -m wm.adapt --ckpt wm/runs/$N/best.pt --data $B1 --embodiment b1 --clips 44 --test_clips 4 --stratify \
      --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0 \
      --out $D/adapted.pt 2>&1 | grep -v -i warn | tail -2
  $PY -m wm.fit_projector --ckpt $D/adapted.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt \
      --out $D/projector.pt 2>&1 | grep -v -i warn | tail -1
  $PY - $D/adapted.pt $D/projector.pt $D/ckpt.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
  CK+=(--ckpt "${N}=$D/ckpt.pt"); M+=(--model "${N}=$D/adapted.pt")
done
echo "=== B1 selection (v3 library)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir $LIB \
    --cache results/wm/cache/selection_eval_cands_v3.pt "${CK[@]}" --windows 21 | grep -E "w=|bounds"
echo "=== shared latent (adapted ITM; B1 on v3)"
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" \
    "${M[@]}" --out results/deck/shared_latent_${OUTTAG:-hexonly_v3_new} 2>&1 | grep -v -i warn | tail -5
echo HEXONLY_V3_NEW_DONE
