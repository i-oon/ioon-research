#!/usr/bin/env bash
# Hexapod-only pretrain (fmd_beh24_s0) with the B1 adapted on v3 frames (usual recipe, same v3 data at every
# stage), for the clean "hexapod only" row of the weekly update: B1 / c08 / c10 selection and shared latent.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
D=wm/runs/fmd_beh24_s0/b1_v3; mkdir -p $D
$PY -m wm.adapt --ckpt wm/runs/fmd_beh24_s0/best.pt --data $B1 --embodiment b1 --clips 44 --test_clips 4 --stratify \
    --lambda_hinge 0.5 --hinge_margin 0.1 --out $D/adapted.pt 2>&1 | grep -v -i warn | tail -2
$PY -m wm.fit_projector --ckpt $D/adapted.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt \
    --out $D/projector.pt 2>&1 | grep -v -i warn | tail -1
$PY -m wm.fit_body_head --ckpt $D/adapted.pt --data $B1 --embodiment b1 --also hexapod=$HEX \
    --cache results/wm/cache/fitbody_v3.pt --out $D/head.pt 2>&1 | grep -v -i warn | tail -1
$PY - $D/head.pt $D/projector.pt $D/ckpt.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
echo "=== B1 (v3 library)"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir $LIB \
    --cache results/wm/cache/selection_eval_cands_v3.pt --ckpt "hexonly_v3=$D/ckpt.pt" --windows 11 | grep -E "w=|bounds"
echo "=== shared latent (pretrained ITM; B1 on v3)"
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" \
    --model "hexonly=wm/runs/fmd_beh24_s0/best.pt" --out results/deck/shared_latent_hexonly_v3 2>&1 | grep -v -i warn | tail -2
echo HEXONLY_V3_DONE
