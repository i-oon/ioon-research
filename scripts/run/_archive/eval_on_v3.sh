#!/usr/bin/env bash
# Do existing joint models depend on the B1 rendering difference? (2026-09-30)
# No retraining: the same pretrained models, the B1 switched to the v3 re-render (rendering matched to the
# hexapod's; F275 addendum). Stage 2 refit on v3 B1 clean-train; B1 selection on the v3 library; shared
# latent with the B1 on v3. Compare with the old-render numbers (F284).
#   bash scripts/run/eval_on_v3.sh jointD_sim_s0 jointD_nosim_s0 jointD_sim_s1
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
LIB=data/egocentric_v3/beh12_b1_ego_flat_cleantrain
CK=(); M=()
for N in "$@"; do
  R=wm/runs/$N; J=$R/stage2only_v3; mkdir -p $J
  $PY -m wm.fit_projector --ckpt $R/best.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v3.pt \
      --out $J/projector.pt 2>&1 | grep -v -i warn | tail -1
  $PY - $R/best.pt $J/projector.pt $J/ckpt.pt <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
  CK+=(--ckpt "${N}_v3=$J/ckpt.pt"); M+=(--model "${N}_v3=$R/best.pt")
done
echo "=== B1 selection, v3 library"
$PY scripts/diagnostics/objective_experiments/selection_eval.py --candidates_dir $LIB \
    --cache results/wm/cache/selection_eval_cands_v3.pt "${CK[@]}" --windows 21 | grep -E "w=|bounds"
echo "=== shared latent, B1 on v3"
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=results/wm/cache/selection_eval_cands_v3.pt" \
    "${M[@]}" --out results/deck/shared_latent_v3 2>&1 | grep -v -i warn | tail -4
echo EVAL_V3_DONE
