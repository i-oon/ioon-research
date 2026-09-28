#!/usr/bin/env bash
# Adapt the stride-5 model to c08f09t09 with switching babble (collect_babble_hex.py), ONE dataset
# for every stage (LAC-WM): Stage 1 LoRA ITM/FTM (38 clips, 10 held out), Stage 2 projector,
# Stage 3 joint projector+FTM (lambda_z 1); body head = pretrain's (F262, no Stage 4).
# Evaluated like F264 (goals: c10 held-out; candidates: c08 beh12 library) against zero-shot.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
PT=wm/runs/beh24_stride5_cleansplit/best.pt
DATA=data/egocentric/babble_c08f09t09_flat
D=wm/runs/beh24_stride5_cleansplit/c08_babble; mkdir -p $D
$PY -m wm.adapt --ckpt $PT --data $DATA --embodiment hexapod --clips 38 --test_clips 10 \
    --horizons 1 2 4 --lambda_hinge 0.5 --hinge_margin 0.1 --out $D/adapted.pt | tail -8
$PY -m wm.fit_projector --ckpt $D/adapted.pt --hex_dir $DATA --b1_dir "" --out $D/projector_hex.pt | tail -4
$PY - $D <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
d = sys.argv[1]
ck = torch.load(f"{d}/adapted.pt", map_location="cpu", weights_only=False)   # pretrain body head
sv = torch.load(f"{d}/projector_hex.pt", map_location="cpu", weights_only=False)
ck["projector"] = sv["projector"]; ck["action_dims"] = action_dims_from(sv)
torch.save(ck, f"{d}/ckpt_s12.pt"); print("merged", d)
PYEOF
$PY -m wm.adapt_joint --ckpt $D/ckpt_s12.pt --projector $D/projector_hex.pt --data $DATA \
    --embodiment hexapod --lambda_z 1 --out $D/ckpt_s123.pt | tail -6
$PY scripts/diagnostics/objective_experiments/selection_eval.py --embodiment hexapod \
    --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat \
    --goal_dir data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout --cache results/wm/cache/selection_eval_c08.pt \
    --ckpt s5_zeroshot=wm/runs/beh24_stride5_cleansplit/c08_zeroshot/ckpt_lib_zeroshot.pt \
    --ckpt s5_babble_s12=$D/ckpt_s12.pt --ckpt s5_babble_s123=$D/ckpt_s123.pt --windows 0 11 \
    | tee results/deck/window_sweep/c08_babble_selection.txt
echo C08_BABBLE_DONE
