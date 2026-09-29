#!/usr/bin/env bash
# Local, sequential (plan 2026-09-29 evening).
#  A. Stronger anchored adaptation on the hexapod-only pretrain fmd_beh24_s0: LoRA rank 8, 3000 steps,
#     same B1 data at every stage (44 + 4 clips for Stage 1, 48 for Stages 2 and 4).
#       A1 Froude-head anchor     A2 Froude-head + similarity anchors
#     Measured: retrieval of the adapted ITM; B1 selection with and without the Stage 4 head refit.
#  B. jointD_sim_s1: joint hexapod + B1 pretraining, Froude only, similarity 0.05, seed 1 -- the server's
#     jointD_sim_s0 on another seed. Then retrieval and B1 selection with Stage 2 only.
#   nohup bash scripts/run/local_overnight.sh > results/wm/logs/local_overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
B1=data/egocentric/beh24_b1_ego_flat_cleantrain
HEX=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
merge() {
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3]); print("merged ->", sys.argv[3])
PYEOF
}

E=wm/runs/fmd_beh24_s0/strong_adapt; mkdir -p $E
CK=(); M=()
for arm in "froude:1.0:0" "both:1.0:1.0"; do
  IFS=: read name wf ws <<< "$arm"
  echo "=== A: Stage 1 rank 8, 3000 steps, anchor $name $(date)"
  $PY -m wm.adapt --ckpt wm/runs/fmd_beh24_s0/best.pt --data $B1 --embodiment b1 --clips 44 --test_clips 4 \
      --stratify --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 \
      --anchor_froude $wf --anchor_sim $ws --out $E/adapted_$name.pt 2>&1 | grep -v -i warn | tail -6
  $PY -m wm.fit_projector --ckpt $E/adapted_$name.pt --hex_dir $HEX --b1_dir $B1 --out $E/projector_$name.pt \
      2>&1 | grep -v -i warn | tail -1
  merge $E/adapted_$name.pt $E/projector_$name.pt $E/ckpt_${name}_noS4.pt
  $PY -m wm.fit_body_head --ckpt $E/adapted_$name.pt --data $B1 --embodiment b1 --also hexapod=$HEX \
      --out $E/head_$name.pt 2>&1 | grep -v -i warn | tail -1
  merge $E/head_$name.pt $E/projector_$name.pt $E/ckpt_${name}_S4.pt
  CK+=(--ckpt "A_${name}_noS4=$E/ckpt_${name}_noS4.pt" --ckpt "A_${name}_S4=$E/ckpt_${name}_S4.pt")
  M+=(--model "strong_$name=$E/adapted_$name.pt")
done
$PY scripts/diagnostics/objective_experiments/selection_eval.py "${CK[@]}" --windows 11 | grep -E "w=|bounds"
$PY scripts/figures/shared_latent_figure.py --device cuda "${M[@]}" --out results/deck/shared_latent_strong_adapt \
  2>&1 | grep -v -i warn | tail -3
echo STRONG_ADAPT_DONE

N=jointD_sim_s1
echo "=== B: train $N $(date)"
$PY -m wm.train \
  --sources hexapod=$HEX b1=$B1 \
  --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval b1=data/egocentric/beh24_b1_ego_flat_cleanval \
  --lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0 \
  --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 50 \
  --checkpoint_every 2 --resume auto --lambda_sim 0.05 --seed 1 --name $N >> results/wm/logs/$N.log 2>&1 \
  || { echo "TRAIN FAILED"; exit 1; }
J=wm/runs/$N/stage2only; mkdir -p $J
$PY -m wm.fit_projector --ckpt wm/runs/$N/best.pt --hex_dir $HEX --b1_dir $B1 --out $J/projector.pt 2>&1 | grep -v -i warn | tail -1
merge wm/runs/$N/best.pt $J/projector.pt $J/ckpt_stage2only.pt
$PY scripts/diagnostics/objective_experiments/selection_eval.py --ckpt "${N}_stage2only=$J/ckpt_stage2only.pt" --windows 11 | grep -E "w=|bounds"
$PY scripts/figures/shared_latent_figure.py --device cuda --model "$N=wm/runs/$N/best.pt" \
  --out results/deck/shared_latent_$N 2>&1 | grep -v -i warn | tail -2
echo LOCAL_OVERNIGHT_DONE
