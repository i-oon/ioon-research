#!/usr/bin/env bash
# The standard evaluation: every number that goes on a slide comes from here. Conventions: memory
# `project_timescale_conventions` / FINDINGS F295, F301 (1 s edge-correct switch-aware Froude labels at the CoM,
# read-out window 21). DATA = data/counterfactual_walks (doc/DATA.md, doc/DATA_PLAN.md): TEST = the heldout split only (never in training or
# checkpoint selection; `check_splits.py`). Goals = all 24 hexapod heldout clips; libraries = B1 heldout,
# c08 (never trained), hexapod heldout with the goal clip itself excluded (leave-goal-out); read-out start states
# from the same heldout libraries. Data dirs are set once below (override with env vars).
#
#   bash scripts/run/eval_suite.sh NAME joint   wm/runs/<run>/best.pt     # B1 was in pretraining
#   bash scripts/run/eval_suite.sh NAME hexonly wm/runs/<run>/best.pt     # B1 adapted (current recipe)
#
# Per model, into results/eval/NAME/ (one log per part, summary.txt with every result line):
#   1. checkpoints: hexapod projector (c10, c08); B1: projector only (joint) or LoRA r8 3000 steps with the
#      Froude loss + projector (hexonly)
#   2. selection, normalised score, w=21: B1 (recorded goal and vision-read goal), c08 zero-shot, c10
#   3. shared latent (body-ID, cross-body R2, kNN mixing, retrieval)
#   4. counterfactual read-out on the heldout physics branches (Pearson r across the 24 commands): hexapod, B1,
#      c08 (held-out body, zero-shot, hexapod projector)
# Sequential; one GPU job at a time. Needs CoppeliaSim on port 23000 only if a read-out cache is missing.
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=$1; MODE=$2; PT=$3
PY=.venv/bin/python3
CW=${CW:-data/counterfactual_walks}
HEX=${HEX:-$CW/c10_clips_train}          # hexapod projector fit
B1=${B1:-$CW/b1_clips_train}             # B1 adaptation / projector fit
LIB=${LIB:-$CW/b1_clips_heldout}         # B1 test library
HEXT=${HEXT:-$CW/c10_clips_heldout}      # hexapod test library + goals
C08=${C08:-$CW/c08_clips_heldout}          # c08 test library (held-out morphology, never trained; collect_c08_test_set.py)
LIBC=results/wm/cache/test_v4_b1_heldout.pt
GOALS=$HEXT
OUT=results/eval/$NAME; CK=$OUT/ckpt; mkdir -p $CK
SEL=scripts/diagnostics/objective_experiments/selection_eval.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
merge() {  # base ckpt, projector file, out
  $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
}
echo "=== $NAME ($MODE) $PT $(date)" | tee $OUT/summary.txt

# 1. checkpoints
$PY -m wm.fit_projector --ckpt $PT --hex_dir $HEX --b1_dir "" --out $CK/projector_hex.pt 2>&1 | quiet | tail -1
merge $PT $CK/projector_hex.pt $CK/hex.pt
if [ "$MODE" = joint ]; then
  $PY -m wm.fit_projector --ckpt $PT --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v4.pt \
      --out $CK/projector_b1.pt 2>&1 | quiet | tail -1
  merge $PT $CK/projector_b1.pt $CK/b1.pt
  ITM_B1=$PT
else
  $PY -m wm.adapt --ckpt $PT --data $B1 --embodiment b1 --clips 44 --test_clips 4 --stratify --lambda_hinge 0.5 \
      --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0 --out $CK/adapted_b1.pt 2>&1 | quiet | tail -1
  $PY -m wm.fit_projector --ckpt $CK/adapted_b1.pt --hex_dir $HEX --b1_dir $B1 --cache results/wm/cache/fitproj_v4.pt \
      --out $CK/projector_b1.pt 2>&1 | quiet | tail -1
  merge $CK/adapted_b1.pt $CK/projector_b1.pt $CK/b1.pt
  ITM_B1=$CK/adapted_b1.pt
fi

# 2. selection
{
echo "--- selection B1, recorded goal"
$PY $SEL --conditions all --candidates_dir $LIB --goal_dir $GOALS --cache $LIBC --windows 21 \
    --ckpt $NAME=$CK/b1.pt 2>&1 | grep -E "w=|bounds"
echo "--- selection B1, vision-read goal"
$PY $SEL --conditions all --candidates_dir $LIB --goal_dir $GOALS --cache $LIBC --windows 21 --goal_source vision \
    --ckpt $NAME=$CK/b1.pt 2>&1 | grep -E "w=|bounds"
echo "--- selection c08 zero-shot"
$PY $SEL --conditions all --embodiment hexapod --candidates_dir $C08 --goal_dir $GOALS \
    --cache results/wm/cache/test_v4_c08_heldout.pt --windows 21 --ckpt $NAME=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
echo "--- selection c10 same body"
$PY $SEL --conditions all --embodiment hexapod --candidates_dir $HEXT --goal_dir $GOALS \
    --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --ckpt $NAME=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
} 2>&1 | tee $OUT/selection.txt | tee -a $OUT/summary.txt

# 3. shared latent
echo "--- shared latent" | tee -a $OUT/summary.txt
$PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=$LIBC" --c10 "$HEXT=results/wm/cache/test_v4_hex_heldout.pt" \
    --model "$NAME=$ITM_B1" --out $OUT/shared_latent 2>&1 | quiet | tail -3 | tee $OUT/shared_latent.txt | tee -a $OUT/summary.txt

# 4. counterfactual read-out on the v4 physics branches (heldout), no simulator
CFH=${CFH:-$CW/c10_branches_heldout}; CFB=${CFB:-$CW/b1_branches_heldout}; CFC=${CFC:-$CW/c08_branches_heldout}
RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
{
echo "--- counterfactual read-out hexapod (physics branches, heldout)"
$PY $RO --embodiment hexapod --cf_dir $CFH --ckpt $NAME=$CK/hex.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
echo "--- counterfactual read-out B1 (physics branches, heldout)"
$PY $RO --embodiment b1 --cf_dir $CFB --ckpt $NAME=$CK/b1.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
echo "--- counterfactual read-out c08 zero-shot (physics branches, held-out body, hexapod projector)"
$PY $RO --embodiment hexapod --cf_dir $CFC --ckpt $NAME=$CK/hex.pt --pairs 1 11 2>&1 | quiet | grep -A3 "^==="
} 2>&1 | tee $OUT/readout.txt | tee -a $OUT/summary.txt
echo "EVAL_SUITE_DONE $NAME" | tee -a $OUT/summary.txt
