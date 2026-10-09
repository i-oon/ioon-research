#!/usr/bin/env bash
# Shared-z alignment at adaptation (wm/align.py, wm.adapt --anchor_align): per N, adapt B1 from its own babbling with
# --anchor_align 1.0 vs 0 (everything else equal), then the evaluation of b1_babble_variants.sh:
#   selection B1 with the realistic library (b1_babble_lib24) and the upper-bound library (rr_b1_clips_heldout),
#   goals = rr c10 held-out; B1 counterfactual read-out on rr_b1_branches_heldout (P = 11)
# PLUS the shared-latent numbers (body-ID accuracy, c10-fit Froude transfer to c08 / B1, kNN mixing) on the rr held-out
# dirs, with the adapted ITM of each arm.
# Adaptation recipe = b1_babble_sweep.sh (LoRA r8 on ITM + FTM, hinge, Froude anchor, 3000 steps) + plain z-fit projector,
# so the align-0 arm reproduces the sweep's recipe. FREEZE_FTM=1 -> LoRA on the ITM only; OBJ=rollout -> projector fitted
# through the rollout read (b1_babble_variants.sh A / C).
# Bank for --anchor_align: 48 rr_c10_clips_train clips (~2,900 transitions) through the UNADAPTED ITM; the projection head
# comes from the base checkpoint if it was pretrained with --lambda_align (round1_counterfactual.sh J), else a fresh head.
#   bash scripts/run/b1_babble_align.sh wm/runs/round1_hexonly_s0_rr/best.pt [N ...]      (default N = 44 88)
#   ALIGN=0.5 TAU=0.1 SIGMA=0.25 bash scripts/run/b1_babble_align.sh ...
cd "$(dirname "$0")/../.."
BASE=${1:?base checkpoint}; shift; NS=${*:-44 88}
ALIGN=${ALIGN:-1.0}; TAU=${TAU:-0.1}; SIGMA=${SIGMA:-0.25}; OBJ=${OBJ:-z}
PY=.venv/bin/python3; CW=data/counterfactual_walks; OUT=${OUT:-results/eval/b1_babble_align}; mkdir -p $OUT/ckpt
SEL=scripts/diagnostics/objective_experiments/selection_eval.py; RO=scripts/diagnostics/objective_experiments/counterfactual_readout.py
quiet() { grep -v -i "warn\|Loading weights\|sit still"; }
$PY tests/test_froude_labels.py > /dev/null || { echo "label tests FAIL: wrong code here"; exit 1; }
$PY tests/test_align.py > /dev/null || { echo "align tests FAIL"; exit 1; }
merge() { $PY - "$1" "$2" "$3" <<'PYEOF'
import sys, torch
from wm.models.action_projector import action_dims_from
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sv = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ck["projector"] = sv.get("projector", sv); ck["action_dims"] = action_dims_from(sv)
torch.save(ck, sys.argv[3])
PYEOF
}
# Embedding caches for the shared-latent figure. rr_b1_clips_heldout is in test_v4_b1_heldout.pt (selection below
# builds/refreshes it); rr_c10 / rr_c08 held-out had NO cache on 2026-10-06, so build them here if missing (encoder on GPU).
C10H=$CW/rr_c10_clips_heldout; C08H=$CW/rr_c08_clips_heldout; B1H=$CW/rr_b1_clips_heldout
C10C=results/wm/cache/rr_c10_clips_heldout.pt; C08C=results/wm/cache/rr_c08_clips_heldout.pt; B1C=results/wm/cache/test_v4_b1_heldout.pt
$PY - $C10H=$C10C $C08H=$C08C <<'PYEOF'
import glob, os, sys, torch
sys.path[:0] = [os.getcwd(), "scripts"]
from wm.data.emb_cache import load_cache, note, save_cache
from wm.data.embodiment import REGISTRY, load
from wm.evaluate import encode_clip
enc = None
for spec in sys.argv[1:]:
    d, c = spec.split("=", 1)
    cache = load_cache(c)
    miss = [p for p in sorted(glob.glob(os.path.join(os.getcwd(), d, "*.npz"))) if p not in cache]
    if not miss:
        continue
    print(f"NOTE: {len(miss)} clips of {d} not in {c}; encoding them now")
    if enc is None:
        from vjepa2_encoder import VJEPA2FrameEncoder
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
    for p in miss:
        cache[p] = encode_clip(enc, load(p, REGISTRY["hexapod"])["frames"], 2).cpu().half()
        note(cache, p)
    save_cache(cache, c)
PYEOF
ADAPT="--embodiment b1 --test_clips 4 --lambda_hinge 0.5 --hinge_margin 0.1 --lora_rank 8 --steps 3000 --anchor_froude 1.0
 --align_tau $TAU --align_sigma $SIGMA ${FREEZE_FTM:+--freeze_ftm}"
for N in $NS; do
  D=$CW/b1_babble_n$N
  {
  echo "=== N=$N  base=$BASE  align $ALIGN (tau $TAU sigma $SIGMA)  projector $OBJ  freeze_ftm ${FREEZE_FTM:-0}  $(date)"
  for a in 0 $ALIGN; do
    tag=al${a}_n$N
    $PY -m wm.adapt --ckpt $BASE --data $D --clips $N $ADAPT --anchor_align $a --out $OUT/ckpt/ad_$tag.pt 2>&1 \
        | quiet | grep -E "anchor_align|anchor bank|WARNING|^->"
    $PY -m wm.fit_projector --ckpt $OUT/ckpt/ad_$tag.pt --hex_dir $CW/rr_c10_clips_train --b1_dir $D \
        --cache_dir results/wm/cache/fitproj_files --objective $OBJ --out $OUT/ckpt/proj_$tag.pt 2>&1 | quiet | tail -1
    merge $OUT/ckpt/ad_$tag.pt $OUT/ckpt/proj_$tag.pt $OUT/ckpt/b1_$tag.pt
    echo "##### anchor_align $a"
    echo "--- selection B1, realistic library (babbling)"
    $PY $SEL --conditions all --candidates_dir $CW/b1_babble_lib24 --goal_dir $C10H \
        --cache results/wm/cache/sel_b1_babble_lib24.pt --windows 21 --ckpt $tag=$OUT/ckpt/b1_$tag.pt 2>&1 | grep -E "w=|bounds"
    echo "--- selection B1, upper-bound library (tuned clips)"
    $PY $SEL --conditions all --candidates_dir $B1H --goal_dir $C10H \
        --cache $B1C --windows 21 --ckpt $tag=$OUT/ckpt/b1_$tag.pt 2>&1 | grep -E "w=|bounds"
    echo "--- read-out B1 (rr branches)"
    $PY $RO --embodiment b1 --cf_dir $CW/rr_b1_branches_heldout --ckpt $tag=$OUT/ckpt/b1_$tag.pt --pairs 11 2>&1 | quiet | grep -A3 "^==="
    echo "--- shared latent (rr held-out: c10, c08, B1; adapted ITM)"
    $PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$B1H=$B1C" --c10 "$C10H=$C10C" --c08 "$C08H=$C08C" \
        --model "$tag=$OUT/ckpt/ad_$tag.pt" --out $OUT/shared_latent_$tag 2>&1 | quiet | tail -3
    rm -f $OUT/ckpt/ad_$tag.pt            # ~400 MB; b1_$tag.pt keeps the adapted weights + projector
  done
  } 2>&1 | tee $OUT/n$N.txt
done
echo ALIGN_DONE
