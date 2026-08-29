#!/usr/bin/env bash
#
# Rebuild the adaptation stack on the corrected B1 set: stage 1 (wm.adapt) then stage 2
# (wm.fit_projector). Stage 3 is scripts/com7_stage3_seeds.sh and reads both outputs.
#
#   bash scripts/com7_stage12_rebuild.sh          # from the repository root
#
# **Why it has to be rebuilt at all.** Every B1 checkpoint was deleted on 2026-08-29. The set they
# were adapted on had four defects -- the robot clipped by the image edge in 61% of frames, an
# unpinned camera giving every clip its own background, a forward clip filed as the weakest turn
# level, and turns running opposite to the insect's (F113-F115, F117). `data/beh12_b1_flat` is the
# corrected set; verified before writing this sheet that all three sets now turn the same way, with
# the four B1 turn levels at +0.0097 / +0.0248 / +0.0371 / +0.0759 dimensionless against the
# pretraining insect's +0.0029 / +0.0148 / +0.0353 / +0.0736.
#
# The pretrain it starts from, `wm/runs/beh12_hexonly/best.pt`, is hexapod-only and carries none of
# those defects -- it is not rebuilt here.
#
# **Most of the wall clock is encoding, not training.** `results/wm/cache/` holds no embedding
# caches, so stage 2 re-encodes both robots' beh12 directories through V-JEPA2, ~6,400 frames.
# Both caches are written to disk, so a second run of this sheet is minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python3
RUN=wm/runs/beh12_hexonly
ADAPTED=$RUN/adapted_b1.pt
PROJ=$RUN/projector_b1_adapted.pt

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
[ -f "$RUN/best.pt" ] || { echo "missing $RUN/best.pt -- the hexapod pretrain has to be here" >&2; exit 1; }
[ -d data/beh12_b1_flat ] || { echo "missing data/beh12_b1_flat" >&2; exit 1; }

echo
echo "=== stage 1: adapt the ITM and forward model to the B1  $(date '+%F %T')"
if [ -f "$ADAPTED" ]; then echo "skip $ADAPTED"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.adapt \
    --ckpt $RUN/best.pt \
    --data data/beh12_b1_flat --embodiment b1 --clips 9 \
    --out "$ADAPTED"
fi

echo
echo "=== stage 2: fit the action projector against the *adapted* ITM  $(date '+%F %T')"
# Against the adapted checkpoint, not the pretrain: stage 1 moved what `z` means.
if [ -f "$PROJ" ]; then echo "skip $PROJ"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.fit_projector \
    --ckpt "$ADAPTED" \
    --hex_dir data/beh12_c10f10t10_flat --b1_dir data/beh12_b1_flat \
    --cache results/wm/cache/beh12_embeddings.pt \
    --out "$PROJ"
fi

echo
echo "=== done  $(date '+%F %T')"
echo "next: bash scripts/com7_stage3_seeds.sh   (six checkpoints, three seeds x two objectives)"
