#!/usr/bin/env bash
#
# Stage 3 on the **insect**, so the forward model reads the action on the robot F142 trains on.
#
#   bash scripts/com7_stage3_hexapod.sh          # on com7, about an hour
#
# **Why this is a prerequisite and not an optimisation.** F142's teacher ranks candidate actions by
# rolling the forward model. The insect-side pretrain rolls *well* -- state fidelity 0.757 at one
# step, 0.727 at two, both inside the bar -- and **does not read the action**: `/mean-z` across
# clips is **0.966**, a three percent effect. A teacher whose prediction does not move when the
# action changes labels noise.
#
# The quadruped side does not have this problem because it has already had a contrastive stage 3:
# the same measurement reads **0.476** there (F140). F119 established that the InfoNCE term is what
# repairs it. This applies the same term on the other robot.
#
# **The gate, checked below and before F142 uses the output:** `/mean-z` across clips must fall well
# under 1.0 while the state ratio stays under about 0.8. **If sensitivity arrives and fidelity goes
# past 0.8, stop** -- that is the trade F139 wrongly claimed and F140 refuted on the B1, and seeing
# it appear on the insect would mean the refutation does not generalise.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
. scripts/hex_stage3_clips.sh
RUN=wm/runs/beh12_hex-b1_body3
OUT=$RUN/stage3_hex_nce_s0.pt

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "=== stage 3 on the hexapod, contrastive  $(date '+%F %T')"
if [ -f "$OUT" ]; then echo "skip $OUT"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.adapt3 \
    --ckpt $RUN/best.pt --projector $RUN/projector_b1_adapted.pt \
    --data data/allocentric/beh12_c10f10t10_flat --embodiment hexapod --train_clips $HEX_CLIPS \
    --steps 15000 --lambda_nce 1.0 --batch 8 --seed 0 \
    --cache results/wm/cache/hex_c10.pt --out "$OUT"
fi

echo
echo "=== the gate: fidelity and action-sensitivity on a body it never trained on  $(date '+%F %T')"
$PY -u scripts/diagnostics/rollout_fidelity.py --ckpt "$OUT" \
  --data data/allocentric/beh12_c08f09t09_flat --embodiment hexapod \
  --cache results/wm/cache/fid_hexapod.pt --latent projector --mean_z --horizons 1 2 3 5

echo
echo "before: state 0.757 at h=1, /mean-z across clips 0.966 -- good rollout, deaf to the action."
echo "pass if /mean-z falls well below 1.0 and the state ratio stays under about 0.8."
echo "send back $OUT and this whole log."
