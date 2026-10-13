#!/usr/bin/env bash
# Does per-body centring help a NEW body? (2026-10-14) Two hexapod-only pretrains on the CURRENT recipe (grayscale +
# photometric augmentation + imagined-read loss, = round1_branches_s0_gray_imread without B1), differing ONLY in
# center_embeddings; B1 is then a new body, adapted with the standard recipe (eval_suite.sh hexonly: LoRA r8, 3000
# steps, 44 b1_clips_train), its centring offset computed by wm.adapt from those adaptation clips alone (unlabelled).
#   1. round1_hexonly_s0_gray_imread         c10 clips + branches, 6 epochs (= 30,528 steps, round-1 budget)
#   2. round1_hexonly_s0_gray_imread_center  the same + --center_embeddings True
#   3. eval_suite.sh NAME hexonly for both, then z_channel_transfer (shared-latent read-out per channel) for both
# Compare with each other (the centring effect on a new body) and with round1_branches_s0_gray_imread(_center) (B1 in
# pretraining). One GPU job at a time; gpu_guard pauses training above 83 C. Run from the repo root on the GPU PC:
#   nohup bash scripts/run/hexonly_center_pair.sh > results/wm/logs/hexonly_center_pair.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
CW=data/counterfactual_walks
grep -q "new-body offset" wm/adapt.py || { echo "STOP: wm/adapt.py has no new-body offset (git pull first)"; exit 1; }
grep -q "def offset_as" wm/evaluate.py || { echo "STOP: wm/evaluate.py has no offset_as (git pull first)"; exit 1; }
RECIPE="--grayscale True --aug_brightness 0.0 --aug_contrast 0.4 --aug_blur 2.6 --aug_noise 0.03 --aug_prob 0.5
 --aug_bright_mult 3.0 --aug_clean 0.25 --lambda_imread 1.0 --imread_k 4"

train() {  # RUN_TAG, extra flags
  local name=round1_hexonly_s0$1
  RUN_TAG=$1 EXTRA="$RECIPE $2" bash scripts/run/round1_counterfactual.sh H || exit 1
  sleep 30
  while pgrep -f -- "--name $name\$" > /dev/null || pgrep -f -- "--name $name " > /dev/null; do sleep 120; done
  [ -s wm/runs/$name/best.pt ] || { echo "STOP: $name has no best.pt (see results/wm/logs/$name.log)"; exit 1; }
  grep -E "^epoch" results/wm/logs/$name.log | tail -2
}

train _gray_imread ""
train _gray_imread_center "--center_embeddings True"

for name in round1_hexonly_s0_gray_imread round1_hexonly_s0_gray_imread_center; do
  echo "=== eval $name $(date)"
  bash scripts/run/eval_suite.sh $name hexonly wm/runs/$name/best.pt
  $PY scripts/diagnostics/cross_embodiment/z_channel_transfer.py --model $name=results/eval/$name/ckpt/adapted_b1.pt \
      --c10 $CW/c10_clips_heldout=results/wm/cache/test_v4_hex_heldout.pt \
      --c08 $CW/c08_clips_heldout=results/wm/cache/test_v4_c08_heldout.pt \
      --b1 $CW/b1_clips_heldout=results/wm/cache/test_v4_b1_heldout.pt 2>&1 | grep -v -i warn | tail -20
done
echo "=== summary"
for name in round1_hexonly_s0_gray_imread round1_hexonly_s0_gray_imread_center; do
  echo "--- $name"; cat results/eval/$name/summary.txt; done
echo HEXONLY_CENTER_PAIR_DONE
