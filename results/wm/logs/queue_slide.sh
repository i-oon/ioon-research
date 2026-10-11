#!/usr/bin/env bash
# Slide storyline (2026-10-12): two evaluations only, no training. One GPU job at a time.
#  1. single-frame speed probe on GRAYSCALE frames (started separately; this waits for it)
#  2. shared latent (z transfer) for the grayscale models: build the missing grayscale c08 / refresh c10 caches via
#     selection_eval (the eval_suite step 2 commands), then shared_latent_figure (eval_suite step 3)
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; CW=data/counterfactual_walks; SEL=scripts/diagnostics/objective_experiments/selection_eval.py
while pgrep -f single_frame_speed_probe.py >/dev/null; do sleep 60; done
for N in round1_branches_s0_gray_imread round1_branches_s0_gray; do
  CK=results/eval/$N/ckpt; OUT=results/eval/$N
  $PY $SEL --conditions all --embodiment hexapod --candidates_dir $CW/c08_clips_heldout --goal_dir $CW/c10_clips_heldout \
      --cache results/wm/cache/test_v4_c08_heldout.pt --windows 21 --ckpt $N=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
  $PY $SEL --conditions all --embodiment hexapod --candidates_dir $CW/c10_clips_heldout --goal_dir $CW/c10_clips_heldout \
      --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --ckpt $N=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
  $PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$CW/b1_clips_heldout=results/wm/cache/test_v4_b1_heldout.pt" \
      --c10 "$CW/c10_clips_heldout=results/wm/cache/test_v4_hex_heldout.pt" --c08 "$CW/c08_clips_heldout=results/wm/cache/test_v4_c08_heldout.pt" \
      --model "$N=wm/runs/$N/best.pt" --out $OUT/shared_latent 2>&1 | grep -v Warn | tail -12 | tee $OUT/shared_latent.txt
done
echo QUEUE_SLIDE_DONE
