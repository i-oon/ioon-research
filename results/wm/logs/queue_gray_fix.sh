#!/usr/bin/env bash
# 2026-10-12: rerun OFFLINE SELECTION for the grayscale models after the selection_eval grayscale-order fix (rollout and
# vision-goal scores had been computed on colour frames). Exact eval_suite.sh step-2 commands; old selection.txt kept as
# selection_colourbug.txt. Then the shared-latent step (eval_suite step 3, best.pt as in joint mode). One GPU job at a time.
cd /home/fibo07/ioon/ioon-research
PY=.venv/bin/python3; CW=data/counterfactual_walks; SEL=scripts/diagnostics/objective_experiments/selection_eval.py
LIB=$CW/b1_clips_heldout; GOALS=$CW/c10_clips_heldout; C08=$CW/c08_clips_heldout; HEXT=$CW/c10_clips_heldout
LIBC=results/wm/cache/test_v4_b1_heldout.pt
for NAME in round1_branches_s0_gray_imread round1_branches_s0_gray; do
  OUT=results/eval/$NAME; CK=$OUT/ckpt
  [ -f $OUT/selection_colourbug.txt ] || cp $OUT/selection.txt $OUT/selection_colourbug.txt
  {
  echo "--- selection B1, recorded goal"
  $PY $SEL --conditions all --candidates_dir $LIB --goal_dir $GOALS --cache $LIBC --windows 21 --ckpt $NAME=$CK/b1.pt 2>&1 | grep -E "w=|bounds|grayscale"
  echo "--- selection B1, vision-read goal"
  $PY $SEL --conditions all --candidates_dir $LIB --goal_dir $GOALS --cache $LIBC --windows 21 --goal_source vision \
      --goal_cache results/wm/cache/test_v4_goals_c10_heldout.pt --ckpt $NAME=$CK/b1.pt 2>&1 | grep -E "w=|bounds"
  echo "--- selection c08 zero-shot"
  $PY $SEL --conditions all --embodiment hexapod --candidates_dir $C08 --goal_dir $GOALS \
      --cache results/wm/cache/test_v4_c08_heldout.pt --windows 21 --ckpt $NAME=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
  echo "--- selection c10 same body"
  $PY $SEL --conditions all --embodiment hexapod --candidates_dir $HEXT --goal_dir $GOALS \
      --cache results/wm/cache/test_v4_hex_heldout.pt --windows 21 --ckpt $NAME=$CK/hex.pt 2>&1 | grep -E "w=|bounds"
  } 2>&1 | tee $OUT/selection.txt
  VJEPA_GRAY=1 $PY scripts/figures/shared_latent_figure.py --device cuda --b1 "$LIB=$LIBC" --c10 "$HEXT=results/wm/cache/test_v4_hex_heldout.pt" \
      --c08 "$C08=results/wm/cache/test_v4_c08_heldout.pt" --model "$NAME=wm/runs/$NAME/best.pt" --out $OUT/shared_latent 2>&1 \
      | grep -v -i "warn\|loading" | tail -12 | tee $OUT/shared_latent.txt
done
echo QUEUE_GRAY_FIX_DONE
