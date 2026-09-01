#!/usr/bin/env bash
#
# STEP 1 of the egocentric plan: does a world model TRAINED on egocentric USE the action?
#
#   bash scripts/step1_egocentric.sh collect     # here, CoppeliaSim GUI, ONE instance, ~10 min
#   bash scripts/step1_egocentric.sh train       # on com7, under tmux, ~6 h
#   bash scripts/step1_egocentric.sh measure     # anywhere the checkpoint is
#
# **Q1 showed the signal exists. This asks whether a trained model uses it.** F155 through F169 are
# a record of signals that existed and were then ignored by a trained predictor, so the two are not
# the same question and only this one decides whether anything downstream has ground to stand on.
#
# **Three gates, in order. Each one can stop the run, and none is a report to read afterwards.**
#
#   GATE A  appearance leak      colour must not predict heading, or Q1 and everything after it
#                                measure a landmark instead of the view
#   GATE B  null separability    `ITM(e_t, e_t)` must still be a *different vector* from a real
#                                transition. **If it is not, `null/real` compares a thing with
#                                itself** and a ratio near 1.0 would mean nothing -- an F160-shaped
#                                confound. This cannot be checked before training
#   GATE C  the question         `null/real` on the HELD-OUT body, against F157's allocentric 1.03
#
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
MORPH=c08f09t09=medauroidea_c08f09t09.ttt
RAW=data/egocentric/beh12_c08f09t09_ego_raw
HELD=data/egocentric/beh12_c08f09t09_ego_flat
TRAIN_HEX=data/egocentric/beh12_c10f10t10_ego_flat
TRAIN_B1=data/egocentric/beh12_b1_ego_flat
RUN=wm/runs/beh12_ego

case "${1:-}" in

collect)
  # **The held-out body, egocentric.** Every number this project decides on is measured on
  # `c08f09t09`, which no run trains on -- F159's 0.779, F155's 1.03, every rollout figure. Without
  # its egocentric twin, GATE C would be read on a body the model was trained on and could not be
  # compared with the allocentric baseline at all.
  #
  # `--spin_sign -1` reproduces the dataset's turn direction. **The default is +1 and produces turns
  # the other way round**, which is F117's week-long error waiting to happen again.
  #
  # `--ego_seed 0` with `--repeats 4` gives room = repeat index, so the four clips of every
  # condition sit in four different rooms and every room hosts all twelve behaviours. **The same
  # rule the two training sets already follow**, which is what makes a slot pair across bodies.
  [ -d "$HELD" ] && { echo "$HELD exists; nothing to collect"; exit 0; }
  $PY scripts/dataset/collect_beh12.py --morph "$MORPH" --out "$RAW" --repeats 4 \
      --spin_sign -1 --extra --view egocentric --ego_seed 0
  $PY scripts/dataset/merge_behaviour_dirs.py --src "$RAW" --out "$HELD" --embodiment hexapod

  echo
  echo "############ GATE A -- appearance leak ############"
  echo "colour must not predict heading. Exits non-zero on a leak; add seeds and re-collect."
  $PY scripts/diagnostics/check_appearance_leak.py --data "$HELD" --embodiment hexapod

  echo
  echo "look at a clip before training on it:"
  echo "  $PY sim/render/npz_to_video.py --data $HELD --out results/egocentric"
  ;;

train)
  # **Input swap only -- confirmed, not assumed.** ITM, FTM, MotionDecoder and the body head take
  # `e_t` and an action and carry no camera assumption; `center_embeddings` is false so no fixed-view
  # statistics are baked in; the encoder cache is keyed by absolute path so egocentric and
  # allocentric embeddings cannot be mixed. Everything below except `--sources` matches
  # `com7_pretrain_body3.sh` exactly, so the comparison against the allocentric run is like for like.
  [ -f "$RUN/best.pt" ] && { echo "skip, $RUN/best.pt exists"; exit 0; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_ego \
    --sources hexapod="$TRAIN_HEX" b1="$TRAIN_B1" \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2
  ;;

measure)
  echo "############ GATE B -- null separability, on the egocentric-trained ITM ############"
  echo "The pre-training reading of 0.952 / 0.982 used the ALLOCENTRIC ITM on egocentric embeddings,"
  echo "which is out of distribution. This is the reading that counts."
  echo "Reference: the allocentric checkpoint reads 0.903 overall, 0.922 on its worst family."
  for SPEC in "hexapod $HELD nullsep_ego_hex" "b1 $TRAIN_B1 nullsep_ego_b1"; do
    set -- $SPEC
    $PY scripts/diagnostics/null_separability.py --ckpt "$RUN/best.pt" \
        --data "$2" --embodiment "$1" --cache "results/wm/cache/$3.pt"
  done

  echo
  echo "############ GATE C -- does the trained model USE the action? ############"
  echo "Allocentric baseline, same measurement, same held-out body: null/real = 1.03 (F155, F157)."
  echo "PASS: clearly above ~1.10 on both bodies.  FAIL: near 1.03 -- the viewpoint was necessary"
  echo "and not sufficient, which is reportable as it stands."
  $PY scripts/diagnostics/action_necessity.py --ckpt "$RUN/best.pt" \
      --data "$HELD" --embodiment hexapod --lags 1 2 3 5
  $PY scripts/diagnostics/action_necessity.py --ckpt "$RUN/best.pt" \
      --data "$TRAIN_B1" --embodiment b1 --lags 1 2 3 5

  echo
  echo "############ context, not gates ############"
  $PY scripts/diagnostics/body_head_calibration.py --ckpt "$RUN/best.pt" \
      --data hexapod="$TRAIN_HEX" b1="$TRAIN_B1"
  ;;

*)
  echo "usage: $0 {collect|train|measure}"; exit 2 ;;
esac
