#!/bin/bash
# Q21/W15-1: clean, from-scratch, single-script redo of the B1 babble pipeline, so the exact
# data/checkpoint provenance behind the final 2x2x2 numbers is reproducible from this file alone,
# not reconstructed from memory across several ad-hoc runs the way this session's F190-F193
# numbers were. See doc/OPEN_QUESTION.md Q21 ("Status as of 2026-09-11") for why this redo exists.
#
# Usage:
#   bash scripts/run/b1_babble_clean_redo.sh 2>&1 | tee results/wm/dataset/b1_babble/clean_redo_log.txt
#
# Every stage is idempotent-ish (re-running overwrites its own output dir only) but NOT resumable
# step-by-step -- if it dies partway, re-run from the top rather than guessing which stage finished.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
S=results/wm/dataset/b1_babble/batch_clean   # pre-render, intermediate -- redundant once rendered
R=data/egocentric/b1_babble_ego_flat          # final, self-sufficient dataset (frames+actions+motion)
OUT=wm/runs/b1_babble_clean
HEXGOAL=data/egocentric/beh12_c10f10t10_ego_flat
EXPERT_CKPT=wm/runs/b1_adapt/body_head_b1.pt
EXPERT_DIR=data/egocentric/beh12_b1_ego_flat

echo "=== stage 0: collect a diverse babble batch (forward/lateral/yaw, F190's validated params) ==="
mkdir -p "$S"
for freq in 1.0 1.5 2.0; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq "$freq" --thigh_amp 1.0 --calf_amp 0.5 \
      --noise 0.03 --seed "$seed" --steps 100 --out "$S/fwd_f${freq}_s${seed}.npz"
  done
done
for freq in 0.8 1.0 1.2; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq "$freq" --thigh_amp 0.1 --strafe_amp 1.0 \
      --calf_amp 0.4 --noise 0.03 --seed "$seed" --steps 100 --out "$S/lat_f${freq}_s${seed}.npz"
  done
done
for pv in 1.2 1.5 1.8; do
  for seed in 0 1 2 3; do
    $PY sim/collect/collect_b1_cpg_babble.py --freq 1.0 --thigh_amp 0.6 --pivot_amp "$pv" \
      --calf_amp 0.6 --noise 0.03 --seed "$seed" --steps 100 --out "$S/yaw_p${pv}_s${seed}.npz"
  done
done
echo "collected $(ls "$S"/*.npz | wc -l) clips"

echo "=== stage 0b: render egocentric video for every clip (needed for stage 1's ITM/FTM fit) ==="
mkdir -p "$R"
for f in "$S"/*.npz; do
  $PY sim/render/render_b1_replay.py --scene sim/env/b1_flat.ttt \
    --traj "$f" --out "$R" --ego --align_yaw --ego_seed 0
done

echo "=== stage 1-2: wm.finetune_new_body (ITM/FTM adapt, then projector fit; stage 3 skipped "
echo "    per its own rollout-gap gate; default, non-rehearsed stage 4 also produced here) ==="
$PY -m wm.finetune_new_body \
  --base_ckpt wm/runs/beh12_hexonly_stopgrad/best.pt \
  --embodiment b1 --data "$R" --out_dir "$OUT"

echo "=== stage 4b: refit body_head WITH hexapod rehearsal (F186's mechanism -- confirmed "
echo "    necessary a second time in F191, not optional) ==="
$PY -m wm.fit_body_head \
  --ckpt "$OUT/teacher_b1.pt" --data "$R" --embodiment b1 \
  --also "hexapod=$HEXGOAL" --latent both --epochs 400 \
  --out "$OUT/body_head_b1_hex.pt"

echo "=== stage 5: the corrected, single-mechanism 2x2x2 (F193's fix -- windowed scoring "
echo "    throughout, fit/candidate pairing never crossed, per-family breakdown always shown) ==="
echo
echo "--- expert-fit + expert-candidates ---"
$PY scripts/diagnostics/objective_experiments/final_2x2x2_test.py \
  --pool expert --ckpt "$EXPERT_CKPT" --candidates_dir "$EXPERT_DIR" --per_condition 1 \
  --goal_dir "$HEXGOAL"

echo "--- babble-fit + babble-candidates (this run's own fresh checkpoint) ---"
$PY scripts/diagnostics/objective_experiments/final_2x2x2_test.py \
  --pool babble --ckpt "$OUT/body_head_b1_hex.pt" --candidates_dir "$R" \
  --goal_dir "$HEXGOAL"

echo "=== DONE ==="
echo "data:      $S, $R"
echo "checkpoints: $OUT/{adapted_b1,projector_b1,teacher_b1,body_head_b1,body_head_b1_hex}.pt"
echo "log:       whatever you piped this script's stdout/stderr to (see this file's Usage line)"
