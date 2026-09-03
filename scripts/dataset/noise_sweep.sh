#!/usr/bin/env bash
#
# F164. The largest command noise that still leaves twelve behaviours -- and does it still work?
#
#   bash scripts/dataset/noise_sweep.sh          # CoppeliaSim GUI, exactly ONE instance
#
# F163 found the redundancy is data-side: 0.05 rad of correlated command noise took single-frame
# action R2 from 0.764 to 0.196 and doubled the pair-minus-single gap from +0.084 to +0.173. **And
# it destroyed the dataset**: 24 of 66 condition pairs fell below 2x their own spread, against 0 of
# 66 on the clean arm, so the twelve conditions stopped being twelve behaviours.
#
# **Separability gates R2 at every level.** A level that opens the gap by dissolving the behaviours
# has not found anything -- it has collected one noisy condition twelve times. Read the separability
# block first and only then the R2 table. This ordering is the point of the script.
#
# **A clean arm is recollected here even though F163 has one**, because F163's was a different
# sitting. If it reproduces 0.764 / +0.084 that also retires the worry about comparing across
# sittings; if it does not, that is worth knowing before anything else is read.
#
# **What a pass proves and does not prove, fixed before the run:**
#
#   passes separability AND opens the gap
#       the MECHANISM is confirmed -- the redundancy is data-side and can be broken while the
#       behaviours stay intact. Direction B becomes "train on perturbed data", no encoder rebuild.
#
#   what it does NOT prove
#       that a world model trained on it is USEFUL. **The injected noise is random by construction,
#       so a model learns to read jitter**, and jitter does not help control. The gap opening is
#       **partly tautological**: a component was added that only the transition can carry.
#       **Mechanism is not usefulness.** No result from this script licenses a claim about a
#       working controller.
#
#   no level passes separability while opening the gap
#       random perturbation cannot break the redundancy without destroying the behaviours. B's
#       cheap data form is dead, and only meaningful perturbation or an encoder rebuild remain.
#
# **The standing next question, which this run is only a proxy for**: the tautology-free version is
# *meaningful* phase-breaking motion -- real turns, transitions and speed-breaks that are not
# phase-locked, where the action carries intent rather than jitter. This sweep is the fast test of
# whether phase-breaking works **at all**, before anyone invests in collecting that.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
MORPH=c10f10t10=medauroidea_c10f10t10.ttt
CKPT=wm/runs/beh12_hex-b1_body3/best.pt
LEVELS="0.0 0.02 0.03"

for N in $LEVELS; do
  TAG=$(echo "n$N" | tr -d '.')
  RAW="data/allocentric/beh12_c10f10t10_sweep${TAG}_raw"
  OUT="data/allocentric/beh12_c10f10t10_sweep${TAG}_flat"
  echo "=== collect cmd_noise $N -> $OUT"
  if [ -d "$OUT" ]; then echo "skip, exists"; continue; fi
  $PY scripts/dataset/collect_beh12.py --morph "$MORPH" --out "$RAW" --repeats 2 \
      --extra --cmd_noise "$N" --noise_tau 5
  $PY scripts/dataset/merge_behaviour_dirs.py --src "$RAW" --out "$OUT" --embodiment hexapod
done

echo
echo "############ SEPARABILITY -- read this before any R2 ############"
for N in $LEVELS; do
  TAG=$(echo "n$N" | tr -d '.')
  echo "--- cmd_noise $N"
  $PY scripts/dataset/collect_beh12.py --separability "data/allocentric/beh12_c10f10t10_sweep${TAG}_flat" || true
done

echo
echo "############ ACTION R2 -- only meaningful where separability passed ############"
for N in $LEVELS; do
  TAG=$(echo "n$N" | tr -d '.')
  echo "--- cmd_noise $N"
  $PY scripts/diagnostics/objective_experiments/inverse_dynamics_r2.py --ckpt "$CKPT" \
      --data "data/allocentric/beh12_c10f10t10_sweep${TAG}_flat" --embodiment hexapod \
      --cache "results/wm/cache/sweep${TAG}.pt" --stride 2
done

echo
echo "=== the nulls to read against: clean +0.084 gap at 0.764 single frame, 0.05 rad +0.173 at 0.196"
echo "=== done  $(date '+%F %T')   send back this whole log"
