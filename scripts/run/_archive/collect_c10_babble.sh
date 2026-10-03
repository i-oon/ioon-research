#!/usr/bin/env bash
# c10f10t10 babble for the switching-pretrain test: B = switching, C = steady control (same drive
# distribution, one drive per clip). 48 clips each, random rooms. Then flat dirs + review renders.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3
CEN=data/egocentric/beh24_c10f10t10_ego_flat/hexapod_ep10004.npz   # side_L_lvl1: --symmetric --ik_iters 8 pose
$PY scripts/dataset/collect_babble_hex.py --clips 48 --seed 2 --morph c10f10t10=medauroidea_c10f10t10.ttt \
    --centre_from $CEN --out data/egocentric/babble_c10f10t10_switch
$PY scripts/dataset/collect_babble_hex.py --clips 48 --seed 3 --steady --morph c10f10t10=medauroidea_c10f10t10.ttt \
    --centre_from $CEN --out data/egocentric/babble_c10f10t10_steady
for v in switch steady; do
  F=data/egocentric/babble_c10f10t10_${v}_flat; mkdir -p $F
  for d in data/egocentric/babble_c10f10t10_$v/clip*; do n=$(basename $d)
    ln -sf ../babble_c10f10t10_$v/$n/c10f10t10_babble_ep0.npz $F/hexapod_babble_$n.npz; done
  $PY scripts/figures/render_babble_review.py --babble data/egocentric/babble_c10f10t10_$v --out results/deck/babble_review_c10_$v
done
echo C10_BABBLE_DONE
