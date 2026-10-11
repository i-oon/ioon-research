#!/usr/bin/env bash
# ==========================================================================================
# SUPERSEDED -- do not run. This round used the v3 data (data/egocentric_v3/b1_cf_branches_*,
# hexapod beh24 flat dirs). It is replaced by the v4 plan (data/egocentric_v4/{hex,b1}_main_* and
# {hex,b1}_cf_*, see doc/STATUS.md). Kept only as a record of what was run.
# ==========================================================================================
# Round 1 on the server (2026-10-02): does training on counterfactual transitions fix the rollout?
# Both arms: joint c10 + B1 pretraining, Froude only (D: --lambda_motion 0, --detach_body_z False), no
# similarity loss, seed 0, corrected B1 data (F293) and labels (F295: 1 s edge-correct, switch-aware).
#   GPU 0: cf0_control_s0  c10 + B1 beh24 clean-train                                  (50 epochs)
#   GPU 1: cf1_b1cf_s0     the same + B1 counterfactual branches (3456 x 11 pairs from the branch point)
# Equal gradient-step budget: balanced batches make an epoch max(group) batches, so the branch set makes
# an epoch ~13x longer; 4 epochs of arm 1 ~ 41k steps vs ~36k for 50 epochs of arm 0.
# Needs on the server (rsync -aL, code first): wm/ scripts/ tests/, data/egocentric_v3/beh24_b1_ego_flat*,
#   data/egocentric_v3/b1_cf_branches_{train,val}; hexapod data/egocentric/beh24_c10f10t10_ego_flat_clean{train,val}.
#   bash scripts/run/cf_round1_server.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python3 tests/test_froude_labels.py || { echo "label tests FAIL: wrong code on the server"; exit 1; }
for p in data/egocentric/beh24_c10f10t10_ego_flat_cleantrain data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
         data/egocentric_v3/beh24_b1_ego_flat_cleantrain data/egocentric_v3/beh24_b1_ego_flat_cleanval \
         data/egocentric_v3/b1_cf_branches_train data/egocentric_v3/b1_cf_branches_val; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
.venv/bin/python3 -c "
import numpy as np,glob
f=glob.glob('data/egocentric_v3/b1_cf_branches_train/*.npz')[0]; d=np.load(f,allow_pickle=True)
assert 'segment' in d.files and 'first_pair' in d.files, 'branches not tagged'
f=glob.glob('data/egocentric_v3/beh24_b1_ego_flat/*.npz')[0]; d=np.load(f,allow_pickle=True)
assert 'pose_fix' in d.files, 'B1 data not F293-corrected'
print('data checks ok')" || exit 1
mkdir -p results/wm/logs
COMMON="--lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0
 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --lambda_sim 0 --seed 0
 --resume auto"
HEXT=hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
HEXV=hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python3 -m wm.train $COMMON --epochs 50 --checkpoint_every 2 \
  --sources $HEXT b1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain \
  --val_sources $HEXV b1=data/egocentric_v3/beh24_b1_ego_flat_cleanval \
  --name cf0_control_s0 > results/wm/logs/cf0_control_s0.log 2>&1 &
echo "started cf0_control_s0 on GPU 0 (pid $!)"
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python3 -m wm.train $COMMON --epochs 4 --checkpoint_every 1 \
  --sources $HEXT b1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain b1=data/egocentric_v3/b1_cf_branches_train \
  --val_sources $HEXV b1=data/egocentric_v3/beh24_b1_ego_flat_cleanval b1=data/egocentric_v3/b1_cf_branches_val \
  --name cf1_b1cf_s0 > results/wm/logs/cf1_b1cf_s0.log 2>&1 &
echo "started cf1_b1cf_s0 on GPU 1 (pid $!)"
echo "check after ~5 min: grep -E '^train|^epoch' results/wm/logs/cf*_s0.log | tail"
