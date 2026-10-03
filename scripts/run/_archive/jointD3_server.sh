#!/usr/bin/env bash
# Server job (2026-09-30): the jointD matched pair again, with the B1 on the v3 re-render (rendering matched
# to the hexapod's; F275 addendum, F285). Joint hexapod + B1 pretraining on the current pipeline (D: Froude head
# shapes z, no joint-command motion decoder), similarity loss ON vs OFF -- the matched pair.
#   GPU 0: jointD3_sim_s0    --lambda_sim 0.05
#   GPU 1: jointD3_nosim_s0  --lambda_sim 0
# Everything else identical: beh24 clean-train for both bodies (48 + 48), stride 5, hinge / read-out /
# rollout, --detach_body_z False, --lambda_motion 0, seed 0. TRAINING ONLY (~11 h each, in parallel).
# Needs on the server: updated wm/ (config.py + train.py with lambda_sim), and
#   data/egocentric/beh24_c10f10t10_ego_flat_clean{train,val}, data/egocentric_v3/beh24_b1_ego_flat_clean{train,val}
#   (copy with rsync -aL: symlinks). Bring back per run: wm/runs/<name>/best.pt, config.yaml, and the log.
#   bash scripts/run/jointD3_server.sh       (starts both in the background and returns)
set -uo pipefail
cd "$(dirname "$0")/../.."
for p in data/egocentric/beh24_c10f10t10_ego_flat_cleantrain data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
         data/egocentric_v3/beh24_b1_ego_flat_cleantrain data/egocentric_v3/beh24_b1_ego_flat_cleanval; do
  [ -s "$(ls $p/*.npz 2>/dev/null | head -1)" ] || { echo "MISSING/EMPTY $p"; exit 1; }
done
grep -q lambda_sim wm/config.py || { echo "wm/config.py lacks lambda_sim: copy the updated wm/ code first"; exit 1; }
mkdir -p results/wm/logs
i=0
for arm in "jointD3_sim_s0:0.05" "jointD3_nosim_s0:0"; do
  N=${arm%%:*}; W=${arm#*:}
  CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python3 -m wm.train \
    --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain b1=data/egocentric_v3/beh24_b1_ego_flat_cleantrain \
    --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval b1=data/egocentric_v3/beh24_b1_ego_flat_cleanval \
    --lambda_body 0.5 --detach_body_z False --lambda_motion 0 --lambda_hinge 0.5 --lambda_readout 1.0 \
    --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5 --epochs 50 \
    --checkpoint_every 2 --resume auto --lambda_sim $W --seed 0 --name $N > results/wm/logs/$N.log 2>&1 &
  echo "started $N (lambda_sim $W) on GPU $i (pid $!)"
  i=$((i + 1))
done
echo "after ~2 min: grep -E 'lambda_sim|lambda_motion|detach_body_z' wm/runs/jointD3_*/config.yaml"
echo "              grep '^epoch' results/wm/logs/jointD3_sim_s0.log | tail -1   (should show 'sim ... x-body pairs')"
