#!/usr/bin/env bash
# Four commands, nothing else. Run manually, one at a time.

# 1. Fresh clean pretrain (hexapod only). 24 train / 12 val / 12 test (val and test are
#    disjoint, stratified, 1-per-condition each -- see scripts/dataset/make_clean_split.py).
#    --val_sources points wm.train at the pre-built val directory directly; --sources is then
#    used entirely for training, val_fraction is not used at all here.
.venv/bin/python3 -m wm.train \
  --sources hexapod=data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \
  --val_sources hexapod=data/egocentric/beh12_c10f10t10_ego_flat_cleanval \
  --lambda_recon 1.0 --lambda_motion 1.0 --lambda_body 0.5 \
  --lambda_hinge 0.5 --lambda_rollout 1.0 --lambda_readout 1.0 \
  --hinge_margin 0.1 --hinge_K 2 \
  --body_dim 3 --body_channels 0 1 2 \
  --epochs 50 --batch_size 8 --lr 0.0001 --seed 0 \
  --name beh12_hinge_cleansplit --out_dir wm/runs

# 2. Stage 1 — adapt to B1
.venv/bin/python3 -m wm.adapt \
  --ckpt wm/runs/beh12_hinge_cleansplit/best.pt \
  --data data/egocentric/beh12_b1_ego_flat_cleantrain --embodiment b1 --clips 9 --stratify \
  --lambda_hinge 0.5 --hinge_margin 0.1 \
  --out wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/adapted_b1.pt

# 3. Stage 2 — fit projector
.venv/bin/python3 -m wm.fit_projector \
  --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/adapted_b1.pt \
  --hex_dir data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \
  --b1_dir data/egocentric/beh12_b1_ego_flat_cleantrain \
  --out wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_b1.pt

# 4. Stage 4 — fit body_head (with hexapod rehearsal, per the established fix for the
#    catastrophic-forgetting regression)
.venv/bin/python3 -m wm.fit_body_head \
  --ckpt wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/projector_b1.pt \
  --data data/egocentric/beh12_b1_ego_flat_cleantrain --embodiment b1 \
  --also hexapod=data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \
  --out wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt
