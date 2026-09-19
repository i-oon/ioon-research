#!/usr/bin/env bash
# beh24 counterpart of clean_retrain.sh -- same four commands, same hyperparameters, only the
# --sources/--data/--also paths change (beh24's 48/24/24 split instead of beh12's 24/12/12).
# Run manually, one at a time.

# 1. Fresh clean pretrain (hexapod only). 48 train / 24 val / 24 test (val and test disjoint,
#    stratified, 1-per-condition each -- see scripts/dataset/make_clean_split_beh24.py).
.venv/bin/python3 -m wm.train \
  --sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
  --val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval \
  --lambda_recon 1.0 --lambda_motion 1.0 --lambda_body 0.5 \
  --lambda_hinge 0.5 --lambda_rollout 1.0 --lambda_readout 1.0 \
  --hinge_margin 0.1 --hinge_K 2 \
  --body_dim 3 --body_channels 0 1 2 \
  --epochs 50 --batch_size 8 --lr 0.0001 --seed 0 \
  --name beh24_hinge_cleansplit --out_dir wm/runs

# 2. Stage 1 — adapt to B1
.venv/bin/python3 -m wm.adapt \
  --ckpt wm/runs/beh24_hinge_cleansplit/best.pt \
  --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 --clips 9 --stratify \
  --lambda_hinge 0.5 --hinge_margin 0.1 \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/adapted_b1.pt

# 3. Stage 2 — fit projector
.venv/bin/python3 -m wm.fit_projector \
  --ckpt wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/adapted_b1.pt \
  --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
  --b1_dir data/egocentric/beh24_b1_ego_flat_cleantrain \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/projector_b1.pt

# 4. Stage 4 — fit body_head (with hexapod rehearsal). --ckpt is the STAGE 1 (adapt) output, not
#    stage 2's -- see clean_retrain.sh's own comment on this: only adapted_b1.pt carries
#    config/itm/ftm/md/body_stats, which this script needs unconditionally.
.venv/bin/python3 -m wm.fit_body_head \
  --ckpt wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/adapted_b1.pt \
  --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 \
  --also hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt

# NOTE on the real held-out number: wm.fit_body_head's own printed "held out" column is an
# internal --val_frac (default 0.2) split of whichever --data/--also pool it's given here -- NOT
# the external _cleanheldout set. This was the exact measurement bug F222 corrected for beh12.
# After stage 4, get the real number with:
#   .venv/bin/python3 scripts/diagnostics/objective_experiments/eval_body_head_true_heldout.py \
#     --ckpt wm/runs/beh24_hinge_cleansplit/b1_adapt_clean/body_head_b1_hex_clean.pt \
#     --embodiment b1 \
#     --train_dir data/egocentric/beh24_b1_ego_flat_cleantrain \
#     --heldout_dir data/egocentric/beh24_b1_ego_flat_cleanheldout \
#     --also_embodiment hexapod \
#     --also_train_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
#     --also_heldout_dir data/egocentric/beh24_c10f10t10_ego_flat_cleanheldout
