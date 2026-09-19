#!/usr/bin/env bash
# Corrected B1 chain for beh24 -- stage 1 (hexapod pretrain, best.pt) is unaffected and reused
# as-is. Only stages 2-4 (B1 adapt/projector/body_head) re-run, with --clips scaled from 9 to 18
# to restore the same ~75%-per-family adaptation coverage beh12 had.
#
# --stratify (wm/adapt.py's select_clips) cycles through families (speed/turn/side) picking one
# fresh CONDITION per round until --clips is reached -- it stratifies by family, not by condition.
# beh12 had 4 conditions/family: --clips 9 covered 3/4 = 75%. beh24 doubled every family to 8
# conditions (fwd+bwd, pos+neg): the same --clips 9 only covers 3/8 = 37.5%, a confound this
# session's original beh24 run didn't account for. --clips 18 restores 6/8 = 75%.
#
# Output goes to a NEW subdir (b1_adapt_clips18), not overwriting b1_adapt_clean, so both runs
# stay comparable.

# 2. Stage 1 — adapt to B1 (--clips 18, corrected)
.venv/bin/python3 -m wm.adapt \
  --ckpt wm/runs/beh24_hinge_cleansplit/best.pt \
  --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 --clips 18 --stratify \
  --lambda_hinge 0.5 --hinge_margin 0.1 \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clips18/adapted_b1.pt

# 3. Stage 2 — fit projector
.venv/bin/python3 -m wm.fit_projector \
  --ckpt wm/runs/beh24_hinge_cleansplit/b1_adapt_clips18/adapted_b1.pt \
  --hex_dir data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
  --b1_dir data/egocentric/beh24_b1_ego_flat_cleantrain \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clips18/projector_b1.pt

# 4. Stage 4 — fit body_head (with hexapod rehearsal)
.venv/bin/python3 -m wm.fit_body_head \
  --ckpt wm/runs/beh24_hinge_cleansplit/b1_adapt_clips18/adapted_b1.pt \
  --data data/egocentric/beh24_b1_ego_flat_cleantrain --embodiment b1 \
  --also hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain \
  --out wm/runs/beh24_hinge_cleansplit/b1_adapt_clips18/body_head_b1_hex_clean.pt

# After this: re-run eval_body_head_true_heldout.py with --conditions restricted to the shared 8
# conditions (same command as before, --ckpt pointed at b1_adapt_clips18/body_head_b1_hex_clean.pt
# instead of b1_adapt_clean/...) and compare against both the beh12 checkpoint (0.769/0.864,
# raw MSE 0.00112/0.00120) and beh24's original --clips 9 checkpoint (1.499/1.468, raw MSE
# 0.00234/0.00201). If --clips 18 recovers most of the gap, the --clips-9 confound explains the
# regression; if it doesn't, the interference is real and independent of adaptation coverage.
