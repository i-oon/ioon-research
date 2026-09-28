#!/usr/bin/env bash
# Step 1 of the detach test (plan 2026-09-28): does letting Froude supervision shape z (detach off)
# make z more action-driven or more state-driven, and does that depend on the data?
#
#                     detach on (L_body reads z)      detach off (L_body shapes z)
#   beh24             A, A2 (exist)                    dt0_beh24_s0 / _s1   <- this script
#   switch babble     babsw_stride5_s0 / _s1 (exist)   dt0_babsw_s0 / _s1   <- this script
#
# Everything else identical to switch_pretrain_arms.sh (stride 5, 48 clips, same losses, same
# evaluation: B1 pipeline + eta^2 grid, c08 zero-shot, counterfactual read-out, 5-step prediction).
# Needs GPU only (no simulator). Physics closed loops: scripts/run/detach_physics.sh, on a machine
# with CoppeliaSim.
#
# Server setup (from the workstation, repo root; -L copies symlink targets):
#   rsync -aL --relative <paths printed by: bash scripts/run/detach_2x2_step1.sh --list> ioon@SERVER:ioon-research/
#   plus the V-JEPA2 weights (~/.cache/huggingface/hub/models--facebook--vjepa2-vitg-fpc64-256)
# Run:  nohup bash scripts/run/detach_2x2_step1.sh > results/wm/logs/detach_2x2_step1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
NEED=(
  data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
  data/egocentric/beh24_c10f10t10_ego_flat_cleanval
  data/egocentric/babble_c10f10t10_switch_flat
  data/egocentric/beh24_b1_ego_flat_cleantrain
  data/egocentric/beh12_b1_ego_flat_cleantrain
  data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout
  data/egocentric/beh12_c08f09t09_ego_flat
  results/wm/cache/anova_hex_beh24val.pt
  results/wm/cache/selection_eval_cands.pt
  results/wm/cache/selection_eval_c08.pt
  results/wm/cache/counterfactual_horizon.pt
)
if [ "${1:-}" = "--list" ]; then printf "%s\n" "${NEED[@]}"; exit 0; fi
miss=0; for p in "${NEED[@]}"; do [ -e "$p" ] || { echo "MISSING $p"; miss=1; }; done
[ $miss = 0 ] || { echo "preflight failed: copy the paths above (see header)"; exit 1; }
[ -x .venv/bin/python3 ] || { echo "no .venv/bin/python3"; exit 1; }
mkdir -p results/wm/logs
PY=.venv/bin/python3
COMMON="--val_sources hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleanval --lambda_body 0.5 --lambda_hinge 0.5
 --lambda_readout 1.0 --lambda_rollout 1.0 --hinge_K 2 --body_dim 3 --body_channels 0 1 2 --frame_stride 5
 --epochs 50 --checkpoint_every 2 --resume auto"
eval "$(sed -n '/^arm() {/,/^}/p' scripts/run/switch_pretrain_arms.sh)"
BEH=hexapod=data/egocentric/beh24_c10f10t10_ego_flat_cleantrain
SW=hexapod=data/egocentric/babble_c10f10t10_switch_flat
arm dt0_beh24_s0 --sources $BEH --detach_body_z False
arm dt0_babsw_s0 --sources $SW  --detach_body_z False
arm dt0_beh24_s1 --sources $BEH --detach_body_z False --seed 1
arm dt0_babsw_s1 --sources $SW  --detach_body_z False --seed 1
echo DETACH_STEP1_DONE
