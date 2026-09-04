#!/usr/bin/env bash
#
# FTM(e_t, z) -> Delta-state, warm-started from the egocentric teacher, then the ranking test.
#
#   bash scripts/run/com7_state_head.sh          # on com7, under tmux
#
# **The chain this closes, in order.** F179: fine-ranking through the embedding rollout sits at
# 47%, coin-flip. target_action_share.py: z carries body motion at ridge R2 0.359 and the embedding
# at 0.005 -- the recon target buries the action, the shared coordinate does not. Three identity-
# removal attempts on the state head's pooled-delta input (delta-vs-frame, frozen offset, periodic
# recompute) all failed or were unnecessary: cross_embodiment_swap.py proved the leak is cosmetic
# (wrong-embodiment offset changes ranking R2 by nothing, 0.059=0.059). The number that looked like
# a dead end -- R2 ~0.03-0.15 under every regularised/frozen/moving-FTM control -- was a units bug:
# state_loss trained against RAW body_motion, R2 evaluated against STANDARDISED truths, compared
# directly. Fixed (train.py now reads `body_motion` through `train_set.body_stats`, already
# standardised, and this run's own state_offsets asserts the scale before training starts): the
# SAME controls read R2 0.81-0.85, both embodiments, frozen or moving FTM, matching the offline
# ridge oracle (0.852). That is validated. This is the untested step: does it RANK.
#
# **Warm-started, not trained from scratch.** Every prior run in wm/runs/ trains from scratch;
# everything measured today warm-started from teacher_ego.pt's converged (ITM, FTM). Training this
# from scratch is an untested variant and is deliberately not what this run does.
#
# **The pass bar, fixed before the run so it cannot be moved afterwards.**
#   * F179's local arm, properly powered: n>=40 branch points (SD ~8pts, not 13), F179's own goal
#     clip (hexapod_ep100, speed_c7.1), planning z (proj(a), never ITM(e_t,e_t+1)), --repeat_control 4
#   * f179-scorer (embedding rollout) vs state-head scorer, side by side, same branch points
#   * state-head scorer clearly above BOTH 53% (this session's f179 baseline) and 47% (F179's own
#     number) -> the direction holds to planning, write it up
#   * state-head scorer ~= f179/coin, even with R2 0.81-0.85 offline -> the wall is between
#     reconstruction accuracy and ranking, not the target; report that as the finding
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python3
RUN=wm/runs/beh12_state
BASE=wm/runs/beh12_ego/teacher_ego.pt

echo "=== repository $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

echo
echo "=== train: + StateHead, lambda_state, warm-started from teacher_ego.pt  $(date '+%F %T')"
# lambda_state 1.5, not 0.5: a 3-dim target competing against lambda_recon's 360,448-dim one needs
# its own weight, not the 0.5 that already loses to recon on the embedding path (this is exactly
# the drowning target_action_share.py measured). Watch it via loss_gradient_balance.py, not trusted
# as a number picked in advance.
if [ -f "$RUN/best.pt" ]; then echo "skip $RUN/best.pt"; else
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u -m wm.train \
    --name beh12_state \
    --sources hexapod=data/egocentric/beh12_c10f10t10_ego_flat b1=data/egocentric/beh12_b1_ego_flat \
    --lambda_body 0.5 --body_dim 3 --body_channels 0 1 2 \
    --lambda_state 1.5 \
    --init_ckpt "$BASE" \
    --epochs 50
fi

echo
echo "=== checkpoint selection, three files, read the right one for the right question  $(date '+%F %T')"
echo "  $RUN/best.pt        -- selection = recon+motion only, comparable to every other run"
echo "  $RUN/best_motion.pt -- pure joint-command accuracy"
echo "  $RUN/best_state.pt  -- the epoch to AUDIT the state head at; can differ from best.pt's epoch"
echo "  the ranking test below runs against best_state.pt -- it is the state head's own optimum,"
echo "  not necessarily the checkpoint with the best reconstruction"

echo
echo "=== gradient balance: did lambda_state actually get gradient, or did recon dominate  $(date '+%F %T')"
$PY scripts/diagnostics/objective_experiments/loss_gradient_balance.py --ckpt "$RUN/best_state.pt" \
  --data hexapod=data/egocentric/beh12_c10f10t10_ego_flat b1=data/egocentric/beh12_b1_ego_flat

echo
echo "=== the ranking test. CoppeliaSim must be up, GUI, one instance, insect scene loaded  $(date '+%F %T')"
echo "  medauroidea_c08f09t09.ttt on port 23000 -- see doc/SIM_GUIDE.md"
echo "  this does not launch CoppeliaSim itself: start it, confirm the connection, THEN run this file"
read -p "CoppeliaSim ready on port 23000 with medauroidea_c08f09t09.ttt? [y/N] " ok
[ "$ok" = "y" ] || { echo "stopping -- start the sim first"; exit 1; }

$PY -u scripts/diagnostics/planning/rank_fine_three_ways.py \
  --teacher "$RUN/best_state.pt" \
  --states 40 \
  --goal_clip data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz \
  --repeat_control 4

echo
echo "=== done  $(date '+%F %T')"
echo "send back $RUN/{best.pt,best_state.pt,config.yaml} and the ranking test's printed output"
echo "read the pass bar above BEFORE reading the numbers, not after"
