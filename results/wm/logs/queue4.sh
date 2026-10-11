#!/usr/bin/env bash
# physics loops, original rooms: _gray_imread vs control _gray (direct + rollout), c10 / c08 with --cpg_clock (+ gait-phase
# matched rollout), B1 (no phase match yet). Same scripts / goals / settings as F327.
cd /home/fibo07/ioon/ioon-research
for N in round1_branches_s0_gray_imread round1_branches_s0_gray; do
  O=results/wm/closed_loop_orig/physics/$N
  PORT=25762 BODY=c10 CK=results/eval/$N/ckpt/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
  sleep 180
  PORT=25762 BODY=c08 CK=results/eval/$N/ckpt/hex.pt OUT=$O bash scripts/run/physics_orig_hex.sh
  sleep 180
  PORT=25760 CK=results/eval/$N/ckpt/b1.pt OUT=$O bash scripts/run/physics_orig_b1.sh
  sleep 300
done
.venv/bin/python3 scripts/dataset/render_shift_heldout.py stop
echo QUEUE4_DONE
