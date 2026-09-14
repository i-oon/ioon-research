#!/bin/bash
# Coppelia-native B1 babble, spring joint control (F209/F210): the frozen forward-trot point
# (5.0 Hz, amplitude 0.30) reaches Froude 0.220-0.226, matching MuJoCo's own CPG on the same robot
# (F209) -- against 0.140 under Coppelia's default position-PID joints, which is missing the
# per-joint viscous damping Bullet has no parameter for. `--joint-control spring` supplies it via
# `jointdynctrl_spring` (tau = K(q*-q) - C*qdot) with K/C read from `b1_flat_real.xml`'s own
# system-identified numbers -- no engine swap, no reduced timestep, no custom torque loop.
#
# **Sampling range declared here, before running, justified by uprightness alone -- never by
# Froude coverage against the hexapod goal, which is what F209's whole-day search got wrong before
# these numbers were thrown out and re-measured properly (n=3 per config, not n=1).**
#
#   frequency  fixed at 5.0 Hz      -- the only frequency with margin below its own fall boundary:
#                                       5.0/amp0.22-0.30 all 3/3 upright; 5.0/amp0.38 0/3; 6.0 Hz
#                                       is both slower AND less tolerant (amp0.30 falls at 6.0 Hz
#                                       where it stands at 5.0 Hz) -- raising it buys nothing.
#   amplitude  uniform [0.20, 0.32] -- inside the tested-safe 0.22-0.30 band (3/3 upright at both
#                                       ends, n=3 each) with a small margin trimmed off the observed
#                                       edge (0.30 stood every trial but 0.38 fails hard and cliffs,
#                                       not a gradual falloff), sampled by the script's own
#                                       `--generic-amp-min/-max`, not hand-picked per clip.
#   noise      0.03 (file default)  -- what makes this babble rather than a fixed CPG point.
#   duty/coupling/clearance/hip     -- held at the values already verified stable at this frequency
#                                       (duty 0.55, coupling 0.6, clearance 1.1, hip_ratio 0.0,
#                                       stance_calf_bias -1.2); their own stable ranges are
#                                       unexplored and that is a stated scope limit, not a claim
#                                       they are optimal.
#
# Every rollout is retained, including falls, per this file's own stated policy -- the amplitude
# range above was chosen for a low fall rate, not a zero one, and a fall is real dynamics data.
#
# Usage:
#   bash scripts/run/b1_babble_coppelia_spring.sh 2>&1 | tee results/wm/dataset/b1_babble/coppelia_spring_log.txt
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python3
OUT=data/egocentric/b1_babble_coppelia_spring_ego_flat
N_CLIPS=40

mkdir -p "$OUT"
for seed in $(seq 0 $((N_CLIPS - 1))); do
  echo "=== clip $seed / $((N_CLIPS - 1)) ==="
  $PY sim/collect/collect_b1_coppelia_babble.py \
    --seed "$seed" --ego --ego-seed "$seed" \
    --joint-control spring \
    --cpg-mode generic --generic-phase-layout trot --generic-gait-shape coupled-duty \
    --generic-trot-pairing diagonal --generic-thigh-sign-layout same \
    --generic-coupling-ratio 0.6 --generic-clearance-ratio 1.1 --generic-hip-ratio 0.0 \
    --generic-stance-calf-bias -1.2 --generic-duty-factor 0.55 \
    --generic-frequency 5.0 --generic-amp-min 0.20 --generic-amp-max 0.32 \
    --out "$OUT/b1_babble_s${seed}.npz"
done
echo "done: $N_CLIPS clips in $OUT"
