#!/usr/bin/env bash
# Controlled before/after B1 physics loop (physics_rr_b1.sh settings), goal turn_s0.56 (ep40110) in its paired rr room;
# OLD vs NEW adapted ckpts (controlled_turn056_adapt.sh), direct + rollout x 3 episodes; random once; then the
# allocentric third-person re-render (render_b1_replay default camera, as b1_allo). Needs own CoppeliaSim on $PORT.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python3; PORT=${PORT:-25720}; NEP=${NEP:-3}
B=results/wm/closed_loop_rr/physics/controlled_turn056
CAND=data/counterfactual_walks/rr_b1_clips_heldout
goal=data/counterfactual_walks/rr_c10_clips_heldout/hexapod_ep40110.npz; room=$CAND/b1_ep50110.npz
run() {  # tag mech ckpt outdir
  r=$(nice -n 10 $PY sim/control/close_loop_b1_physics_froude.py --mechanism $2 --window 21 --ckpt $3 \
      --goal $goal --candidates_dir $CAND --rr_room $room --port $PORT --out $4 2>&1 \
      | tee -a $B/loop.log | grep -E "closed loop|Error|FELL|MAE|corr" | tr '\n' ' ')
  echo "$1 $2 turn_s0.56 | $r" | tee -a $B/summary.txt
}
for ep in $(seq 1 $NEP); do for m in old new; do for mech in direct rollout; do
  run "$m ep$ep" $mech $B/ckpt/${m}_b1.pt $B/$m/ep$ep
done; done; done
run "random ep1" random $B/ckpt/new_b1.pt $B/random/ep1
# allocentric third-person of episode 1 (render_b1_replay defaults, b1_flat scene; same as joint_rr/b1_allo)
for m in old new; do for mech in direct rollout; do
  tp=thirdperson; [ $mech = rollout ] && tp=thirdperson_rollout
  src=$B/$m/ep1/${mech}_w21/b1_hexapod_ep40110.npz; tmp=$B/$m/ep1/_traj; mkdir -p $tmp
  $PY - $src $tmp/${tp}_b1_hexapod_ep40110.npz <<'PYEOF'
import sys, numpy as np   # loop run -> render_b1_replay traj (base pose + SDK-ordered joints, 0.05 s)
d = np.load(sys.argv[1], allow_pickle=True)
J = [f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL") for s in ("hip", "thigh", "calf")]
np.savez(sys.argv[2], base_pos=d["base_pos"], base_quat=d["base_quat"], joint_pos=d["joint_pos"],
         joint_order_sdk=np.array(J), dt=np.float64(0.05))
PYEOF
  nice -n 10 $PY sim/render/render_b1_replay.py --port $PORT --scene sim/env/b1_flat.ttt \
      --traj $tmp/${tp}_b1_hexapod_ep40110.npz --out $B/$m/ep1 2>&1 | tail -1
  rm -rf $tmp
done; done
echo LOOP_DONE
