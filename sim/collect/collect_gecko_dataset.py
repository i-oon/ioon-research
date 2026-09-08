"""Drive gecko (CPG, optionally babble-noised) and RECORD to .npz -- the piece
`collect_gecko_cpg.py` explicitly does not do (interactive-only, no saving).

  .venv/bin/python3 sim/collect/collect_gecko_dataset.py --out data/gecko/cpg_0.npz
  .venv/bin/python3 sim/collect/collect_gecko_dataset.py --mode babble --noise 0.0137 \\
      --bias 0.02 --seed 1 --out data/gecko/babble_1.npz

Matches the field contract `wm/data/embodiment.py`'s `_gecko()` reader expects: `frames`,
`action` (16-D, ACTIVE_JOINTS order from `sim/scene/build_gecko_scene.py` -- leg-major,
joint1..4, LEG_ORDER=[lf,rf,lh,rh]), `base_pos`, `base_quat`, `dt`.

**Fall threshold, measured not guessed.** A real CPG walk oscillates base z between 0.051 and
0.086 m (settled standing height 0.051 m -- this is a small robot). `FALL_HEIGHT=0.03` sits
clearly below that whole oscillation band, so it only fires on genuine collapse, not normal gait
bounce.

**Babble mode**: noise added on top of the same working CPG (never from-scratch random targets --
a from-scratch random policy has no reason to stay upright, matching the reasoning already
established for B1's `recollect_b1_noisy.py`). `--bias` adds a small per-rollout constant offset
to swing amplitude (not just per-step noise), since per-step-only noise around one fixed gait
tends to average out and babble specifically needs to SPAN a range of outcomes, not just add
jitter to one outcome.
"""
import argparse
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
from collect_gecko_cpg import (  # noqa: E402
    DUTY, FREQ, LEGS, LIFT_AMP, PHASE, SCENE, SWING_AMP, SWING_SIGN, leg_trajectory,
)

LEG_ORDER = ["lf", "rf", "lh", "rh"]
ACTIVE_JOINTS = [f"joint{j}_{leg}" for leg in LEG_ORDER for j in (1, 2, 3, 4)]
LEFT_LEGS = {"lf", "lh"}   # differential-drive turn bias: LEFT +turn_bias, RIGHT -turn_bias
DT = 0.02
FALL_HEIGHT = 0.03   # measured: real CPG walk oscillates z in [0.051, 0.086], this sits below it
CAM_NAME = "vjepa_cam"


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--mode", choices=["cpg", "babble"], default="cpg")
    ap.add_argument("--steps", type=int, default=66, help="recorded frames, matching this "
                    "project's usual clip length convention")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds before recording starts")
    ap.add_argument("--noise", type=float, default=0.0, help="per-step Gaussian noise on the "
                    "commanded joint target (babble mode); anchor to B1's own verified-safe "
                    "0.0137 ceiling, do not exceed it blind")
    ap.add_argument("--bias", type=float, default=0.0, help="per-ROLLOUT constant swing-amplitude "
                    "offset (babble mode), SYMMETRIC across left/right -- calibration found this "
                    "alone barely moves y_drift/yaw (near-zero range); mostly changes forward speed")
    ap.add_argument("--turn_bias", type=float, default=0.0, help="per-ROLLOUT ASYMMETRIC bias -- "
                    "LEFT legs (lf,lh) get +turn_bias, RIGHT legs (rf,rh) get -turn_bias, on top "
                    "of --bias. Differential drive, the standard generic way to induce turning in "
                    "a legged robot, not gecko-specific tuning. This is what actually produces "
                    "real y_drift/yaw variation -- --bias alone does not (calibration, this session)")
    ap.add_argument("--freq", type=float, default=FREQ, help="gait frequency, Hz -- also affects "
                    "achieved yaw range per the calibration sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    sim.loadScene(SCENE)
    sim.setStepping(True)

    cam = sim.getObject("/" + CAM_NAME)
    root = sim.getObject("/geckobotiv")
    handles, bias2 = {}, {}
    for leg in LEGS:
        for j in (1, 2, 3, 4):
            handles[(j, leg)] = sim.getObject(f"/joint{j}_{leg}")

    sim.startSimulation()
    for leg in LEGS:
        bias2[leg] = sim.getJointPosition(handles[(2, leg)])
    for _ in range(int(args.settle / DT)):
        sim.step()

    rng = np.random.default_rng(args.seed)
    base_bias = args.bias if args.mode == "babble" else 0.0
    turn_bias = args.turn_bias if args.mode == "babble" else 0.0
    frames, actions, base_pos, base_quat = [], [], [], []
    min_z = sim.getObjectPosition(root, sim.handle_world)[2]
    t0 = sim.getSimulationTime()
    for i in range(args.steps):
        tc = sim.getSimulationTime() - t0
        step_action = np.zeros(len(ACTIVE_JOINTS), np.float32)
        for leg in LEGS:
            leg_bias = base_bias + (turn_bias if leg in LEFT_LEGS else -turn_bias)
            swing_amp = max(0.05, SWING_AMP + leg_bias)
            frac = (args.freq * tc + PHASE[leg]) % 1.0
            swing, lift = leg_trajectory(frac, swing_amp=swing_amp)
            lift_sign = -1.0 if bias2[leg] > 0 else 1.0
            swing_t = SWING_SIGN[leg] * swing
            lift_t = bias2[leg] + lift_sign * lift
            if args.mode == "babble" and args.noise > 0.0:
                swing_t += rng.normal(0, args.noise)
                lift_t += rng.normal(0, args.noise)
            sim.setJointTargetPosition(handles[(1, leg)], float(swing_t))
            sim.setJointTargetPosition(handles[(3, leg)], float(swing_t))
            sim.setJointTargetPosition(handles[(2, leg)], float(lift_t))
            sim.setJointTargetPosition(handles[(4, leg)], float(lift_t))
            i1, i3 = ACTIVE_JOINTS.index(f"joint1_{leg}"), ACTIVE_JOINTS.index(f"joint3_{leg}")
            i2, i4 = ACTIVE_JOINTS.index(f"joint2_{leg}"), ACTIVE_JOINTS.index(f"joint4_{leg}")
            step_action[[i1, i3]] = swing_t
            step_action[[i2, i4]] = lift_t
        sim.step()   # advance physics BEFORE capture, so the frame matches the action just applied

        frames.append(capture(sim, cam))
        actions.append(step_action)
        base_pos.append(sim.getObjectPosition(root, sim.handle_world))
        base_quat.append(sim.getObjectQuaternion(root, sim.handle_world))
        min_z = min(min_z, base_pos[-1][2])

    sim.stopSimulation()

    if min_z < FALL_HEIGHT:
        print(f"FELL (min_z={min_z:.4f} < {FALL_HEIGHT}) -- discarding, nothing saved")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out,
                        frames=np.asarray(frames, np.uint8),
                        action=np.asarray(actions, np.float32),
                        base_pos=np.asarray(base_pos, np.float32),
                        base_quat=np.asarray(base_quat, np.float32),
                        dt=np.float64(DT))
    disp = np.asarray(base_pos)[-1, :2] - np.asarray(base_pos)[0, :2]
    print(f"saved {args.steps} frames -> {args.out}  "
         f"x_travel={disp[0]:+.3f}m y_drift={disp[1]:+.3f}m min_z={min_z:.4f}")


if __name__ == "__main__":
    main()
