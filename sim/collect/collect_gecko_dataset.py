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

**Froude-speed calibration (F181) baked into the CPG defaults this now imports.** `FREQ` and the
per-leg `DUTY_PER_LEG` from `collect_gecko_cpg.py` push froude_fwd from ~0.013 to ~0.123-0.125,
matching hexapod (0.134) and B1 (0.131) -- see that module's docstring for the full mechanism and
measurements. Residual yaw drift (36-61 degrees per ~2.6 s clip, down from ~87 unbiased) is real
and not fully solved; some of it is physics-level micro-nondeterminism that no gait bias removes.
Any recollected data should be understood against that, not assumed straight-line clean.

**`--explore` (decaying-prior exploration) exists, is measured, and did NOT solve what it was
built for -- keep it only with that context.** It was added on the diagnosis that gecko's babble was
~2x narrower than B1's in action statistics. That diagnosis was measured through the broken body
frame (F189) and does not survive correcting it. Two things were learned and are worth keeping:

  * pushing commanded-action diversity to B1's level (action std 0.225 -> 0.453) made the achieved
    forward-speed coverage monotonically WORSE (0.031 -> 0.021). The position PID low-passes the
    jitter and the gait degrades toward thrashing. **Command-space entropy is not dynamics coverage.**
  * the per-step babble noise is what makes the action-to-body-motion mapping learnable at all.
    Holding each clip's gait constant (`data/gecko/babble_v3`, collected for coverage) DROPS the
    ground-truth action ceiling from median rho +0.736 to +0.146, because it removes the within-clip
    variation the mapping is fitted on. Prefer the noisy original babble.

The mechanism, for when it is wanted: instead of blending the commanded target toward random joint
angles (which has no reason to keep the robot upright), the CPG's own PARAMETERS random-walk within
bounds, so every commanded target stays a valid gait target. Per-step noise ramps from `--noise` to
`--noise_max`, both gated by a schedule going 0 -> 1 over `--ramp_s`, so a clip starts at the exact
stable calibrated gait and widens away from it. `FALL_HEIGHT` is the manifold boundary.

**Phase is integrated incrementally rather than recomputed from absolute time**, because a drifting
`freq` in the old `(freq * tc)` form would make the commanded phase JUMP whenever freq changed,
which is both off-manifold and a destabiliser. `phase += freq_t * DT` is continuous by construction.
"""
import argparse
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
from collect_gecko_cpg import (  # noqa: E402
    DUTY_PER_LEG, FREQ, LEGS, LIFT_AMP, PHASE, SCENE, SWING_AMP, SWING_SIGN, leg_trajectory,
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
    ap.add_argument("--freq", type=float, default=FREQ, help="gait frequency, Hz. **NOT the "
                    "forward-speed lever it was believed to be.** The sweep that appeared to show "
                    "2.0 -> 0.033 up to 5.0 -> 0.133 was measured through a broken body frame "
                    "(F189). Re-measured on the corrected axis, forward Froude is FLAT across this "
                    "range: 2.0 -> 0.038, 3.0 -> 0.042, 4.0 -> 0.045, 5.0 -> 0.048, 6.0 -> 0.039. "
                    "Gecko sits at ~0.04 against B1's 0.126 and frequency does not close that.")
    ap.add_argument("--lift", type=float, default=LIFT_AMP, help="per-ROLLOUT constant lift "
                    "(ground-clearance) amplitude, rad. Coherent across the clip, unlike "
                    "`--drift_lift`. A second real gait axis that the fixed sweep pinned at a "
                    "constant, so joints 2/4 carried no across-clip variation at all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)

    ap.add_argument("--explore", action="store_true", help="decaying-prior exploration (see module "
                    "docstring): the CPG's own parameters random-walk within bounds over the clip "
                    "and the per-step noise ramps up, both scaled by a 0->1 schedule, so the clip "
                    "starts at the stable calibrated gait and widens away from it ON-MANIFOLD. "
                    "This exists because the fixed-parameter babble measured ~2x narrower than the "
                    "B1 set that grounds successfully, and that narrowness -- not the adaptation "
                    "mechanism -- is what three separate pipeline levers point at.")
    ap.add_argument("--noise_max", type=float, default=0.12, help="terminal per-step noise the "
                    "schedule ramps `--noise` up to (explore mode). The fall check is the real "
                    "guard on this, not the number itself -- sweep it and read the fall rate.")
    ap.add_argument("--ramp_s", type=float, default=0.4, help="seconds over which the schedule goes "
                    "0 -> 1. Short relative to the clip on purpose: the point is to spend most of "
                    "the clip WIDE, with the stable prior only anchoring the first few strides.")
    ap.add_argument("--drift_bias", type=float, default=0.02, help="per-step random-walk scale on "
                    "the symmetric swing-amplitude bias (explore mode)")
    ap.add_argument("--drift_turn", type=float, default=0.02, help="per-step random-walk scale on "
                    "the asymmetric turn bias (explore mode)")
    ap.add_argument("--drift_freq", type=float, default=0.10, help="per-step random-walk scale on "
                    "gait frequency, Hz (explore mode)")
    ap.add_argument("--drift_lift", type=float, default=0.03, help="per-step random-walk scale on "
                    "LIFT amplitude, i.e. ground clearance (explore mode). **This is the single "
                    "biggest missing diversity source in the fixed-parameter babble**: `LIFT_AMP` "
                    "is a constant there, so joints 2 and 4 -- half of the 16 action dims -- carry "
                    "pure noise and no structured variation whatsoever, which caps how wide the "
                    "action distribution can possibly get no matter how the swing knobs are swept.")
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
    explore = args.explore and args.mode == "babble"

    # **Bounds, not free drift.** Each is a range the calibrated gait is known to survive somewhere
    # inside: `bias` shifts SWING_AMP=0.35 over [0.10, 0.60]; `turn_bias` spans well past the
    # +-0.25 the fixed sweep already collected without falling; `freq` brackets the calibrated 4.0.
    BIAS_LO, BIAS_HI = -0.25, 0.25
    TURN_LO, TURN_HI = -0.35, 0.35
    FREQ_LO, FREQ_HI = 2.5, 6.0
    LIFT_LO, LIFT_HI = 0.25, 1.00   # around the fixed LIFT_AMP=0.6: less clearance to noticeably more

    cur_bias, cur_turn, cur_freq = base_bias, turn_bias, args.freq
    cur_lift = args.lift
    phase = dict(PHASE)   # integrated incrementally: a drifting freq must not jump the phase

    frames, actions, base_pos, base_quat = [], [], [], []
    min_z = sim.getObjectPosition(root, sim.handle_world)[2]
    t0 = sim.getSimulationTime()
    for i in range(args.steps):
        tc = sim.getSimulationTime() - t0
        # The decaying prior: 0 at the first recorded step (exactly the stable calibrated gait),
        # 1 once ramped (widest on-manifold exploration the fall check will allow).
        alpha = min(1.0, tc / args.ramp_s) if (explore and args.ramp_s > 0) else float(explore)
        if explore:
            cur_bias = float(np.clip(cur_bias + alpha * rng.normal(0, args.drift_bias),
                                     BIAS_LO, BIAS_HI))
            cur_turn = float(np.clip(cur_turn + alpha * rng.normal(0, args.drift_turn),
                                     TURN_LO, TURN_HI))
            cur_freq = float(np.clip(cur_freq + alpha * rng.normal(0, args.drift_freq),
                                     FREQ_LO, FREQ_HI))
            cur_lift = float(np.clip(cur_lift + alpha * rng.normal(0, args.drift_lift),
                                     LIFT_LO, LIFT_HI))
        sigma = args.noise + alpha * max(0.0, args.noise_max - args.noise) if explore else args.noise

        step_action = np.zeros(len(ACTIVE_JOINTS), np.float32)
        for leg in LEGS:
            leg_bias = cur_bias + (cur_turn if leg in LEFT_LEGS else -cur_turn)
            swing_amp = max(0.05, SWING_AMP + leg_bias)
            frac = phase[leg] % 1.0
            swing, lift = leg_trajectory(frac, duty=DUTY_PER_LEG[leg], swing_amp=swing_amp,
                                         lift_amp=cur_lift)
            lift_sign = -1.0 if bias2[leg] > 0 else 1.0
            swing_t = SWING_SIGN[leg] * swing
            lift_t = bias2[leg] + lift_sign * lift
            if args.mode == "babble" and sigma > 0.0:
                swing_t += rng.normal(0, sigma)
                lift_t += rng.normal(0, sigma)
            sim.setJointTargetPosition(handles[(1, leg)], float(swing_t))
            sim.setJointTargetPosition(handles[(3, leg)], float(swing_t))
            sim.setJointTargetPosition(handles[(2, leg)], float(lift_t))
            sim.setJointTargetPosition(handles[(4, leg)], float(lift_t))
            i1, i3 = ACTIVE_JOINTS.index(f"joint1_{leg}"), ACTIVE_JOINTS.index(f"joint3_{leg}")
            i2, i4 = ACTIVE_JOINTS.index(f"joint2_{leg}"), ACTIVE_JOINTS.index(f"joint4_{leg}")
            step_action[[i1, i3]] = swing_t
            step_action[[i2, i4]] = lift_t
        for leg in LEGS:
            phase[leg] += cur_freq * DT
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
    a = np.asarray(actions, np.float32)
    print(f"saved {args.steps} frames -> {args.out}  "
          f"x_travel={disp[0]:+.3f}m y_drift={disp[1]:+.3f}m min_z={min_z:.4f}  "
          f"act_std={a.std(axis=0).mean():.3f} act_rng={(a.max(axis=0) - a.min(axis=0)).mean():.3f} "
          f"act_d={np.abs(np.diff(a, axis=0)).mean():.3f}")


if __name__ == "__main__":
    main()
