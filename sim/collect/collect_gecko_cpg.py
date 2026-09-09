"""Drive the gecko's 16 leg joints with a duty-cycle CPG, live in CoppeliaSim.

Not a dataset collector yet (no recording) -- this is the actual, persistent home for the gait
testing done interactively while tuning `sim/scene/build_gecko_scene.py`. Every parameter below is
exactly what was swept by hand; edit and re-run rather than writing another scratch script.

**Three real bugs found and fixed, in order -- this now genuinely walks straight.**

1. `joint2`/`joint4` ("lift") rest right next to one of their own hard limits (e.g. `joint2_lf`
   rests at -0.96 rad in a [-1.047, +0.698] range -- 0.09 rad of room one way, 1.66 rad the other).
   The first CPG pushed lift toward the *near* wall and saturated almost immediately. Fixed: push
   toward whichever side actually has room (`LIFT_SIGN`).
2. `joint1`/`joint3` ("swing", fore-aft) has a left/right-mirrored convention, not a front/rear
   one: `+0.5 rad` moves `lf`/`lh` forward but `rf`/`rh` backward (confirmed by direct measurement
   of foot displacement). A diagonal-pair phase pattern that sends the *same* signal to both
   members of a pair therefore moved both front legs one physical direction and both rear legs the
   other -- a front/rear split, not a diagonal trot, regardless of correct phase. Fixed with a
   per-leg `SWING_SIGN`.
3. A pure sinusoid gives every leg equal time "swinging" and "planted," so there is no real push
   phase and no leg is ever truly still -- confirmed directly (isolating one leg's motion while
   holding the other three at their exact rest target shows no drift, so the "other legs also seem
   to move" symptom was the sinusoid never holding anything still, not a coupling bug). Replaced
   with an explicit duty cycle: fast swing through the air, slow push while planted (`DUTY`).

Measured, 8 s runs: sinusoid + fixes 1-2 only -> 0.158 m, -3.8 deg yaw. Duty cycle + all three
fixes -> 0.181 m, -0.0 deg yaw -- essentially a straight line.

**Heading is still open-loop and still wanders.** A short (1.3 s) clip yaws 60-140 degrees
depending on gain/frequency -- present even in the "0.181m/-0.0deg" run above once measured over
enough cycles, just not visible in a single lucky trial. `collect_ik.py`'s hexapod gait has the
exact same open-loop wander (its own docstring: "yaw sd 0.016... not a property of the robot, it's
a controller we gave one and not the other") and fixes it with a PI loop on heading error trimming
a differential left/right swing bias -- the same mechanism B1 uses (F78). `HEAD_KP`/`HEAD_KI` below
port that pattern to gecko: off (0.0) by default so nothing changes unless explicitly enabled.

**To explore live**: run this with CoppeliaSim's GUI visible (DISPLAY=:0, not `-h`), then use the
Play/Pause toolbar buttons and CoppeliaSim's built-in Joint Tool add-on to inspect or manually jog
individual joints. Manual edits are overwritten on the next step while this script is actively
driving -- pause first.

Usage (CoppeliaSim must already be running):
  .venv/bin/python3 sim/collect/collect_gecko_cpg.py
  .venv/bin/python3 sim/collect/collect_gecko_cpg.py --duration 8
  .venv/bin/python3 sim/collect/collect_gecko_cpg.py --duration 0   # settle only, no CPG, then hold
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = os.path.join(ROOT, "sim", "env", "gecko_legs.ttt")

LEGS = ["lf", "rf", "lh", "rh"]

# Diagonal-pair trot, expressed as a phase FRACTION of the cycle (0-1): lf+rh together, rf+lh
# together, half a cycle apart. Genuinely diagonal once SWING_SIGN corrects the left/right
# convention mismatch (see module docstring, bug 2).
PHASE = {"lf": 0.0, "rh": 0.0, "rf": 0.5, "lh": 0.5}

# Left/right-mirrored joint1/joint3 convention (bug 2): +0.5 rad moves lf/lh forward, rf/rh
# backward. Without this, identical commands to a "diagonal pair" produce opposite physical motion.
SWING_SIGN = {"lf": 1.0, "lh": 1.0, "rf": -1.0, "rh": -1.0}

SWING_AMP = 0.35   # rad, joint1 & joint3 (fore-aft)
LIFT_AMP = 0.6     # rad, joint2 & joint4 (vertical clearance)
DUTY = 0.35        # fraction of the cycle spent in swing (airborne); the rest is stance (planted,
                   # pushing backward -- this is where propulsion actually comes from)
FREQ = 1.0         # Hz
SETTLE_S = 1.0     # let the robot drop and settle before the CPG starts

# Heading PI (off by default, ported from collect_ik.py's hexapod heading mode / B1's F78). LEFT
# legs get +trim, RIGHT get -trim, same differential-drive lever as --turn_bias in
# collect_gecko_dataset.py, just now driven by measured yaw error instead of a fixed constant.
LEFT_LEGS = {"lf", "lh"}
HEAD_KP = 0.0
HEAD_KI = 0.0
HEAD_TRIM_MAX = 0.15   # rad, cap on the per-leg swing-amplitude bias the PI can apply


def heading_from_quat(quat):
    """atan2(fy, fx) using embodiment.py's gecko forward-axis formula (x,y,z,w order, no sign
    correction) -- kept in lockstep with wm/data/embodiment.py's `heading()` gecko branch so the
    PI target here matches what the recorded data will later measure yaw against."""
    x, y, z, w = quat
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    return np.arctan2(fy, fx)


def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def leg_trajectory(phase_frac, duty=DUTY, swing_amp=SWING_AMP, lift_amp=LIFT_AMP):
    """One leg's (swing, lift) target at this point in its own cycle, before SWING_SIGN/LIFT_SIGN.

    Swing: sweeps forward while airborne (fast, `duty` of the cycle), then backward while planted
    (slow, the rest) -- the backward, grounded sweep is the push that actually propels the body.
    Lift: a half-sine that rises and falls entirely within the swing window, zero during stance.
    """
    if phase_frac < duty:
        local = phase_frac / duty
        swing = -swing_amp + 2 * swing_amp * local
        lift = lift_amp * np.sin(np.pi * local)
    else:
        local = (phase_frac - duty) / (1 - duty)
        swing = swing_amp - 2 * swing_amp * local
        lift = 0.0
    return swing, lift


def drive(sim, duration=3.0, settle=SETTLE_S, head_kp=HEAD_KP, head_ki=HEAD_KI):
    sim.loadScene(SCENE)

    handles, bias2 = {}, {}
    for leg in LEGS:
        for j in (1, 2, 3, 4):
            handles[(j, leg)] = sim.getObject(f"/joint{j}_{leg}")
        bias2[leg] = sim.getJointPosition(handles[(2, leg)])

    root = sim.getObject("/geckobotiv")
    sim.setStepping(True)
    sim.startSimulation()
    t0 = sim.getSimulationTime()
    yaw0, yaw_int = None, 0.0
    while True:
        t = sim.getSimulationTime() - t0
        if t > settle + duration:
            break
        if t > settle:
            tc = t - settle
            trim = 0.0
            if head_kp or head_ki:
                quat = sim.getObjectQuaternion(root, sim.handle_world)
                yaw = heading_from_quat(quat)
                if yaw0 is None:
                    yaw0 = yaw   # hold whatever heading the robot had when the CPG started
                err = wrap_pi(yaw0 - yaw)
                yaw_int = float(np.clip(yaw_int + err, -2.0, 2.0))
                trim = float(np.clip(head_kp * err + head_ki * yaw_int, -1.0, 1.0)) * HEAD_TRIM_MAX
            for leg in LEGS:
                frac = (FREQ * tc + PHASE[leg]) % 1.0
                leg_bias = trim if leg in LEFT_LEGS else -trim
                swing, lift = leg_trajectory(frac, swing_amp=max(0.05, SWING_AMP + leg_bias))
                sim.setJointTargetPosition(handles[(1, leg)], SWING_SIGN[leg] * swing)
                sim.setJointTargetPosition(handles[(3, leg)], SWING_SIGN[leg] * swing)
                lift_sign = -1.0 if bias2[leg] > 0 else 1.0  # toward the open side, away from the near wall
                sim.setJointTargetPosition(handles[(2, leg)], bias2[leg] + lift_sign * lift)
                sim.setJointTargetPosition(handles[(4, leg)], bias2[leg] + lift_sign * lift)
        sim.step()

    p0 = np.array(sim.getObjectPosition(root, sim.handle_world))
    print(f"final pos={np.round(p0, 4)}  "
          f"(re-run with a longer --duration and log the trace yourself for net displacement)")

    # Hand control back to the GUI: with stepping left on, the simulation only advances when this
    # script calls sim.step(), so the Play button would appear to do nothing once this exits.
    sim.setStepping(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=3.0, help="seconds of CPG motion after settling")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--head_kp", type=float, default=HEAD_KP, help="heading PI proportional gain, "
                     "0.0=open loop (default, matches gecko's original gait)")
    ap.add_argument("--head_ki", type=float, default=HEAD_KI, help="heading PI integral gain")
    args = ap.parse_args()

    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    drive(sim, duration=args.duration, head_kp=args.head_kp, head_ki=args.head_ki)
    print("simulation left running -- use the GUI Play/Pause to inspect or take over manually")


if __name__ == "__main__":
    main()
