"""Drive the gecko's 16 leg joints with a duty-cycle CPG, live in CoppeliaSim.

**EVERY FROUDE AND YAW NUMBER BELOW WAS MEASURED THROUGH A BROKEN BODY FRAME. See F189.**
`forward_axis`/`heading` read gecko's forward from the body x axis, which points almost straight up
(horizontal projection mean length 0.065, reaching 0.000), so forward, lateral and yaw were all
resolved onto a randomly-spinning frame. Re-measured on the corrected `-(body y)` axis:

  * gecko's forward Froude is **0.038-0.048**, NOT the 0.123-0.138 quoted below, against B1's 0.126.
    The "matches or exceeds hexapod/B1" conclusion is withdrawn -- gecko is ~2.6x slower.
  * `FREQ` is **not** the speed lever it appears to be here: re-measured, forward Froude is flat
    across 2-6 Hz (0.038 / 0.042 / 0.045 / 0.048 / 0.039). The apparent 10x gain from FREQ 1.0 -> 4.0
    is not reproducible on a correct frame.
  * the "chaotic yaw" story (-62/+114/+124/+126 degrees across identical runs) was largely this bug:
    per-step heading noise, not physics. On the corrected axis, per-step heading jumps above 90
    degrees go from 16.8% of steps to **zero**.

The gait-shape fixes below (LIFT_SIGN, SWING_SIGN, the duty cycle) are unaffected -- they were
verified against foot displacement and net travel, not against `body_motion`. Treat the tuning
CONSTANTS as still-unvalidated against a correct measurement.

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

**Froude-speed calibration (this project's F181: gecko's forward speed was ~10x below hexapod/B1,
0.013 vs ~0.13).** `FREQ` 1.0 -> 4.0 and rear legs (`lh`/`rh`) given a shorter `DUTY` than front
(more of the cycle in propulsive stance) together push measured `body_velocity()` froude_fwd from
~0.013 to 0.14-0.15 -- matching or exceeding hexapod (0.134) and B1 (0.131).

**That speed increase came with a heading cost, and here's the real mechanism.** A heading-PI trim
(the same mechanism hexapod/B1 use, F69) was tried first and failed: measured directly, running the
*identical* config four times in a row gave yaw drift of -62, +114, +124, +126 degrees -- the
disturbance is chaotic (sign and magnitude both vary run to run), not a fixed bias, and a PI loop
cannot correct noise. Root cause, also measured directly: even passive settling (no gait command at
all) is not perfectly reproducible between scene reloads -- quaternion components vary by ~0.001
across identical runs, a real ~0.1% non-determinism in the physics stepping itself. A diagonal trot
(only 2 feet down at a time) has a narrow support margin that amplifies that noise into large,
unpredictable-sign yaw swings within a handful of strides.

Two things fixed this, stacked:
1. **A wave-style asymmetric duty (this file's `DUTY`, rear shorter than front) turns out to also
   convert the chaotic drift into a *consistent* one** (same sign, tight range, run to run) --
   confirmed directly: this exact rear-shortened-duty config yaws +85 to +89 degrees across 3 fresh
   identical reps, vs. the uniform-duty baseline's chaotic -62 to +126 degree spread. A consistent
   bias, unlike noise, is steerable.
2. **A small additional left/right `DUTY` asymmetry (`DUTY_BIAS`, right legs get longer stance than
   left) cuts that consistent bias roughly in half** -- 0.123-0.125 froude_fwd (still matching
   hex/B1) at 36-61 degrees of yaw drift per ~2.6 s clip, down from ~87 degrees unbiased. Pushing
   `DUTY_BIAS` further (-0.07, -0.08) overcorrects and starts costing real speed for diminishing,
   less consistent yaw gains -- -0.06 is the measured sweet spot.

**Update: the real bottleneck was the joint gain, not the gait shape.** Even after fix 1/2 above,
the swing joints were still only achieving ~52% of their commanded arc (0.364/0.700 rad) -- the
shipped position-PID gain (`bullet_joint_pospid1=0.1`, no damping) in `build_gecko_scene.py`
couldn't develop enough torque to track a fast-changing CPG target, independent of the gait's
timing. Raising it (still undamped -- adding matched damping, the obvious "fix like the other joint
groups" move, was tried and made this measurably worse, since those groups hold a FIXED target
while these track a MOVING one, and Bullet's D-term fights against the tracking velocity itself)
was swept: P=0.15 gives froude_fwd 0.169-0.175, comfortably PAST hexapod (0.134) and B1 (0.131)
rather than matching them. **P=0.11 was chosen instead**, to match rather than exceed: froude_fwd
0.133-0.138, and, stacked with `DUTY_BIAS` above, yaw drift -3 to +40 degrees per ~2.6s clip (down
from +34 to +37 at the original P=0.1). Without `DUTY_BIAS` the gain fix alone still yaws badly, so
both fixes are required together, not redundant. See `build_gecko_scene.py`'s active-joint loop for
the actual `pospid1=0.11` setting.

**Residual yaw (-3 to +40 degrees per clip) is real and not fully solved.** Smaller than the +34 to
+37 degrees before this gain fix in the best cases, comparable in the worst, but not zero -- some of
it is the same underlying physics micro-nondeterminism (measured directly: ~0.1% quaternion
variation between identical scene reloads, even with no gait command at all) that no steering trim
can remove, only outrun. Downstream consumers of gecko's `body_motion` yaw channel should know this.

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
FREQ = 4.0         # Hz -- was 1.0; the froude-speed calibration's main lever (see module docstring)
SETTLE_S = 1.0     # let the robot drop and settle before the CPG starts

LEFT_LEGS = {"lf", "lh"}

# Front/rear duty split (module docstring, fix 1): rear legs get less swing/more stance than front.
BASE_DUTY = {"lf": 0.35, "rf": 0.35, "lh": 0.20, "rh": 0.20}
# Left/right duty asymmetry (module docstring, fix 2): the measured sweet spot that roughly halves
# yaw drift without meaningfully costing speed. Right legs get +DUTY_BIAS more stance (planted
# longer), left legs -DUTY_BIAS less.
DUTY_BIAS = -0.06
DUTY_PER_LEG = {leg: BASE_DUTY[leg] + (DUTY_BIAS if leg in LEFT_LEGS else -DUTY_BIAS) for leg in LEGS}


def heading_from_quat(quat):
    """atan2(fy, fx) using embodiment.py's gecko forward-axis formula (x,y,z,w order, no sign
    correction) -- kept in lockstep with wm/data/embodiment.py's `heading()` gecko branch so any
    yaw measured here matches what the recorded data measures too."""
    x, y, z, w = quat
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    return np.arctan2(fy, fx)


def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def leg_trajectory(phase_frac, duty, swing_amp=SWING_AMP, lift_amp=LIFT_AMP):
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


def drive(sim, duration=3.0, settle=SETTLE_S):
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
    while True:
        t = sim.getSimulationTime() - t0
        if t > settle + duration:
            break
        if t > settle:
            tc = t - settle
            for leg in LEGS:
                frac = (FREQ * tc + PHASE[leg]) % 1.0
                swing, lift = leg_trajectory(frac, duty=DUTY_PER_LEG[leg])
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
    args = ap.parse_args()

    sim = RemoteAPIClient("localhost", port=args.port).require("sim")
    drive(sim, duration=args.duration)
    print("simulation left running -- use the GUI Play/Pause to inspect or take over manually")


if __name__ == "__main__":
    main()
