"""Step A collector: IK-retargeted dataset with the fixed camera.

For each chosen expert episode (66-step forward walk) and each morphology:
  1. shared foot path  = base body's expert motor_pos -> foot-in-abdomen (FK)
  2. per-body commands = IK that path (scaled to be reachable) for THIS body
  3. drive the commands open-loop, fixed world-frame camera, distance-gated
  4. record RGB frames + a_t (the IK commands) + measured foot forces + head pose

Behaviour (the foot path) is shared; commands differ per body -> non-vacuous.
Forces are the *measured* contact on each body (not the expert's), so contact
labels reflect what actually happened.

Straight episodes (clean forward walk): 926,521,625,144,285,997,727,728
Curvy episodes (turning, for later):    472,111,630

Usage (CoppeliaSim up, launched from the venv):
  python3 sim/collect/collect_ik.py --port 23000 --episodes 926,521,625 --scale 0.5 --travel 0.8 --out data/ik_v1
  python3 sim/collect/collect_ik.py --port 23000 --episodes 472 --loops 3 --travel 0 --out data/ik_v1
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.bodies import WALK_FORWARD_M, WALK_LATERAL_M, walk_check  # noqa: E402,F401

ENV = os.path.join(ROOT, "sim", "env")
CSV = f"{ENV}/expert_66k_aug3c_fcontact.csv"
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = {"m1": "TC", "m2": "CF", "m3": "FT"}
SCENES = [("long", "medauroidea_stick_insect.ttt"),
          ("medium", "medauroidea_stick_insect_medium.ttt"),
          ("short", "medauroidea_stick_insect_short.ttt")]
# The reference body whose forward kinematics turn the expert's joint angles into the shared
# Cartesian foot path. It must NOT follow --morphs: that flag replaces SCENES, so reading the
# reference from SCENES[0] silently rebuilt the trajectory out of whichever body was listed
# first, and every body then chased a path derived from something other than the base insect.
REFERENCE_SCENE = "medauroidea_stick_insect.ttt"
SENSOR = "vjepa_cam"
TRACK = "/head"
ROBOT_ROOT = "/abdomen"
FORCE_NAMES = [f"/forceSensor_{leg}" for leg in LEGS]
EP = 66

# Seconds per recorded step. The insect expert runs at 20 Hz (`sim_time` in
# expert_66k_aug3c_fcontact.csv), so a 66-step clip is 3.30 s. Written down rather than left
# implicit: F74 was a frame rate nobody had recorded, and it made every cross-embodiment number
# compare 20 ms on one robot against 50 ms on the other.
STEP_DT = 0.05
CHAIN_NAMES = ("m1", "coxa", "m2", "femur", "m3", "tibia", "tibial", "forceSensor", "foot")


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def read_forces(sim, force_h):
    out = np.zeros(len(force_h), np.float32)
    for i, h in enumerate(force_h):
        r = sim.readForceSensor(h)
        fv = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else [0, 0, 0]
        out[i] = float(np.sqrt(fv[0] ** 2 + fv[1] ** 2 + fv[2] ** 2))
    return out


def get_optional(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


def leg_subtree(sim, leg):
    handles = []
    root = get_optional(sim, f"/m1_{leg}")
    if root is not None:
        handles.extend(sim.getObjectsInTree(root, sim.handle_all, 1) + [root])
    for name in CHAIN_NAMES:
        h = get_optional(sim, f"/{name}_{leg}")
        if h is not None and h not in handles:
            handles.append(h)
    return handles


def ghost_remove_legs(sim, legs):
    """Hide and de-respond selected legs without deleting handles the scene script may expect."""
    disabled_shapes = 0
    disabled_joints = 0
    for leg in legs:
        for h in leg_subtree(sim, leg):
            typ = sim.getObjectType(h)
            if typ == sim.object_shape_type:
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
                try:
                    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
                except Exception:
                    pass
                disabled_shapes += 1
            elif typ == sim.object_joint_type:
                try:
                    sim.setJointTargetForce(h, 0.0)
                except Exception:
                    pass
                disabled_joints += 1
    return disabled_shapes, disabled_joints


def leg_length(sim, leg="FL"):
    """Total rigid-link length, measured directly from the loaded scene."""
    points = [sim.getObject(f"/{jn}_{leg}") for jn in SEG]
    points.append(sim.getObject(f"/foot_{leg}"))
    xyz = [np.asarray(sim.getObjectPosition(h), dtype=float) for h in points]
    return float(sum(np.linalg.norm(b - a) for a, b in zip(xyz[:-1], xyz[1:])))



def body_rel_via_fk(sim, df, rows):
    """Shared foot path in the abdomen frame (base body FK on expert motor_pos)."""
    sim.loadScene(f"{ENV}/{REFERENCE_SCENE}")
    abd = sim.getObjectParent(sim.getObject("/m1_FL"))
    jh = {(leg, jn): sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG}
    foot_h = {leg: sim.getObject(f"/foot_{leg}") for leg in LEGS}
    out = {leg: [] for leg in LEGS}
    for t in rows:
        r = df.iloc[t]
        for (leg, jn), h in jh.items():
            sim.setJointPosition(h, float(r[f"motor_pos_{leg}_{SEG[jn]}"]))
        for leg in LEGS:
            out[leg].append(np.array(sim.getObjectPosition(foot_h[leg], abd)))
    return {leg: np.array(v) for leg, v in out.items()}


def frame_axes(brel):
    """Which axis of the abdomen frame is lateral, which is fore-aft, which is vertical.

    **Found from what the legs mean, not from how far they move.** An earlier version took the
    widest-ranging axis as fore-aft and assumed the third was vertical, and got both wrong: in this
    scene z runs front-to-back, y left-to-right and x up-down, so the generated gait swept sideways
    and lifted forwards -- 0.39 m of lateral drift against 0.02 m of travel.

        lateral    the axis whose mean differs between the L and R legs; nothing else does
        fore-aft   of the two left, the one whose per-leg means spread out, since front and hind
                   feet sit at different points along the body
        vertical   whatever remains
    """
    means = {leg: np.asarray(path, dtype=float).mean(0) for leg, path in brel.items()}
    left = np.stack([m for leg, m in means.items() if leg.endswith("L")]).mean(0)
    right = np.stack([m for leg, m in means.items() if leg.endswith("R")]).mean(0)
    lateral = int(np.argmax(np.abs(left - right)))
    rest = [k for k in range(3) if k != lateral]
    spread = [np.ptp([m[k] for m in means.values()]) for k in rest]
    fore = rest[int(np.argmax(spread))]
    up = [k for k in range(3) if k not in (lateral, fore)][0]
    return lateral, fore, up


TRIPOD_A = ("FL", "HL", "MR")   # the two alternating groups of an insect tripod


def stride_period(paths, lo=6, hi=None):
    """Frames per stride, from how well the path repeats itself one period later.

    **Not an FFT bin.** A 66-frame clip only resolves periods of 66, 33, 22, 16.5 ..., so the bin
    nearest the truth can be several frames out, and folding cycles onto a wrong length averages
    them out of phase and flattens the gait -- measured: the folded path walked 0.01 m where the
    recording walks 0.55. Scoring every integer period by `|path[:L] - path[L:2L]|` picks the length
    the path actually repeats at.
    """
    stack = np.concatenate([np.asarray(v, dtype=float) for v in paths], axis=1)
    T = len(stack)
    hi = hi or T // 2
    best, best_score = lo, float("inf")
    for L in range(lo, hi + 1):
        n = T // L
        if n < 2:
            break
        cycles = stack[:n * L].reshape(n, L, -1)
        score = float(np.abs(cycles - cycles.mean(0)).mean())
        if score < best_score:
            best, best_score = L, score
    return best


def cpg_frame(recipe, leg_gain, leg_off, t, spin):
    """One frame of the oscillator, with `spin` supplied per step rather than fixed.

    **This exists so the hexapod can hold a heading the way the B1 does.** The B1's rollout runs a
    PI controller on heading error (F78) and drifts 0.33 deg/s; the insect's oscillator is open loop
    and wanders 3x more (yaw sd 0.016 against 0.005 in the conditions where neither robot is
    turning). That difference is not a property of the two robots -- it is a controller we gave one
    and not the other -- and it lands in the yaw channel's noise floor, where two thirds of the
    conditions live. Closing the loop on both sides removes it at the source instead of matching
    the noise afterwards, which would be fitting the dataset to the method.
    """
    r = recipe
    ph = 2 * np.pi * r["cycles"] * t / r["frames"]
    o = (np.sin(ph + 2 * np.pi * r["lead"]), np.sin(ph),
         np.sin(ph + 2 * np.pi * r["ft_phase"]))
    cmd = r["bias"].astype(np.float64).copy()
    for i, leg in enumerate(LEGS):
        sign = 1.0 if leg in TRIPOD_A else -1.0
        mirror = 1.0 if leg.endswith("L") else -1.0
        for k in range(3):
            axis = mirror if k in r["mirror_joints"] else 1.0
            span = r["amps"][k] * (leg_gain[i] if k > 0 else 1.0)
            cmd[i * 3 + k] += sign * axis * span * o[k]
            if k > 0:
                cmd[i * 3 + k] += leg_off[i]
        if spin:
            cmd[i * 3] += sign * spin * r["spin_amp"] * o[0]
        if r["strafe"]:
            cmd[i * 3 + 2] += sign * r["strafe"] * r["amps"][2] * o[2]
    return cmd


def cpg_commands(sim, scene, frames, centre, cycles=6.0, amps=(0.25, 0.20, 0.20),
                 lead=0.25, mirror_joints=(0, 1, 2), strafe=0.0, spin=0.0,
                 spin_amp=None, ft_phase=0.0, symmetric=False, legtune=None):
    """Joint-space oscillator, the pattern from the lab's `student_Locomotion_Control_olaf_6legs`.

    **No IK.** Two sinusoids a quarter cycle apart drive the three joints of every leg, with the
    sign flipped between the two tripod groups. Nothing asks where a foot should be, so none of the
    ways a foot target can be wrong apply: unreachable corners, a stance line that leaves the
    workspace, planted feet at inconsistent heights. That is the reason to have it.

    It also replaces a route that did not work. Re-timing the recorded wave into a tripod tracks its
    targets exactly -- IK residual 0.00 mm -- and still pitches the body over and slides backwards,
    because leg paths authored for a gait that lifts one leg at a time do not support the body when
    three lift together. A gait has to be designed as a whole; re-phasing one does not give another.

    **What it is and is not.** This is a fixed-amplitude open-loop oscillator, not the lab's CPG
    (Larsen et al. 2023), which adapts its timing per body and bounds its own output. It is the
    demonstration controller from the Olaf scene, ported to these bodies and given their own rest
    pose rather than Olaf's.

    **The whole controller in one place, because the three gaits are one equation.**

    Three oscillators share a clock, `ph = 2*pi * cycles * t / T`:

        TC  sweep    o0 = sin(ph + 2pi*lead)     lead 0.25 -> cos(ph), a quarter cycle ahead
        CF  lift     o1 = sin(ph)                the clock; what decides which feet are down
        FT  extend   o2 = sin(ph + 2pi*ft_phase)

    and each joint gets

        theta = bias                              the walking pose, from the mean of the IK commands
              + sign * mirror * A[k] * o[k]       the mirrored drive
              + sign *          spin * spin_amp * o0     TC only, un-mirrored
              + sign *        strafe * A[2]     * o2     FT only, un-mirrored

    with `sign` +1 on tripod A (FL HL MR) and -1 on B, `mirror` +1 on the left and -1 on the right.

    **Whether a term is mirrored is what decides the behaviour, and the reason is that the two
    sides are mirror images**: equal joint angles across the midline mean *opposite* motion in the
    world. A mirrored term therefore moves both sides the same way through the world and its
    sideways parts cancel, leaving forward walking. An un-mirrored one cancels the fore-aft parts
    instead, and what remains depends on the joint it went to -- TC spins, FT crabs, CF only rolls.

    So the useful question about any of these gaits is **which joint carries a left-right amplitude
    imbalance**. Resolved per leg, in radians:

                    TC sweep            CF lift          FT extend        |left| : |right|
        straight    0.250 both        0.200 both        0.300 both        1:1 and 1:1
        spin 0.8    0.450 / 0.050     0.200 both        0.300 both        **9:1 at TC**
        sideways    0.037 both        0.200 both        0.060 / 0.540     **1:9 at FT**

    **CF is byte-for-byte identical in all three.** It is the clock, not a drive, and nothing here
    ever touches it. Everything else follows one rule:

        both even          -> walks forward
        TC uneven          -> spins
        FT uneven, TC ~0   -> walks sideways

    **`spin` and `strafe` are balance knobs, not throttle knobs**: they give |left| = A(1 + knob)
    against |right| = A(1 - knob). At `knob = 1` one side's joint **stops moving entirely**, and
    past 1 it reverses. Both gaits run at 0.8 -- close to that edge without falling over it -- and
    `--strafe 1.0` was measured collapsing to 0.12 m from 0.8's 0.33, which is that edge.

    The last two settings are phase, not amplitude. `lead 0.25` puts the sweep a quarter cycle
    ahead of the lift, so the leg sweeps back while planted; negating it does not reverse the walk,
    it steers. `ft_phase` locks the splay to the lift: **0.5 is exact antiphase**, o2 = -o1, which
    means splay while the foot is down and fold while it is up -- the definition of a sideways step.
    Forward walking uses 0.125 instead, which flattens the bottom of the foot path for a stance to
    slide along. Running the sideways gait at forward walking's 0.125 splays in the air and drags
    on the ground, and **80% of the stroke is lost to slip**: 0.24 rad of splay on a 0.5-0.77 m leg
    should carry 0.12-0.18 m per cycle and carried 0.03.

    **Neither of the two steering knobs this used to carry works, and both are gone.** `--turn_bias`
    added a constant to one side's swing joints: at most 8 degrees of heading against 30 of natural
    wander, because moving where a leg sits does not change how far it pushes. `--turn` scaled one
    side's amplitude down instead, and over three repeats moved the heading **+2 degrees at +0.3 and
    +14 at -0.3** -- neither monotonic nor symmetric, against `--spin`'s 73. What it does instead is
    brake, travel 0.37 to 0.21 m, because shortening both legs on one side slows the robot more than
    it turns it. **`--spin` is the steering drive** (F71).
    """
    sim.loadScene(f"{ENV}/{scene}")
    # **Oscillate around the animal's walking posture, not the model's spawn pose.** The scene's
    # default joint angles are where the body happens to sit when loaded, and using them put the
    # abdomen at 0.284 m against the recorded gait's 0.129 -- and at a different height again for
    # every behaviour, 0.248 turning and 0.103 backwards. Froude divides by that height, so a
    # posture that moves between behaviours makes the one quantity both robots share incomparable.
    # The mean of the IK commands is the pose the recording actually walks in.
    bias = np.asarray(centre, dtype=np.float32)
    if symmetric:
        # **Average each left-right pair, because the animal is not symmetric and the pose is its
        # mean.** The model's two sides mirror exactly -- the same physical stance reads as equal
        # and opposite joint angles -- but the recording does not: the middle pair's lift joints sit
        # at +0.706 and -0.846, 0.140 radians apart, and the front pair at +0.978 and -1.027. Frozen
        # as a standing pose that asymmetry decides which foot touches first, and MR, the worst
        # offender, is the leg that never joined its tripod in any variant tried.
        #
        # Front, middle and hind stay different: those legs are 0.771, 0.489 and 0.638 long and
        # attach 0.29 apart, which is anatomy rather than noise.
        for a, c in (("FL", "FR"), ("ML", "MR"), ("HL", "HR")):
            ia, ic = LEGS.index(a) * 3, LEGS.index(c) * 3
            for k in range(3):
                half = 0.5 * (bias[ia + k] - bias[ic + k])
                bias[ia + k], bias[ic + k] = half, -half

    # **The quarter-cycle between sweep and lift is what sets the direction of travel**, and its
    # sign is the direction. Negating the sweep amplitude does not reverse the walk -- that is the
    # same operation as mirroring the sides, and it steers instead. Neither does flipping the
    # mirror. Only moving the sweep relative to the lift decides which half of the stroke the foot
    # is planted for.
    ph = 2 * np.pi * cycles * np.arange(frames) / frames
    # **The extend joint gets its own phase.** Driving it from the same signal as the lift, which
    # is what the Olaf scene does, makes the foot trace a circle: lowest for an instant and rising
    # everywhere else. Offsetting it flattens the bottom of that path into something a stance can
    # push along -- measured kinematically, height variation across the lower half of the cycle
    # falls from 0.22 of the lift to 0.12 at `ft_phase` 0.25.
    o = np.stack([np.sin(ph + 2 * np.pi * lead),
                  np.sin(ph),
                  np.sin(ph + 2 * np.pi * ft_phase)], axis=1)

    cmds = np.tile(bias, (frames, 1))
    # Kept so the collector can regenerate one frame with a different spin, which is what closing
    # the heading loop needs. Everything below writes into `cmds`; this records what it took.
    recipe = dict(bias=bias.copy(), amps=tuple(amps), lead=lead, ft_phase=ft_phase,
                  mirror_joints=tuple(mirror_joints), strafe=strafe, spin=spin,
                  spin_amp=amps[0] if spin_amp is None else spin_amp, cycles=cycles,
                  frames=frames)

    # **One amplitude on six unequal legs does not give six equal strokes.** The pairs measure
    # 0.771, 0.489 and 0.638 long, so the same joint angles swing the front feet 0.111 and the
    # middle feet 0.045, and the front feet reach 0.038 deeper than the middle pair and 0.056
    # deeper than the hind. Whichever feet reach lowest carry the robot, so the contact pattern
    # follows leg length rather than the phase the oscillator asks for -- which is why the middle
    # legs' bars stayed short and broken through every phase and amplitude tried.
    #
    # `scripts/diagnostics/tune_legs.py` solves two numbers per leg against the kinematics: a gain
    # on the lift and extend amplitudes so every foot rises the same distance, and an offset on
    # the same joints so every stroke bottoms out at the same height. They close the two spreads
    # from 0.072 and 0.056 to under 0.0001. This is geometry, not tuning -- it is the correction
    # for legs the animal does not build identically, and it is per body.
    leg_gain = np.ones(len(LEGS), np.float32)
    leg_off = np.zeros(len(LEGS), np.float32)
    if legtune is not None:
        order = [str(x) for x in legtune["legs"]]
        for i, leg in enumerate(LEGS):
            j = order.index(leg)
            leg_gain[i], leg_off[i] = legtune["gain"][j], legtune["offset"][j]

    swing = amps[0] if spin_amp is None else spin_amp
    for i, leg in enumerate(LEGS):
        sign = 1.0 if leg in TRIPOD_A else -1.0
        mirror = 1.0 if leg.endswith("L") else -1.0
        gain = 1.0
        for k in range(3):
            axis = mirror if k in mirror_joints else 1.0
            # the per-leg gain and offset touch the lift and extend joints only; the swing joint
            # sets stride length, and equalising it would make every leg cover the same ground
            # regardless of how far from the turn centre it sits
            span = amps[k] * (leg_gain[i] if k > 0 else 1.0)
            cmds[:, i * 3 + k] += sign * axis * gain * span * o[:, k]
            if k > 0:
                cmds[:, i * 3 + k] += leg_off[i]

        # **Whether a drive is mirrored between the sides decides what it does to the body.** A
        # mirrored term moves the two sides in the same world direction, so its sideways parts
        # cancel and the robot walks. An un-mirrored one moves them oppositely, so the fore-aft
        # parts cancel instead -- and what is left depends on which joint it went to:
        #
        #     swing joint (TC)    one side sweeps forward while the other sweeps back  -> spins
        #     extend joint (FT)   one side pushes out while the other pulls in         -> crabs
        #     lift joint (CF)     one side rises while the other drops                 -> rolls
        #
        # The first attempt at sideways motion used the lift joint on the reasoning that any
        # un-mirrored term would translate. It rolled and yawed instead: over three runs, sideways
        # travel 0.05 of forward against straight walking's 0.01, with a 24 degree heading change.
        if spin:
            # **Spin needs a swing amplitude of its own, because the sideways gait sets `amps[0]`
            # to zero.** Scaling it by the fore-aft amplitude, as this did until 2026-08-22, made
            # `--spin` multiply by zero in exactly the configuration that needed it -- a sweep of
            # `--spin 0.15` against `0.25` moved the heading by 1 degree and read as "spin cannot
            # cancel this yaw" when spin had not been applied at all.
            cmds[:, i * 3] += sign * spin * swing * o[:, 0]
        if strafe:
            # Splitting this by leg row was tried and does not cancel the yaw: front alone
            # yaws +1 degree, middle +12, hind +80, so it is not a front-against-hind couple and
            # the best of four gain sets still left 20 degrees. Switching the fore-aft swing off
            # is what fixes it (F71).
            cmds[:, i * 3 + 2] += sign * strafe * amps[2] * o[:, 2]

    return cmds, dict(target_leg_length=leg_length(sim), scale=1.0,
                      residual_mean_mm=0.0, residual_max_mm=0.0,
                      leg_gain=leg_gain.copy(), leg_off=leg_off.copy(), **recipe)


def parse_schedule(spec):
    """`"1@0.4 0@0.2 1@0.4"` -> [(1.0, 0.4), (0.0, 0.2), (1.0, 0.4)].

    Each segment is `rate@fraction`: how fast the foot path plays, and what share of the clip's
    frames it gets. Rate 0 holds the pose, which is a stop. The same string drives
    `rollout_b1_mujoco.py`, so one schedule can be handed to both robots and mean the same thing.
    """
    segments = []
    for token in spec.split():
        rate, _, frac = token.partition("@")
        if not frac:
            raise SystemExit(f"schedule segment {token!r} needs rate@fraction, e.g. 1@0.4")
        segments.append((float(rate), float(frac)))
    if not segments:
        raise SystemExit("empty --schedule")
    if all(r == 0 for r, _ in segments):
        raise SystemExit("--schedule has no moving segment, so the clip would cover no ground")
    if sum(f for _, f in segments) > 1.001:
        raise SystemExit(f"--schedule fractions sum to {sum(f for _, f in segments):.2f}; "
                         "they are shares of one clip and cannot exceed 1")
    return segments


def schedule_path(brel, segments):
    """Play the shared foot path at a piecewise rate: the general form of `retime`.

    A stop mid-clip is the point of this. Every clip today accelerates from rest at frame 0 and
    slows at the end, so *when* the robot is stationary is fixed and a probe reads it off the body's
    position in frame. Put the pauses somewhere the frame cannot predict and the same visual state
    is followed by more than one future -- the condition an inverse model needs before its latent
    carries anything the current frame does not already supply.

    **Rate means the same thing as `retime`'s speed: source frames consumed per output frame.**
    Rate 1 is the recorded pace, 0 holds the pose, 1.2 is twenty percent faster. It is deliberately
    *not* renormalised so the clip still covers the whole path -- an earlier version did that, and it
    made a pause silently speed up every moving segment to compensate (a schedule pausing for 30
    percent of the clip walked the rest 1.43x faster, putting the insect at Froude 0.23 against the
    B1's 0.17 on the same schedule string). Stopping and walking faster would then always arrive
    together, which is exactly the confound the schedule exists to break. A clip that pauses covers
    less ground instead, which the `--travel` gate and the fixed camera both handle.
    """
    T = len(next(iter(brel.values())))
    pos, here = [], 0.0
    for rate, frac in segments:
        for _ in range(max(1, int(round(frac * T)))):
            pos.append(min(here, T - 1.0))
            here += rate
    pos = np.clip(np.asarray(pos), 0, T - 1)

    src = np.arange(T, dtype=float)
    return {leg: np.stack([np.interp(pos, src, path[:, k]) for k in range(3)], axis=1)
            for leg, path in brel.items()}


def retime(brel, speed, speed_end=None):
    """Resample the shared foot path along time, so the same stride takes fewer or more steps.

    Body speed in a kinematic replay comes from the stance feet sweeping backwards relative to the
    abdomen, so playing the same Cartesian path through fewer samples makes the robot cover the
    same ground in less time. `speed 1.15` is 15 percent faster.

    **Every leg is resampled by the same time map**, so the inter-leg phase relationships are
    untouched. That matters here more than it looks: the expert is a real stick insect walking a
    variable wave, and F56 measured that its five non-reference legs land at near-uniform phase
    (concentration 0.07-0.24) where a B1's are pinned at 0.99-1.00. That variability is a property
    of the animal and the reason no tight cross-robot pairing exists; retiming preserves it, where
    authoring a synthetic tripod path would throw it away along with the rest of the recording.

    Why this is needed at all: the expert walks **one speed**. Across 1,000 episodes its forward
    velocity has a standard deviation of 0.0086 m/s on 0.454, which is 1.9 percent. A body-level
    quantity cannot be a shared supervisory signal between the two robots when one of them never
    varies it.
    """
    if speed == 1.0 and speed_end is None:
        return brel
    T = len(next(iter(brel.values())))
    end = speed if speed_end is None else speed_end

    # A **ramp**, not a constant factor, when speed_end differs. F58 measured why this matters:
    # with one speed per clip the body-speed target takes 12 distinct values across 32 clips, and
    # the shared decoding head learns the lookup table rather than the quantity -- train loss 0.077
    # against 0.855 on held-out clips, where 1.0 is "predict the mean". Sweeping the speed inside a
    # clip makes the target continuous, so there is no table to memorise, and it raises the
    # insect's between-clip-signal to within-clip-rocking ratio, which is the 1.45-against-7.28 gap
    # that leaves `insect->b1` negative while `b1->insect` transfers.
    #
    # The source path is walked at a rate that varies linearly across the output, so the sampling
    # positions are the running sum of that rate rather than an even spacing. Renormalising to span
    # the path exactly means the clip still covers the same ground, which keeps distance constant
    # and puts the whole speed change into elapsed time -- the same invariant the constant case has.
    mean_rate = 0.5 * (speed + end)
    T2 = max(4, int(round((T - 1) / mean_rate)) + 1)
    rate = np.linspace(speed, end, T2)
    pos = np.concatenate([[0.0], np.cumsum(rate[:-1])])
    pos = pos * (T - 1) / pos[-1]
    src = np.arange(T, dtype=float)
    return {leg: np.stack([np.interp(pos, src, path[:, k]) for k in range(3)], axis=1)
            for leg, path in brel.items()}


def precompute_commands(sim, simIK, scene, brel, scale, ik_iters=1):
    """Kinematic IK -> (T,18) joint commands, leg-major [FL m1..m3, ML ...]."""
    sim.loadScene(f"{ENV}/{scene}")
    T = len(next(iter(brel.values())))
    cmds = np.zeros((T, 18), np.float32)
    target_leg_length = leg_length(sim)
    residuals_mm = []
    col = 0
    for leg in LEGS:
        base = sim.getObjectParent(sim.getObject(f"/m1_{leg}"))
        tip = sim.getObject(f"/foot_{leg}")
        m1_local = np.array(sim.getObjectPosition(sim.getObject(f"/m1_{leg}"), base))
        joints = [sim.getObject(f"/{jn}_{leg}") for jn in SEG]
        target = sim.createDummy(0.01)
        sim.setObjectParent(target, base, True)
        env = simIK.createEnvironment()
        grp = simIK.createGroup(env)
        simIK.addElementFromScene(env, grp, base, tip, target, simIK.constraint_position)
        for t in range(T):
            # One shared absolute Cartesian behavior for every morphology.  IK
            # must therefore produce different joint commands for different links.
            tgt = m1_local + scale * (brel[leg][t] - m1_local)
            sim.setObjectPosition(target, base, list(tgt))
            # **Iterated, because one call is a fixed number of solver steps, not a solution.**
            # The recorded path moves the target smoothly enough that a single call keeps up; a
            # re-timed one does not, and the residual reads as if the target were unreachable when
            # it is simply not converged -- 612 mm against the recording's 37 on foot positions that
            # all came out of that same recording. Extra calls cost nothing when already converged.
            for _ in range(ik_iters):
                simIK.handleGroup(env, grp, {"syncWorlds": True, "allowError": True})
            foot = np.asarray(sim.getObjectPosition(tip, base), dtype=float)
            residuals_mm.append(float(np.linalg.norm(foot - tgt) * 1000.0))
            for k, j in enumerate(joints):
                cmds[t, col + k] = sim.getJointPosition(j)
        sim.removeObjects([target])
        col += 3
    diagnostic = dict(target_leg_length=target_leg_length,
                      scale=scale,
                      residual_mean_mm=float(np.mean(residuals_mm)),
                      residual_max_mm=float(np.max(residuals_mm)))
    return cmds, diagnostic


def drive_and_record(sim, scene, cmds, travel, warmup, cam_dx=0.0, cam_dy=0.0, spawn=None,
                     active_legs=None, remove_legs=None, yaw=0.0, heading=None):
    """Drive cmds with the FIXED camera; returns frames/actions/forces/head.

    **`heading` closes the loop on body direction, and exists to remove an asymmetry we created.**
    The B1's rollout runs a PI controller on heading error and drifts 0.33 deg/s (F78); the insect's
    oscillator was open loop and wandered three times as much -- yaw sd 0.016 against 0.005 in the
    eight conditions where neither robot turns, which is two thirds of the dataset. That difference
    is not a property of the two animals, it is a controller given to one and not the other, and it
    lands squarely in the yaw channel's noise floor (F85). Pass
    `dict(kp=, ki=, recipe=, leg_gain=, leg_off=)` to correct it the same way, by modulating the
    oscillator's own `--spin` term rather than by matching the noise afterwards.

    cam_dx/cam_dy shift the camera in the world plane on top of the scene's authored offset.
    With the authored offset alone the robot starts against the right image edge and stays
    partly outside it for roughly the first two thirds of every clip.
    """
    sim.loadScene(f"{ENV}/{scene}")
    settle(sim)
    active_legs = active_legs or LEGS
    remove_legs = remove_legs or []
    if remove_legs:
        ds, dj = ghost_remove_legs(sim, remove_legs)
        print(f"    ghost-removed {','.join(remove_legs)}: shapes_off={ds}, joints_zero={dj}")
    cam = sim.getObject("/" + SENSOR)
    track = sim.getObject(TRACK)
    body = sim.getObject(ROBOT_ROOT)

    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG]  # matches cmds order
    active_cols = [LEGS.index(leg) * 3 + k for leg in active_legs for k in range(3)]
    cmds = np.asarray(cmds, np.float32)
    if cmds.shape[1] == len(active_cols):
        expanded = np.zeros((len(cmds), len(LEGS) * len(SEG)), np.float32)
        expanded[:, active_cols] = cmds
        cmds = expanded
    elif cmds.shape[1] != len(LEGS) * len(SEG):
        raise ValueError(f"cmds has {cmds.shape[1]} columns; expected {len(active_cols)} "
                         f"for active_legs={active_legs} or {len(LEGS) * len(SEG)} full joints")
    force_h = [sim.getObject(f"/forceSensor_{leg}") for leg in active_legs]

    # authored camera offset (encodes RUNWAY_AIM); must be read BEFORE any respawn, or it
    # measures the camera against the moved robot instead of the authored framing
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off_xy, cam_z = cam0[:2] - trk0[:2], cam0[2]

    # The scene spawns the robot near the floor's edge, which puts the floor corner inside the
    # frame. Re-spawning at the floor centre keeps the edge outside the field of view, and using
    # the same spawn for every embodiment makes them stand on identical floor -- without that,
    # the insect and B1 backgrounds differ across ~27% of pixels.
    if yaw:
        # **Turn the robot to face the way it walks, rather than fight the gait.** The oscillator
        # gait travels along -x for this model whatever phase or sign it is given -- lead 0.25 and
        # 0.75, every combination of lift signs, all of them walk backwards and straight. The
        # camera and the `--travel` gate are both built around a robot crossing the frame along
        # +x, so the cheap fix is to spawn it facing the other way; the alternative is to invert
        # two conventions that every earlier clip depends on.
        # about the world's vertical axis through the robot itself. Adding the angle to the third
        # Euler component instead tilts it -- Euler angles do not compose that way -- and the body
        # ended up at 0.285 m instead of 0.133, on its side rather than turned around.
        root = sim.getObject(ROBOT_ROOT)
        here = sim.getObjectPosition(root, sim.handle_world)
        m = sim.rotateAroundAxis(sim.getObjectMatrix(root, sim.handle_world),
                                 [0.0, 0.0, 1.0], here, float(np.radians(yaw)))
        sim.setObjectMatrix(root, sim.handle_world, m)

    if spawn is not None:
        root = sim.getObject(ROBOT_ROOT)
        pos = sim.getObjectPosition(root, sim.handle_world)
        head = sim.getObjectPosition(track, sim.handle_world)
        sim.setObjectPosition(root, sim.handle_world,
                              [spawn[0] + pos[0] - head[0], spawn[1] + pos[1] - head[1], pos[2]])

    sim.setStepping(True)
    sim.startSimulation()
    # settle holding the first pose
    for _ in range(warmup):
        for h, v in zip(joints, cmds[0]):
            sim.setJointTargetPosition(h, float(v))
        for leg in remove_legs:
            for jn in SEG:
                try:
                    sim.setJointTargetForce(sim.getObject(f"/{jn}_{leg}"), 0.0)
                except Exception:
                    pass
        sim.step()

    frames, actions, forces, heads, oris = [], [], [], [], []
    start_xy = None
    yaw0, yaw_int, prev = None, 0.0, None
    for t in range(len(cmds)):
        step_cmd = cmds[t]
        if heading is not None and prev is not None:
            # **Error is (current - start), and the sign is not free.** Positive `--spin` yaws
            # *negative* -- measured, `spin 0.4` gives omega -0.416, and F75 had to re-collect the
            # turn set at negative spin to make both robots turn the same way. So a body that has
            # drifted positive needs a *positive* trim to come back, which means the error must be
            # (current - start). Written the other way round it is positive feedback: the first
            # attempt drove lateral travel from 0.04 m to 0.41 and failed the walk check.
            err = float(np.arctan2(np.sin(prev - yaw0), np.cos(prev - yaw0)))
            yaw_int = float(np.clip(yaw_int + err * STEP_DT, -2.0, 2.0))
            trim = float(np.clip(heading["kp"] * err + heading["ki"] * yaw_int, -1.0, 1.0))
            step_cmd = cpg_frame(heading["recipe"], heading["leg_gain"], heading["leg_off"],
                                 t, heading["recipe"]["spin"] + trim).astype(np.float32)
        for h, v in zip(joints, step_cmd):
            sim.setJointTargetPosition(h, float(v))
        for leg in remove_legs:
            for jn in SEG:
                try:
                    sim.setJointTargetForce(sim.getObject(f"/{jn}_{leg}"), 0.0)
                except Exception:
                    pass
        sim.step()
        p = np.array(sim.getObjectPosition(track, sim.handle_world))
        if start_xy is None:
            start_xy = p[:2].copy()
            sim.setObjectPosition(cam, sim.handle_world,
                                  [p[0] + off_xy[0] + cam_dx, p[1] + off_xy[1] + cam_dy, cam_z])
        frames.append(capture(sim, cam))
        actions.append(step_cmd[active_cols].copy())
        forces.append(read_forces(sim, force_h))
        heads.append(p)
        # **Body orientation, which was never recorded and is why strafing cannot be checked.**
        # Position alone cannot separate crabbing from turning: both change where the robot goes,
        # and only attitude says whether it is still facing the way it started. It is also three of
        # the six channels of the body pose delta that the shared head should eventually target
        # (F70), and none of them could even be screened without this.
        # **The abdomen, not the head.** `/head` is a segment that swings with every step -- 129
        # degrees of sway across a clip -- so its attitude says almost nothing about where the body
        # is pointing. Measured off the head, straight walking read as a 108 degree turn.
        # **A quaternion, not Euler angles.** The body sits near beta = -84 degrees, a hand's
        # breadth from gimbal lock, where recovering a yaw from three Euler numbers amplifies any
        # convention mismatch into nonsense: straight walking read as 99 degrees of turn with 128
        # degrees of sway, on a body the video shows holding steady.
        q = sim.getObjectQuaternion(body, sim.handle_world)
        oris.append(q)
        if heading is not None:
            x, y, z, w = q
            prev = float(np.arctan2(-2 * (y * z - w * x), -2 * (x * z + w * y)))
            if yaw0 is None:
                yaw0 = prev
        if travel > 0 and float(np.linalg.norm(p[:2] - start_xy)) >= travel:
            break
    sim.stopSimulation(); settle(sim)
    return (np.asarray(frames, np.uint8), np.asarray(actions, np.float32),
            np.asarray(forces, np.float32), np.asarray(heads, np.float32),
            np.asarray(oris, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--episodes", type=str, default="926,521,625",
                    help="comma-separated expert episode indices (each is 66 steps)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="shared absolute foot-path scale about each target body's hip")
    ap.add_argument("--travel", type=float, default=0.8, help="distance gate (m); keeps robot in the fixed frame")
    ap.add_argument("--warmup", type=int, default=20)
    # -0.6 and (0, 0) are not cosmetic and are not a starting point to tune from. The scene
    # anchors the camera to the robot's *start* pose aiming 0.75 m down the runway, which put the
    # robot outside the right image edge walking in: 67 percent of all frames were clipped, and
    # unequally per body, so morphology decodability was partly measuring framing. These values
    # give 0/66 clipped and keep the floor edge out of view (direction_plan.md, PROGRESS §16).
    #
    # **They were recorded as the fix and never made the default**, so every collection run since
    # has had to remember two flags or silently produce clipped data. That happened again on
    # 2026-08-17: 75 clips collected with the old defaults, 56-70 percent of frames touching the
    # right edge, thrown away. The knowledge belongs in the code, not only in the plan.
    ap.add_argument("--cam_dx", type=float, default=-0.6,
                    help="shift the fixed camera along world x; see drive_and_record")
    ap.add_argument("--cam_dy", type=float, default=0.0, help="shift the fixed camera along world y")
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="respawn the robot head at this world x y; use the same value for every\n                         embodiment so they stand on identical floor")
    ap.add_argument("--repeats", type=int, default=1,
                    help="record each (episode,morph) this many times (fresh chaotic draw each) "
                         "-> repeated same-body-same-behavior for the render-lock gate")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="time-scale the shared foot path: >1 walks faster, <1 slower. The "
                         "collected bodies currently sit at Froude 0.155 and the B1 spans "
                         "0.113-0.209, so 0.75-1.35 covers the quadruped's range. Verify the "
                         "achieved speed from the clips rather than trusting this factor, and "
                         "watch the video before training on it.")
    ap.add_argument("--speed_end", type=float, default=None,
                    help="sweep the speed across the clip, from --speed to this. Leave unset for "
                         "a constant factor. A ramp makes the body-speed target continuous instead "
                         "of one value per clip, which is what stops a decoding head memorising "
                         "it (F58).")
    ap.add_argument("--loops", type=int, default=1,
                    help="repeat each 66-step expert foot path into one longer clip")
    ap.add_argument("--behavior", type=str, default="walk",
                    help="behavior label saved in every clip (walk / turn / stop)")
    ap.add_argument("--stop", type=int, default=0,
                    help="if >0: STOP mode — hold the stance for this many frames (no stepping)")
    ap.add_argument("--dump_brel", action="store_true",
                    help="print the recorded foot path's geometry per leg and exit. The first "
                         "synthetic gait was built from this path's mean and range without "
                         "looking at either, and asked for positions half a leg outside the "
                         "workspace: IK residual 365 mm against the recording's 37 mm. Measure "
                         "the shape before generating anything inside it")
    ap.add_argument("--gait", choices=("recorded", "cpg"), default="recorded",
                    help="'recorded' replays the animal's variable wave; 'tripod' generates an "
                         "alternating tripod inside the same reach, measured off that recording")
    ap.add_argument("--cycles", type=float, default=6.0,
                    help="stride cycles per clip, --gait tripod and cpg")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="rotate the robot this many degrees at spawn, so a gait that travels the "
                         "wrong way still crosses the frame along +x")
    ap.add_argument("--symmetric", action="store_true",
                    help="average each left-right pair of the standing pose, removing the animal's "
                         "own asymmetry while keeping the front/middle/hind differences. "
                         "--gait cpg only")
    ap.add_argument("--ft_phase", type=float, default=0.0,
                    help="phase of the extend joint relative to the lift joint, in cycles. 0 is "
                         "the shipped setting, where both run from one signal and the foot path is "
                         "a circle. --gait cpg only")
    ap.add_argument("--legtune", default="",
                    help="npz from scripts/diagnostics/tune_legs.py: per-leg gain and offset that "
                         "make six unequal legs trace the same stroke. --gait cpg only")
    ap.add_argument("--head_kp", type=float, default=0.0,
                    help="proportional gain of a heading controller on the oscillator's own --spin. "
                         "0 leaves the gait open loop, which is how every clip before 2026-08-23 "
                         "was recorded. The B1 has had PI heading control since F78 and drifts 0.33 "
                         "deg/s where the insect wanders three times as much; that asymmetry is a "
                         "controller we gave one robot and not the other, and it lands in the yaw "
                         "channel's noise floor (F85). --gait cpg only")
    ap.add_argument("--head_ki", type=float, default=0.0,
                    help="integral gain. Proportional alone cannot reject a constant disturbance "
                         "below disturbance/gain, which is exactly how the B1's standing bias "
                         "survived for so long")
    ap.add_argument("--spin_amp", type=float, default=None,
                    help="rad; swing amplitude --spin scales, when the gait's own is zero. "
                         "Defaults to --amps[0]")
    ap.add_argument("--strafe", type=float, default=0.0,
                    help="step sideways: an un-mirrored drive on the extend joint, so one side "
                         "pushes out as the other pulls in. --gait cpg only")
    ap.add_argument("--spin", type=float, default=0.0,
                    help="turn on the spot: an un-mirrored drive on the swing joint, so one side "
                         "sweeps forward as the other sweeps back. --gait cpg only")
    ap.add_argument("--mirror", type=int, nargs="*", default=(0, 1, 2), metavar="J",
                    help="which joint indices are mirrored between the sides (0=TC 1=CF 2=FT). "
                         "Whether a joint angle swings a leg the same way on both sides depends on "
                         "which way its axis points, and getting this wrong shows up as a standing "
                         "yaw rather than as a failure")
    ap.add_argument("--lead", type=float, default=0.25,
                    help="how far the sweep leads the lift, in cycles. This sets the direction of "
                         "travel: 0.25 and 0.75 walk opposite ways. --gait cpg only")
    ap.add_argument("--amps", type=float, nargs=3, default=(0.25, 0.20, 0.20),
                    metavar=("TC", "CF", "FT"),
                    help="oscillation amplitude in radians per joint, --gait cpg only")
    ap.add_argument("--ik_iters", type=int, default=1,
                    help="solver calls per frame. 1 reproduces every measurement taken before "
                         "2026-08-21; raise it when the target path is not the smooth recorded one")
    ap.add_argument("--stance", type=float, default=0.9,
                    help="fraction of the recorded stance sweep to use, --gait tripod only. Below "
                         "1.0 because the reachable set narrows toward the extremes of the sweep")
    ap.add_argument("--schedule", type=str, default="",
                    help="piecewise pace as 'rate@fraction' segments, e.g. '1@0.4 0@0.2 1@0.4' to "
                         "walk, stand still for a fifth of the clip, then walk. Rate 0 is a stop. "
                         "Supersedes --speed/--speed_end. The same string works on "
                         "rollout_b1_mujoco.py, so both robots can be given one schedule.")
    ap.add_argument("--morphs", type=str, nargs="+", default=None, metavar="NAME=SCENE",
                    help="bodies to record, e.g. c10f06t10=medauroidea_c10f06t10.ttt. Scene paths "
                         "are relative to sim/env. Names must not contain '_' because "
                         "wm/data/dataset.py reads the body from the filename prefix. "
                         "Defaults to the three uniform-scale bodies.")
    ap.add_argument("--active_legs", type=str, default=",".join(LEGS),
                    help="comma-separated legs to save in actions/forces, leg-major order. "
                         "Use FL,HL,FR,HR for the middle-loss 4-leg insect.")
    ap.add_argument("--remove_legs", type=str, default="",
                    help="comma-separated legs to ghost-remove at runtime, e.g. ML,MR. "
                         "Handles stay present for scene scripts, but shapes are hidden/non-respondable.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    episodes = [int(x) for x in args.episodes.split(",")]
    active_legs = [x for x in args.active_legs.split(",") if x]
    remove_legs = [x for x in args.remove_legs.split(",") if x]
    bad_legs = sorted((set(active_legs) | set(remove_legs)) - set(LEGS))
    if bad_legs:
        raise SystemExit(f"unknown leg(s): {bad_legs}; valid={LEGS}")
    global SCENES
    if args.morphs:
        SCENES = []
        for spec in args.morphs:
            name, _, scene = spec.partition("=")
            if not scene:
                raise SystemExit(f"--morphs wants NAME=SCENE, got {spec!r}")
            if "_" in name:
                raise SystemExit(f"body name {name!r} must not contain '_'")
            if not os.path.exists(os.path.join(ENV, scene)):
                raise SystemExit(f"scene not found: {os.path.join(ENV, scene)}")
            SCENES.append((name, scene))
        print("bodies: " + ", ".join(f"{n} <- {s}" for n, s in SCENES))
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(CSV)
    c = RemoteAPIClient("localhost", port=args.port)
    sim = c.require("sim"); simIK = c.require("simIK")
    settle(sim)

    # ---- STOP mode: hold the walk's first stance (no stepping) ----
    if args.stop > 0:
        rows = list(range(episodes[0] * EP, episodes[0] * EP + EP))
        brel = body_rel_via_fk(sim, df, rows)
        man = []
        for morph, scene in SCENES:
            cmds, ikdiag = precompute_commands(sim, simIK, scene, brel, args.scale,
                                              ik_iters=args.ik_iters)
            if args.gait == "cpg":
                cmds, ikdiag = cpg_commands(sim, scene, EP, cmds.mean(0), cycles=args.cycles,
                                            amps=tuple(args.amps), lead=args.lead,
                                            mirror_joints=tuple(args.mirror),
                                            strafe=args.strafe,
                                            spin=args.spin, spin_amp=args.spin_amp,
                                            ft_phase=args.ft_phase,
                                            symmetric=args.symmetric, legtune=legtune)
            print(f"  {morph:6s} leg={ikdiag['target_leg_length']:.4f}m "
                  f"shared-scale={ikdiag['scale']:.3f} "
                  f"IK residual mean/max={ikdiag['residual_mean_mm']:.2f}/{ikdiag['residual_max_mm']:.2f}mm")
            stance = np.tile(cmds[0], (args.stop, 1))               # hold pose -> stand still
            for rep in range(args.repeats):
                f, a, fc, h, o = drive_and_record(
                    sim, scene, stance, 0.0, args.warmup, args.cam_dx, args.cam_dy, args.spawn,
                    active_legs=active_legs, remove_legs=remove_legs)
                tag = f"{morph}_stop_r{rep}" if args.repeats > 1 else f"{morph}_stop"
                np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                    frames=f, actions=a, forces=fc, head=h, body_quat=o,
                                    foot_order=np.array(active_legs), step_idx=np.arange(len(f)),
                                    morph=morph, expert_episode=-1, repeat=rep, scale=args.scale,
                                    behavior="stop")
                fwd, lat, verdict = walk_check(h)
                man.append(dict(tag=tag, morph=morph, ep=-1, rep=rep, n=len(f),
                                forward=fwd, lateral=lat, verdict=verdict))
                print(f"  {tag:16s} frames={f.shape} forward={fwd:+.2f}m "
                      f"lateral={lat:.2f}m  hip={hip:.3f}m Fr={froude:.3f} wander={wander:.2f}  {verdict}")
        np.save(os.path.join(args.out, "manifest_stop.npy"), man, allow_pickle=True)
        print(f"\n{len(man)} stop clips -> {args.out}")
        return

    legtune = np.load(args.legtune, allow_pickle=True) if args.legtune else None

    manifest = []
    for ep in episodes:
        rows = list(range(ep * EP, ep * EP + EP))
        brel = body_rel_via_fk(sim, df, rows)  # shared Cartesian behavior, once per episode
        if args.dump_brel:
            print(f"\nrecorded foot path, episode {ep}, abdomen frame, metres")
            print(f"{'leg':<5}{'axis':>6}{'min':>10}{'mean':>10}{'max':>10}{'range':>10}")
            for leg, path in brel.items():
                a = np.asarray(path, dtype=float)
                for k, name in enumerate("xyz"):
                    print(f"{leg if k == 0 else '':<5}{name:>6}{a[:,k].min():>10.4f}"
                          f"{a[:,k].mean():>10.4f}{a[:,k].max():>10.4f}{np.ptp(a[:,k]):>10.4f}")
            print("\nthe sweep axis is the one with the largest range; the vertical axis should be")
            print("the one whose mean sits well below its max, since a foot spends most of a stride")
            print("on the ground and only briefly above it")
            return
        brel = (schedule_path(brel, parse_schedule(args.schedule)) if args.schedule
                else retime(brel, args.speed, args.speed_end))
        for morph, scene in SCENES:
            cmds, ikdiag = precompute_commands(sim, simIK, scene, brel, args.scale,
                                              ik_iters=args.ik_iters)
            if args.gait == "cpg":
                cmds, ikdiag = cpg_commands(sim, scene, EP, cmds.mean(0), cycles=args.cycles,
                                            amps=tuple(args.amps), lead=args.lead,
                                            mirror_joints=tuple(args.mirror),
                                            strafe=args.strafe,
                                            spin=args.spin, spin_amp=args.spin_amp,
                                            ft_phase=args.ft_phase,
                                            symmetric=args.symmetric, legtune=legtune)
            print(f"  {morph:6s} leg={ikdiag['target_leg_length']:.4f}m "
                  f"shared-scale={ikdiag['scale']:.3f} "
                  f"IK residual mean/max={ikdiag['residual_mean_mm']:.2f}/{ikdiag['residual_max_mm']:.2f}mm")
            if args.loops > 1:
                cmds = np.tile(cmds, (args.loops, 1))
            pre = "" if args.behavior == "walk" else f"{args.behavior}_"
            for rep in range(args.repeats):
                # only the oscillator can be steered this way: the recorded gait replays a fixed
                # foot path and has no spin term to modulate
                loop = None
                if args.gait == "cpg" and (args.head_kp or args.head_ki):
                    loop = dict(kp=args.head_kp, ki=args.head_ki,
                                recipe={k: ikdiag[k] for k in
                                        ("bias", "amps", "lead", "ft_phase", "mirror_joints",
                                         "strafe", "spin", "spin_amp", "cycles", "frames")},
                                leg_gain=ikdiag["leg_gain"], leg_off=ikdiag["leg_off"])
                f, a, fc, h, o = drive_and_record(
                    sim, scene, cmds, args.travel, args.warmup, args.cam_dx, args.cam_dy, args.spawn,
                    active_legs=active_legs, remove_legs=remove_legs,
                    yaw=args.yaw, heading=loop)  # fresh draw each
                tag = f"{morph}_{pre}ep{ep}_r{rep}" if args.repeats > 1 else f"{morph}_{pre}ep{ep}"
                np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                    frames=f, actions=a, forces=fc, head=h, body_quat=o,
                                    foot_order=np.array(active_legs), step_idx=np.arange(len(f)),
                                    morph=morph, expert_episode=ep, repeat=rep, scale=args.scale,
                                    behavior=args.behavior, schedule=args.schedule,
                                    gait=args.gait)
                fwd, lat, verdict = walk_check(h)
                hip = float(np.median(h[:, 2]))
                # Froude beside the raw distance: the whole cross-robot comparison is
                # dimensionless, so a clip's speed is only meaningful next to its own hip height
                froude = (float(np.linalg.norm(h[-1, :2] - h[0, :2])) / (len(h) * 0.05)
                          / np.sqrt(9.81 * max(hip, 1e-6)))
                # **How straight the path was, which start-and-end distance cannot see.** A body that
                # yaws its way across the floor and happens to finish the right distance away passes
                # `walk_check`; watching the video is what caught it. 1.0 is a straight line. The
                # recorded gait runs 1.28; the first oscillator settings ran 1.44, and wander that is
                # not commanded shows up later as lateral variation belonging to no behaviour (F70).
                xy = h[:, :2]
                net = float(np.linalg.norm(xy[-1] - xy[0]))
                wander = (float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) / net
                          if net > 1e-6 else float('nan'))
                manifest.append(dict(tag=tag, morph=morph, ep=ep, rep=rep, n=len(f),
                                     forward=fwd, lateral=lat, verdict=verdict))
                print(f"  {tag:16s} frames={f.shape} forward={fwd:+.2f}m "
                      f"lateral={lat:.2f}m  hip={hip:.3f}m Fr={froude:.3f} wander={wander:.2f}  {verdict}")

    np.save(os.path.join(args.out, "manifest.npy"), manifest, allow_pickle=True)
    tot = sum(m["n"] for m in manifest)
    print(f"\n{len(manifest)} clips, {tot} frames -> {args.out}")


if __name__ == "__main__":
    main()
