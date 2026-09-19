"""CPG collection pilots for B1 under live CoppeliaSim Bullet dynamics.

    .venv/bin/python3 sim/collect/collect_b1_coppelia_babble.py --seed 1 \\
        --out data/b1_babble_coppelia/b1_1.npz

**Why this file exists.** `q22_handoff_prompt.md` (2026-09-11) is explicit: no Coppelia-native
babble collector exists, and the CoppeliaSim CPG built so far
(`scripts/diagnostics/objective_experiments/b1_coppelia_cpg_controller.py`) only has three
FAMILY-TARGETED presets (forward/lateral/yaw) -- "these are family-designed motion primitives, not
undirected motor babble" (that file's own docstring). `collect_b1_cpg_babble.py` already built and
verified a genuinely generic, randomized babble mechanism (per-step noise, per-rollout bias/
turn_bias/strafe_amp/pivot_amp, with every sign empirically checked, not assumed) -- but it runs on
MuJoCo. The handoff's own requirement (#3): expert and babble must share "identical scene, Bullet
version, timestep, initialization, camera/room" -- so MuJoCo-collected babble cannot pair with a
Coppelia-native expert controller once one exists. This file ports `collect_b1_cpg_babble.py`'s
verified randomization mechanics onto `b1_coppelia_cpg_controller.py`'s verified live-Bullet
plumbing (scene, joint init, PID gains, camera, telemetry). **Nothing about the signs or bounds
below is re-derived -- every constant is imported or copied verbatim from whichever of the two
files already measured it.**

**Scopes, stated honestly.** `--cpg-mode designed` contains the old forward/lateral/yaw
mechanisms and is diagnostic only: it is not undirected babble. `--cpg-mode generic` follows the
Egocentric-VSM-style collection protocol: a predeclared quadruped CPG family, randomized CPG
parameters, and Gaussian motor noise at every step. It has no command or behaviour-family input.
It retains every rollout, including falls. Froude is logged for later reporting, never used to
retain, reject, or tune an individual generic rollout.

**Real rendered frames, unlike the MuJoCo version.** `collect_b1_cpg_babble.py` stores a `(steps,
4, 4, 3)` placeholder because nothing downstream of it reads vision. This file captures real
`/vjepa_cam` frames every step, matching `b1_coppelia_cpg_controller.py`'s own `--out`/`--video`
behaviour, since a Coppelia-native babble set is exactly the kind of data a vision-based world model
would actually train on.

**Per the handoff's guardrails**: every rollout is validated for the full requested step count (no
early stability declaration), uses `b1_flat_convex.ttt` (the non-convex original hangs Bullet), and
tracks `trunk_respondable` for the dynamics root, not `base_visual`.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))
# 2026-09-18: `ACTION_SCALE`/`DEFAULT_IL`/`PHASE`/`il_to_sdk` now come from
# `b1_coppelia_cpg_controller.py` (already imported below, and no longer via `collect_b1_cpg_babble`
# -- that import broke once already when a parallel session moved that file to `_archive/`;
# `b1_coppelia_cpg_controller.py` now carries its own inlined copy instead of re-exporting).
from b1_coppelia_cpg_controller import (  # noqa: E402
    ACTION_SCALE, DEFAULT_IL, PHASE, il_to_sdk,
    FOOT_ALIASES_SDK, JOINT_ALIASES_SDK, JOINT_LIMITS, LEFT_LEGS, LEGS_IL, MAX_FORCE,
    ROOT_ALIAS, SENSOR_ALIAS, TOUCH_SDK_TO_IL, clip_sdk_targets, settle,
)
from wm.data.embodiment import body_velocity, heading, yaw_rate  # noqa: E402

FRONT_LEGS = {"FL", "FR"}


def babble_action_at(t, freq, thigh_amp, calf_amp, phase_lag, bias, turn_bias, strafe_amp,
                     pivot_amp, noise, rng):
    """Same per-leg mechanics as `collect_b1_cpg_babble.py`'s babble loop -- own-phase diagonal
    trot for thigh/calf, hip channel carries strafe (same sign every leg) and/or pivot (ONE side,
    front-vs-rear opposite sign, tied to the global clock, not own-phase -- see that file's
    `--pivot_amp` docstring for why an own-phase differential is mathematically degenerate here).
    """
    action = np.zeros(12, np.float32)
    global_theta = 2.0 * np.pi * freq * t
    for li, leg in enumerate(LEGS_IL):
        leg_bias = bias + (turn_bias if leg in LEFT_LEGS else -turn_bias)
        swing_amp = max(0.02, thigh_amp + leg_bias)
        ph = 2.0 * np.pi * PHASE[leg] + global_theta
        swing = -swing_amp * np.sin(ph)
        lift = max(0.0, calf_amp * np.sin(ph + phase_lag))
        hip = -strafe_amp * np.sin(ph) if strafe_amp else 0.0
        if pivot_amp and leg in LEFT_LEGS:
            fr_sign = 1.0 if leg in FRONT_LEGS else -1.0
            hip += pivot_amp * thigh_amp * fr_sign * (-np.sin(global_theta))
        action[li] = hip
        action[4 + li] = swing
        action[8 + li] = lift
        if noise > 0.0:
            action[li] += rng.normal(0, noise)
            action[4 + li] += rng.normal(0, noise)
            action[8 + li] += rng.normal(0, noise)
    return action


def generic_cpg_action_at(t, frequency, amplitudes, phases, noise, rng):
    """Apply one shared-frequency sinusoidal CPG plus independent motor noise."""
    return (amplitudes * np.sin(2.0 * np.pi * frequency * t + phases)
            + rng.normal(0.0, noise, size=12)).astype(np.float32)


def generic_stance_swing_action_at(t, frequency, amplitude, calf_ratio, noise, rng):
    """Generic quadruped diagonal stance/swing CPG plus independent motor noise.

    This is a locomotion prior, not a behaviour controller: no requested velocity, turn, strafe,
    body state, or outcome feeds into it. Its fixed phase table and hip/thigh/calf roles must be
    held fixed across the collection; only predeclared sampled parameters and motor noise vary.
    """
    action = np.zeros(12, np.float32)
    leg_phase = np.asarray([0.0, np.pi, np.pi, 0.0])  # FL, FR, RL, RR
    ramp = min(1.0, max(0.0, t / 1.0))
    for li, offset in enumerate(leg_phase):
        phase = 2.0 * np.pi * frequency * t + offset
        action[li] = ramp * 0.1 * amplitude * np.sin(phase + np.pi / 2.0)
        action[4 + li] = ramp * -amplitude * np.sin(phase)
        action[8 + li] = ramp * calf_ratio * amplitude * max(0.0, np.sin(phase - np.pi / 2.0))
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_trot_pair_phases(pairing):
    """Return FL, FR, RL, RR leg phases for one fixed generic gait.

    `diagonal`/`left-right`/`front-rear` are 2-group gaits: exactly 2 legs share each phase, so
    at most 2 feet are ever down at once (a diagonal trot's own structural ceiling -- see
    `q22_handoff_prompt.md`'s reaction-torque/support-margin discussion). `wave` is a genuinely
    different structure: each leg offset by a quarter cycle (the standard real-quadruped walk
    footfall order, LF -> RH -> RF -> LH), so a high enough duty factor keeps 3+ feet down almost
    always -- the biggest support base this generic family can produce, same principle as
    gecko's own diagonal-trot-to-wave-gait fix earlier this session. Still a fixed, predeclared
    phase table -- no feedback, no behaviour-family input.
    """
    if pairing == "diagonal":
        return np.asarray([0.0, np.pi, np.pi, 0.0])
    if pairing == "left-right":
        return np.asarray([0.0, np.pi, 0.0, np.pi])
    if pairing == "front-rear":
        return np.asarray([0.0, 0.0, np.pi, np.pi])
    if pairing == "wave":
        # FL, FR, RL, RR phases. Footfall order by phase is LF(0) -> LH(1/4) -> RF(1/2) ->
        # RH(3/4), which read from a hind leg is LH -> RF -> RH -> LF: the DIAGONAL-sequence walk,
        # where each footfall is on the opposite side from the one before.
        # The comment here used to claim LF -> RH -> RF -> LH (the LATERAL-sequence walk) and was
        # wrong about the array beneath it. Swapping RL/RR on 2026-09-13 to match that comment was
        # measured directly and made it strictly worse: the one config that stood (2.5 Hz, amp
        # 0.60, duty 0.65) fell, and so did all six others tried. Reverted; the comment is the
        # thing that was wrong.
        return np.asarray([0.0, np.pi, 0.5 * np.pi, 1.5 * np.pi])
    raise ValueError(f"unknown generic trot pairing: {pairing}")


def generic_leg_signs(layout):
    """Fixed coordinate sign maps in IL leg order FL, FR, RL, RR."""
    if layout == "same":
        return np.asarray([1.0, 1.0, 1.0, 1.0])
    if layout == "left-right":
        return np.asarray([1.0, -1.0, 1.0, -1.0])
    if layout == "front-rear":
        return np.asarray([1.0, 1.0, -1.0, -1.0])
    if layout == "diagonal":
        return np.asarray([1.0, -1.0, -1.0, 1.0])
    raise ValueError(f"unknown generic sign layout: {layout}")


def generic_trot_sine_action_at(t, frequency, amplitude, hip_ratio, calf_ratio,
                                calf_phase, pairing, thigh_sign_layout,
                                calf_sign_layout, noise, rng):
    """One structured generic quadruped sine CPG: hip < thigh < swing-only calf.

    All four legs receive exactly the same three-joint waveform, shifted only by the fixed
    diagonal leg phase.  There is no velocity/turn command or per-leg amplitude adjustment.
    """
    action = np.zeros(12, np.float32)
    leg_phase = generic_trot_pair_phases(pairing)
    thigh_sign = generic_leg_signs(thigh_sign_layout)
    calf_sign = generic_leg_signs(calf_sign_layout)
    ramp = min(1.0, max(0.0, t / 1.0))
    for li, offset in enumerate(leg_phase):
        phase = 2.0 * np.pi * frequency * t + offset
        action[li] = ramp * hip_ratio * amplitude * np.sin(phase)
        action[4 + li] = ramp * thigh_sign[li] * -amplitude * np.sin(phase)
        action[8 + li] = ramp * calf_sign[li] * calf_ratio * amplitude * max(
            0.0, np.sin(phase + calf_phase))
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_duty_cycle_action_at(t, frequency, amplitude, calf_ratio, duty_factor,
                                 swing_thigh_lift, pairing, thigh_sign_layout,
                                 calf_sign_layout, noise, rng):
    """Generic diagonal walk CPG: slow planted stroke, fast raised return.

    Each leg uses the same phase waveform.  During the stance fraction, the calf is neutral and
    the thigh performs a slow stroke; during the remaining swing fraction, the calf follows one
    positive arch while the thigh returns.  The action contains no requested travel direction or
    body feedback.  A final collection may randomise its CPG parameters, but must retain every
    sampled rollout irrespective of its measured result.
    """
    if not 0.5 <= duty_factor < 1.0:
        raise ValueError("generic duty factor must be in [0.5, 1.0)")
    action = np.zeros(12, np.float32)
    leg_phase = generic_trot_pair_phases(pairing) / (2.0 * np.pi)
    thigh_sign = generic_leg_signs(thigh_sign_layout)
    calf_sign = generic_leg_signs(calf_sign_layout)
    ramp = min(1.0, max(0.0, t / 1.0))
    for li, offset in enumerate(leg_phase):
        cycle = (frequency * t + offset) % 1.0
        if cycle < duty_factor:
            # Planted stroke: front to rear, spread across the longer stance interval.
            thigh = amplitude * (1.0 - 2.0 * cycle / duty_factor)
            calf = 0.0
        else:
            # Raised return: rear to front, with a single smooth clearance arch.
            swing = (cycle - duty_factor) / (1.0 - duty_factor)
            # A common swing-phase thigh bend coordinates the whole serial leg with the knee
            # arch.  It is deliberately identical for every leg; this is not a front-leg fix.
            thigh = (amplitude * (-1.0 + 2.0 * swing)
                      + swing_thigh_lift * amplitude * np.sin(np.pi * swing))
            calf = calf_ratio * amplitude * np.sin(np.pi * swing)
        action[li] = ramp * 0.1 * amplitude
        action[4 + li] = ramp * thigh_sign[li] * thigh
        action[8 + li] = ramp * calf_sign[li] * calf
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_coupled_action_at(t, frequency, amplitude, coupling_ratio, pairing,
                              thigh_sign_layout, noise, rng):
    """Generic quadruped CPG with the calf ALGEBRAICALLY COUPLED to the thigh, not independently
    driven -- the actual mechanism found in Egocentric VSM's own reference implementation
    (`doc/ref/Egocentric_VSM/env_agent.py`'s `move_altas`), not a per-joint feedback controller.
    Their Atlas gait only randomizes ONE number per leg per phase (hip); knee and ankle are fixed
    linear functions of it (`knee = 0.6 - hip`, `ankle = -(hip+knee)`) that keep the foot's
    orientation coherent through the whole stride, by construction, not by reading robot state.

    Every B1 gait shape above independently modulates hip/thigh/calf with separate sines/ratios/
    phases -- nothing enforces the calf stays kinematically coherent with the thigh's own swing,
    which is a real candidate explanation for this project's own measured lateral-roll instability
    (`q22_handoff_prompt.md`'s amplitude/duty-factor push: ruled out the hip channel, ruled out
    duty-factor alone raising the ceiling -- an uncoupled calf/thigh relationship was never tested).

    This is a design-time structural prior (same category as the diagonal-trot phase table or the
    duty-cycle shape already used elsewhere in this file), not real-time feedback -- it never reads
    robot state. `calf = -coupling_ratio * thigh` keeps the foot pointed roughly the same way
    throughout the thigh's swing, the same role Atlas's fixed knee/ankle relationship plays.
    """
    action = np.zeros(12, np.float32)
    leg_phase = generic_trot_pair_phases(pairing) / (2.0 * np.pi)
    thigh_sign = generic_leg_signs(thigh_sign_layout)
    ramp = min(1.0, max(0.0, t / 1.0))
    for li, offset in enumerate(leg_phase):
        phase = 2.0 * np.pi * (frequency * t + offset)
        thigh = -amplitude * np.sin(phase)
        calf = -coupling_ratio * thigh   # foot-orientation-preserving coupling, not independent
        action[4 + li] = ramp * thigh_sign[li] * thigh
        action[8 + li] = ramp * thigh_sign[li] * calf
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_coupled_duty_action_at(t, frequency, amplitude, coupling_ratio, duty_factor,
                                   clearance_ratio, pairing, thigh_sign_layout, noise, rng,
                                   hip_ratio=0.0, hip_bias=None, hip_sign_layout="same",
                                   hip_phase=0.0, hip_clock="leg", stance_calf_bias=0.0):
    """Duty-cycle stance/swing shape (propulsion) with the calf ALGEBRAICALLY COUPLED to the
    thigh during stance (Egocentric-VSM-style foot-orientation coupling, for the stability
    `generic_coupled_action_at` showed but without its own near-zero forward speed -- that
    version never lifts the foot, so it likely drags the whole stride). During swing the
    coupling is broken by one added clearance arc, the same role Atlas's own 3-phase cycle
    plausibly plays implicitly. Still no robot-state feedback of any kind.

    **`hip_ratio`/`hip_bias`: the real trained B1 expert (`data/egocentric/beh12_b1_ego_flat`)
    was checked directly (not assumed) and drives hip at a std comparable to or LARGER than
    thigh (0.1-0.6 across several episodes) plus a real per-leg static offset -- every gait shape
    in this file before this addition left hip at exactly 0. This is a design-time prior derived
    from the real controller's own recorded statistics, not feedback: hip oscillates on the SAME
    phase clock as thigh/calf, `hip_ratio * amplitude` in magnitude, plus an optional fixed
    per-leg bias (4 values, FL/FR/RL/RR order, matching the real data's own asymmetric offsets).
    """
    if not 0.5 <= duty_factor < 1.0:
        raise ValueError("generic duty factor must be in [0.5, 1.0)")
    action = np.zeros(12, np.float32)
    leg_phase = generic_trot_pair_phases(pairing) / (2.0 * np.pi)
    thigh_sign = generic_leg_signs(thigh_sign_layout)
    hip_sign = generic_leg_signs(hip_sign_layout)
    ramp = min(1.0, max(0.0, t / 1.0))
    bias = hip_bias if hip_bias is not None else np.zeros(4)
    for li, offset in enumerate(leg_phase):
        cycle = (frequency * t + offset) % 1.0
        if cycle < duty_factor:
            thigh = amplitude * (1.0 - 2.0 * cycle / duty_factor)
            calf = (-coupling_ratio * thigh
                    + stance_calf_bias * amplitude)  # generic planted-leg extension/preload
        else:
            swing = (cycle - duty_factor) / (1.0 - duty_factor)
            thigh = amplitude * (-1.0 + 2.0 * swing)
            # coupling still holds (keeps returning toward the same foot orientation) PLUS one
            # clearance arc so the foot actually leaves the ground during the return.
            calf = (-coupling_ratio * thigh
                    + clearance_ratio * amplitude * np.sin(np.pi * swing))
        hip_cycle = cycle if hip_clock == "leg" else frequency * t
        hip = ramp * (bias[li] + hip_sign[li] * hip_ratio * amplitude
                      * np.sin(2.0 * np.pi * hip_cycle + hip_phase))
        action[li] = hip
        action[4 + li] = ramp * thigh_sign[li] * thigh
        action[8 + li] = ramp * thigh_sign[li] * calf
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_smooth_duty_action_at(t, frequency, amplitude, coupling_ratio, duty_factor,
                                  clearance_ratio, pairing, thigh_sign_layout, noise, rng,
                                  hip_ratio=0.0, hip_bias=None, hip_sign_layout="same",
                                  hip_phase=0.0, hip_clock="leg", stance_calf_bias=0.0):
    """`coupled-duty` with the thigh's corners removed, and nothing else changed.

    **Why this exists, stated so it can be checked rather than trusted.** `coupled-duty` ramps the
    thigh LINEARLY down through stance and LINEARLY back up through swing, so the joint reverses
    direction instantaneously twice per cycle: commanded acceleration is unbounded at both seams.
    That single property, not the robot and not the engine, is what pinned every gait this file
    could produce. Stance and swing cover the same thigh excursion in different amounts of time,
    so the swing's joint speed is `duty/(1-duty)` times the stance's -- 1.2x at duty 0.55, but 3x
    at duty 0.75. Raising duty to get a third foot on the ground therefore BUYS support by making
    the leg flick back harder, and measured directly, the flick wins: at 2.5 Hz every duty-0.75
    rollout fell while duty 0.65 stood. Being stuck near duty 0.5 forces the 2-feet-down diagonal
    pairing, which in turn forces >= 4.5 Hz to stay upright -- the whole chain follows from the
    corners.

    **The fix carries no task knowledge and adds no parameter.** Stance is reparameterised as
    phase 0..pi and swing as pi..2pi, with the thigh riding `cos` of that phase. Thigh speed goes
    as `sin(phase)`, which is exactly zero at both seams, so the phase rate may jump at the
    stance/swing boundary without the joint's velocity jumping with it: C1 for free, from the
    parameterisation alone. Identical treatment for every leg, no robot state, no behaviour or
    command input, same knobs and same meanings as `coupled-duty` -- duty still sets the stance
    fraction, amplitude still sets the thigh excursion, and the calf coupling, clearance arc and
    hip channel are copied across verbatim.

    It is NOT a claim that this gait is better for the task: whether the reachable Froude range
    changes is an outcome to be measured on a declared, randomised parameter distribution, never a
    target to tune toward.
    """
    if not 0.5 <= duty_factor < 1.0:
        raise ValueError("generic duty factor must be in [0.5, 1.0)")
    action = np.zeros(12, np.float32)
    leg_phase = generic_trot_pair_phases(pairing) / (2.0 * np.pi)
    thigh_sign = generic_leg_signs(thigh_sign_layout)
    hip_sign = generic_leg_signs(hip_sign_layout)
    ramp = min(1.0, max(0.0, t / 1.0))
    bias = hip_bias if hip_bias is not None else np.zeros(4)
    for li, offset in enumerate(leg_phase):
        cycle = (frequency * t + offset) % 1.0
        if cycle < duty_factor:
            theta = np.pi * cycle / duty_factor                       # 0 -> pi over stance
            thigh = amplitude * np.cos(theta)                         # +amp -> -amp
            calf = -coupling_ratio * thigh + stance_calf_bias * amplitude
        else:
            theta = np.pi * (1.0 + (cycle - duty_factor) / (1.0 - duty_factor))   # pi -> 2pi
            thigh = amplitude * np.cos(theta)                         # -amp -> +amp
            # clearance arc, zero at both seams like the thigh's own speed
            calf = (-coupling_ratio * thigh
                    + clearance_ratio * amplitude * np.sin(theta - np.pi))
        hip_cycle = cycle if hip_clock == "leg" else frequency * t
        hip = ramp * (bias[li] + hip_sign[li] * hip_ratio * amplitude
                      * np.sin(2.0 * np.pi * hip_cycle + hip_phase))
        action[li] = hip
        action[4 + li] = ramp * thigh_sign[li] * thigh
        action[8 + li] = ramp * thigh_sign[li] * calf
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def motion_quality_metrics(actions, joint_targets, joint_pos, contacts, foot_pos, dt):
    """Reporting-only quality metrics for babble pilots.

    These are deliberately not used to keep/discard rollouts here.  The final babble collection
    rule remains retain-every-rollout; these values are gates for judging whether a frozen
    generator is smooth/contact-consistent enough before any downstream work is built on it.
    """
    actions = np.asarray(actions, dtype=np.float32)
    joint_targets = np.asarray(joint_targets, dtype=np.float32)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    contacts = np.asarray(contacts, dtype=np.float32)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)

    def finite_diff(x, order):
        y = x
        for _ in range(order):
            if len(y) < 2:
                return np.zeros_like(y)
            y = np.diff(y, axis=0) / dt
        return y

    joint_vel = finite_diff(joint_pos, 1)
    joint_acc = finite_diff(joint_pos, 2)
    joint_jerk = finite_diff(joint_pos, 3)
    target_acc = finite_diff(joint_targets, 2)
    action_acc = finite_diff(actions, 2)
    action_jerk = finite_diff(actions, 3)
    contact_switches = np.abs(np.diff(contacts, axis=0)) if len(contacts) > 1 else np.zeros_like(contacts)
    support_count = contacts.sum(axis=1) if len(contacts) else np.zeros(0, dtype=np.float32)
    foot_z = foot_pos[:, :, 2] if foot_pos.size else np.zeros((0, 4), dtype=np.float32)

    metrics = {
        "joint_vel_rms": float(np.sqrt(np.mean(joint_vel ** 2))) if joint_vel.size else 0.0,
        "joint_acc_rms": float(np.sqrt(np.mean(joint_acc ** 2))) if joint_acc.size else 0.0,
        "joint_acc_var": float(np.var(joint_acc)) if joint_acc.size else 0.0,
        "joint_jerk_rms": float(np.sqrt(np.mean(joint_jerk ** 2))) if joint_jerk.size else 0.0,
        "target_acc_rms": float(np.sqrt(np.mean(target_acc ** 2))) if target_acc.size else 0.0,
        "action_acc_rms": float(np.sqrt(np.mean(action_acc ** 2))) if action_acc.size else 0.0,
        "action_jerk_rms": float(np.sqrt(np.mean(action_jerk ** 2))) if action_jerk.size else 0.0,
        "mean_abs_tracking_error": float(np.mean(np.abs(joint_targets - joint_pos)))
        if joint_targets.size else 0.0,
        "max_abs_tracking_error": float(np.max(np.abs(joint_targets - joint_pos)))
        if joint_targets.size else 0.0,
        "contact_switches_total": int(contact_switches.sum()) if contact_switches.size else 0,
        "contact_switches_per_second": float(contact_switches.sum() / max(dt * max(len(contacts) - 1, 1), dt))
        if len(contacts) else 0.0,
        "contact_duty_per_foot": [float(v) for v in contacts.mean(axis=0)] if len(contacts) else [],
        "support_count_mean": float(support_count.mean()) if support_count.size else 0.0,
        "support_count_min": float(support_count.min()) if support_count.size else 0.0,
        "support_frac_0_or_1_feet": float(np.mean(support_count <= 1.0)) if support_count.size else 0.0,
        "support_frac_2_feet": float(np.mean(support_count == 2.0)) if support_count.size else 0.0,
        "support_frac_3_or_4_feet": float(np.mean(support_count >= 3.0)) if support_count.size else 0.0,
        "foot_z_range_per_foot": [float(v) for v in (foot_z.max(axis=0) - foot_z.min(axis=0))]
        if len(foot_z) else [],
        "foot_z_max_per_foot": [float(v) for v in foot_z.max(axis=0)] if len(foot_z) else [],
    }
    # Conservative preview flags only.  These are intentionally soft and visible; they do not
    # change retention.  The thresholds should be recalibrated after enough B1 pilots exist.
    metrics["preview_quality_flags"] = {
        "low_tracking_error": bool(metrics["max_abs_tracking_error"] < 0.75),
        "not_airborne_mostly": bool(metrics["support_frac_0_or_1_feet"] < 0.50),
        "has_contact_variation": bool(metrics["contact_switches_total"] > 0),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160, help="matches b1_coppelia_cpg_controller.py's "
                    "own validated minimum (guardrail: 100-160 steps hides rollover failure below it)")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--freq", type=float, default=0.5, help="Hz -- b1_coppelia_cpg_controller.py's "
                    "own validated forward-preset frequency, a generic starting point, not re-swept")
    ap.add_argument("--thigh-amp", type=float, default=0.23, help="action units -- matches the "
                    "validated forward preset; amplitudes 0.30-0.40 rolled over even with stronger "
                    "PID gains (measured, see b1_coppelia_cpg_controller.py), do not raise blind")
    ap.add_argument("--calf-amp", type=float, default=0.25)
    ap.add_argument("--phase-lag", type=float, default=-np.pi / 2.0, help="knee-vs-hip phase lag; "
                    "sign verified empirically on MuJoCo (collect_b1_cpg_babble.py), re-verify "
                    "under Bullet before trusting -x_travel here without checking it directly")
    ap.add_argument("--noise", type=float, default=0.03, help="per-step Gaussian noise on the "
                    "commanded action, babble's actual entropy source")
    ap.add_argument("--bias", type=float, default=0.0, help="per-rollout constant thigh-amplitude "
                    "offset, symmetric left/right")
    ap.add_argument("--turn-bias", type=float, default=0.0, help="per-rollout asymmetric thigh "
                    "bias, LEFT +turn_bias RIGHT -turn_bias -- shifts drift within a forward-"
                    "dominant clip, does not by itself make yaw dominant (use --pivot-amp for that,"
                    " per collect_b1_cpg_babble.py's own measured finding)")
    ap.add_argument("--strafe-amp", type=float, default=0.0, help="hip oscillation, same sign every "
                    "leg -- generic strafing primitive")
    ap.add_argument("--pivot-amp", type=float, default=0.0, help="genuine turning, hip channel, "
                    "front-vs-rear opposite sign, ONE side only -- fraction of --thigh-amp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpg-mode", choices=("designed", "generic"), default="designed")
    ap.add_argument("--generic-freq-min", type=float, default=0.5)
    ap.add_argument("--generic-freq-max", type=float, default=2.0)
    ap.add_argument("--generic-amp-min", type=float, default=0.10)
    ap.add_argument("--generic-amp-max", type=float, default=0.80)
    ap.add_argument("--generic-frequency", type=float, default=None,
                    help="fixed shared frequency for a precommitted controlled grid")
    ap.add_argument("--generic-amplitude", type=float, default=None,
                    help="fixed shared base amplitude for a precommitted controlled grid")
    ap.add_argument("--generic-phase-layout", choices=("random", "trot"), default="random",
                    help="trot is one fixed diagonal quadruped phase table, never a behavior mode")
    ap.add_argument("--generic-joint-role-layout", choices=("equal", "quadruped-trot"),
                    default="equal", help="quadruped-trot uses generic hip/thigh/calf amplitude roles")
    ap.add_argument("--generic-gait-shape",
                    choices=("sine", "stance-swing", "trot-sine", "duty-cycle", "coupled",
                             "coupled-duty", "smooth-duty"),
                    default="sine")
    ap.add_argument("--generic-calf-ratio", type=float, default=1.5)
    ap.add_argument("--generic-coupling-ratio", type=float, default=0.6, help="coupled gait shape "
                    "only: calf = -coupling_ratio * thigh, algebraic not independent (see "
                    "generic_coupled_action_at's docstring)")
    ap.add_argument("--generic-clearance-ratio", type=float, default=1.5, help="coupled-duty gait "
                    "shape only: swing clearance arc as a multiple of base amplitude")
    ap.add_argument("--generic-hip-ratio", type=float, default=0.05,
                    help="hip/thigh amplitude ratio for the generic trot-sine CPG; also used by "
                    "coupled-duty when > 0 to actively drive hip on the same phase clock (see "
                    "generic_coupled_duty_action_at's docstring -- the real B1 expert's own "
                    "recorded actions use hip at a magnitude comparable to thigh, not zero)")
    ap.add_argument("--generic-hip-sign-layout",
                    choices=("same", "left-right", "front-rear", "diagonal"), default="same",
                    help="coupled-duty only: fixed generic hip sign convention. This is a sampled "
                    "CPG parameter for lateral/yaw coverage, not a separate behavior primitive")
    ap.add_argument("--generic-hip-phase", type=float, default=0.0,
                    help="coupled-duty only: hip oscillator phase offset, radians")
    ap.add_argument("--generic-hip-clock", choices=("leg", "global"), default="leg",
                    help="coupled-duty only: hip oscillator clock. leg uses each leg's CPG phase; "
                    "global uses one shared clock with the sampled sign layout")
    ap.add_argument("--generic-hip-bias", type=float, nargs=4, default=None,
                    help="coupled-duty only: fixed per-leg hip offset, FL FR RL RR order, added "
                    "on top of the oscillation -- matches the real expert's own static asymmetry")
    ap.add_argument("--generic-stance-calf-bias", type=float, default=0.0,
                    help="coupled-duty only: generic planted-leg calf offset as a multiple of "
                    "base amplitude. Negative usually means more extension/lower stance on B1, "
                    "but this is a sampled CPG parameter, not per-leg tuning")
    ap.add_argument("--generic-calf-phase", type=float, default=-np.pi / 2.0,
                    help="calf phase relative to thigh for the generic trot-sine CPG")
    ap.add_argument("--generic-trot-pairing",
                    choices=("diagonal", "left-right", "front-rear", "wave"), default="diagonal",
                    help="fixed quadruped phase convention. wave is a genuine 4-phase gait (each "
                    "leg 1/4 cycle apart, real quadruped walk footfall order), not a 2-group one "
                    "-- needs duty_factor >= 0.75 to guarantee 3+ feet down at once")
    ap.add_argument("--generic-thigh-sign-layout",
                    choices=("same", "left-right", "front-rear", "diagonal"), default="same",
                    help="fixed joint-axis sign convention for generic trot-sine thighs")
    ap.add_argument("--generic-calf-sign-layout",
                    choices=("same", "left-right", "front-rear", "diagonal"), default="same",
                    help="fixed joint-axis sign convention for generic trot-sine calves")
    ap.add_argument("--generic-duty-factor", type=float, default=0.65,
                    help="fixed stance fraction for the generic duty-cycle CPG")
    ap.add_argument("--generic-swing-thigh-lift", type=float, default=0.0,
                    help="uniform swing-only thigh bend as a multiple of base amplitude")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--pid-p", type=float, default=300.0, help="verified stable value, see "
                    "q22_handoff_prompt.md")
    ap.add_argument("--pid-d", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="", help="optional .mp4, separate from --out's stored frames")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--cam-fov", type=float, default=24.0, help="matches corrected B1 dataset "
                    "rendering convention")
    ap.add_argument("--floor-scale", type=float, default=3.0)
    ap.add_argument("--ego", action="store_true",
                    help="mount /vjepa_cam using the exact egocentric dataset convention")
    ap.add_argument("--ego-seed", type=int, default=0,
                    help="paired room/ground appearance seed for --ego")
    ap.add_argument("--screen-only", action="store_true",
                    help="skip camera/NPZ output; retain the YAML outcome for a fast parameter screen")
    ap.add_argument("--joint-control", choices=("position", "spring"), default="position",
                    help="'spring' runs a per-joint PD INSIDE the physics engine "
                         "(tau = K(q*-q) - C*qdot) instead of Coppelia's generic position PID. "
                         "It is the only way to give these joints a viscous damping term: Bullet "
                         "exposes no joint damping or friction parameter, and that term is what "
                         "makes this same open-loop CPG walk in MuJoCo -- +1.997 m on "
                         "b1_flat_real.xml (system-identified per-joint damping/frictionloss) "
                         "against -0.234 m on the uniform-physics b1_flat.xml, same seed, same "
                         "gait. K and C default to b1_flat_real.xml's own numbers.")
    ap.add_argument("--spring-k", type=float, nargs=3, default=(550.0, 700.0, 970.0),
                    metavar=("HIP", "THIGH", "CALF"),
                    help="b1_flat_real.xml's actuator kp per segment")
    ap.add_argument("--spring-c", type=float, nargs=3, default=(2.745, 4.700, 5.201),
                    metavar=("HIP", "THIGH", "CALF"),
                    help="b1_flat_real.xml's actuator kv PLUS its system-identified joint damping "
                         "(2.0+0.745, 3.0+1.700, 3.0+2.201). The Coulomb frictionloss term "
                         "(0.882/2.698/6.166) has no equivalent in this mode and is NOT modelled.")
    ap.add_argument("--max-joint-vel", type=float, default=None,
                    help="rad/s ceiling on every joint. Bullet has no joint damping/friction "
                         "parameter, so this is the only lever here that bounds swing overshoot. "
                         "Left unset, the scene's 15.6-23.3 rad/s stands.")
    ap.add_argument("--contact-friction", type=float, default=None,
                    help="Bullet friction for the feet and the floor. The scene ships both at 0.50, "
                         "and Bullet MULTIPLIES the two surfaces, so the traction the feet actually "
                         "get is 0.25 -- against 1.0 in the MuJoCo model of the same robot, where "
                         "the same CPG family reaches Froude 0.196 instead of 0.127. Set 1.0 to "
                         "match MuJoCo. Left unset, the scene's own values are untouched.")
    ap.add_argument("--fall-height", type=float, default=0.35, help="same criterion "
                    "b1_coppelia_cpg_controller.py and collect_b1_cpg_babble.py both use")
    args = ap.parse_args()

    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)
    sim.loadScene(os.path.join(ROOT, "sim/env/b1_flat_convex.ttt"))
    settle(sim)
    if sim.getInt32Param(sim.intparam_dynamic_engine) != 0:
        raise RuntimeError("b1_flat_convex.ttt must use Bullet (engine 0)")

    joints_by_name = {sim.getObjectAlias(h): h
                      for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [joints_by_name[name] for name in JOINT_ALIASES_SDK]
    shapes_by_name = {sim.getObjectAlias(h): h
                      for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root = shapes_by_name[ROOT_ALIAS]
    force_sensors = {sim.getObjectAlias(h): h
                     for h in sim.getObjectsInTree(sim.handle_scene, sim.object_forcesensor_type)}
    feet = [force_sensors[name] for name in FOOT_ALIASES_SDK]
    if args.screen_only and args.ego:
        raise ValueError("--screen-only and --ego are mutually exclusive; rerun survivors with --ego")
    cam = None if args.screen_only else sim.getObject(SENSOR_ALIAS)
    if cam is not None:
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                float(np.deg2rad(args.cam_fov)))

    ego_room = None
    if args.ego:
        sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
        from ego_camera import (WALK_PITCH, attach_ego, build_texture_box,  # noqa: E402
                                randomise_ground, room_for)
        camera_root = sim.getObject("/base_visual")
        root_position = sim.getObjectPosition(camera_root, sim.handle_world)
        ego_room = room_for(root_position[2])
        build_texture_box(sim, size=ego_room["size"], height=ego_room["height"],
                          tile=ego_room["tile"], seed=args.ego_seed,
                          centre=(float(root_position[0]), float(root_position[1])))
        qx, qy, qz, qw = sim.getObjectQuaternion(camera_root, sim.handle_world)
        psi = float(heading(np.asarray([[qw, qx, qy, qz]]), "b1")[0])
        info = attach_ego(sim, cam, camera_root,
                          [float(np.cos(psi)), float(np.sin(psi)), 0.0],
                          offset_frac=ego_room["offset_frac"],
                          pitch_comp=WALK_PITCH["b1"])
        print(f"egocentric camera: {info}")

    if args.floor_scale > 0:
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                 if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            top = sim.getObject("/Floor")

            def floor_surface():
                position = sim.getObjectPosition(top, sim.handle_world)
                bounds = sim.getShapeBB(top)
                bounds = bounds[0] if isinstance(bounds[0], list) else bounds
                return position[2] + bounds[2] / 2

            surface_before = floor_surface()
            sim.scaleObjects(floors, float(args.floor_scale), False)
            surface_shift = floor_surface() - surface_before
            for handle in floors:
                position = sim.getObjectPosition(handle, sim.handle_world)
                sim.setObjectPosition(handle, sim.handle_world,
                                      [position[0], position[1], position[2] - surface_shift])
    if args.ego:
        randomise_ground(sim, seed=args.ego_seed, uv=ego_room["ground_uv"])

    if args.contact_friction is not None:
        contact_shapes = [h for name, h in shapes_by_name.items()
                         if name.endswith("_foot_respondable")]
        contact_shapes += [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                          if sim.getObjectAlias(h, 1).startswith("/Floor")]
        for handle in contact_shapes:
            sim.setEngineFloatParam(sim.bullet_body_friction, handle, float(args.contact_friction))
        print(f"contact friction set to {args.contact_friction} on {len(contact_shapes)} shapes "
             f"(feet + floor); scene ships 0.50, Bullet multiplies the pair")

    neutral = il_to_sdk(DEFAULT_IL)
    for alias, handle, target in zip(JOINT_ALIASES_SDK, joints, neutral):
        segment = alias.split("_")[1]
        sim.setJointMode(handle, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setObjectInt32Param(handle, sim.jointintparam_motor_enabled, 1)
        if args.joint_control == "spring":
            k = dict(zip(("hip", "thigh", "calf"), args.spring_k))[segment]
            c = dict(zip(("hip", "thigh", "calf"), args.spring_c))[segment]
            sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_spring)
            sim.setObjectFloatParam(handle, sim.jointfloatparam_kc_k, float(k))
            sim.setObjectFloatParam(handle, sim.jointfloatparam_kc_c, float(c))
        else:
            sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_p, args.pid_p)
            sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_d, args.pid_d)
        if args.max_joint_vel is not None:
            # Bullet exposes NO joint damping or friction parameter (only pospid1/2/3, cfm, erp),
            # so the system-identified viscous+Coulomb terms that make this same CPG walk in MuJoCo
            # -- +1.997 m on b1_flat_real.xml vs -0.234 m on the uniform-physics b1_flat.xml, same
            # seed -- cannot be expressed here at all. A velocity ceiling is not friction, but it
            # bounds the same overshoot the damping term would otherwise absorb. The scene ships
            # 15.6-23.3 rad/s while the gait uses ~1.6.
            sim.setObjectFloatParam(handle, sim.jointfloatparam_maxvel, float(args.max_joint_vel))
        sim.setJointMaxForce(handle, MAX_FORCE[segment])
        sim.setJointPosition(handle, float(target))
        sim.setJointTargetPosition(handle, float(target))

    sim.setStepping(True)
    sim.startSimulation()
    for _ in range(args.warmup):
        sim.step()

    dt = float(sim.getSimulationTimeStep())
    rng = np.random.default_rng(args.seed)
    generic_parameters = None
    if args.cpg_mode == "generic":
        if not (0 <= args.generic_amp_min <= args.generic_amp_max <= 1.0):
            raise ValueError("generic amplitudes must satisfy 0 <= min <= max <= 1")
        if not (0 < args.generic_freq_min <= args.generic_freq_max):
            raise ValueError("generic frequencies must satisfy 0 < min <= max")
        if args.generic_phase_layout == "trot":
            # IL order is four hips, four thighs, four calves; within each group it is
            # FL, FR, RL, RR. FL+RR and FR+RL are the two fixed diagonal pairs. Joint-type
            # offsets create one cyclic leg motion without adding behavior-specific mechanisms.
            leg_phase = np.asarray([0.0, np.pi, np.pi, 0.0])
            phases = np.concatenate((leg_phase + np.pi / 2.0,
                                     leg_phase,
                                     leg_phase - np.pi / 2.0))
        else:
            phases = rng.uniform(-np.pi, np.pi, 12)
        base_amplitude = (float(args.generic_amplitude) if args.generic_amplitude is not None else
                          float(rng.uniform(args.generic_amp_min, args.generic_amp_max)))
        if args.generic_joint_role_layout == "quadruped-trot":
            amplitudes = base_amplitude * np.asarray([0.1] * 4 + [1.0] * 4 + [1.0] * 4)
        else:
            amplitudes = np.full(12, base_amplitude)
        generic_parameters = {
            "frequency": (float(args.generic_frequency) if args.generic_frequency is not None else
                          float(rng.uniform(args.generic_freq_min, args.generic_freq_max))),
            "amplitudes": amplitudes,
            "phases": phases,
        }
    positions, quaternions, uprights, joint_errors = [], [], [], []
    actions, targets, actuals, contacts, foot_positions, frames = [], [], [], [], [], []
    for step in range(args.steps):
        if args.cpg_mode == "generic" and args.generic_gait_shape == "stance-swing":
            action = generic_stance_swing_action_at(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, calf_ratio=args.generic_calf_ratio,
                noise=args.noise, rng=rng)
        elif args.cpg_mode == "generic" and args.generic_gait_shape == "trot-sine":
            action = generic_trot_sine_action_at(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, hip_ratio=args.generic_hip_ratio,
                calf_ratio=args.generic_calf_ratio, calf_phase=args.generic_calf_phase,
                pairing=args.generic_trot_pairing,
                thigh_sign_layout=args.generic_thigh_sign_layout,
                calf_sign_layout=args.generic_calf_sign_layout,
                noise=args.noise, rng=rng)
        elif args.cpg_mode == "generic" and args.generic_gait_shape == "duty-cycle":
            action = generic_duty_cycle_action_at(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, calf_ratio=args.generic_calf_ratio,
                duty_factor=args.generic_duty_factor,
                swing_thigh_lift=args.generic_swing_thigh_lift,
                pairing=args.generic_trot_pairing,
                thigh_sign_layout=args.generic_thigh_sign_layout,
                calf_sign_layout=args.generic_calf_sign_layout,
                noise=args.noise, rng=rng)
        elif args.cpg_mode == "generic" and args.generic_gait_shape == "coupled":
            action = generic_coupled_action_at(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, coupling_ratio=args.generic_coupling_ratio,
                pairing=args.generic_trot_pairing,
                thigh_sign_layout=args.generic_thigh_sign_layout,
                noise=args.noise, rng=rng)
        elif (args.cpg_mode == "generic"
              and args.generic_gait_shape in ("coupled-duty", "smooth-duty")):
            shape_fn = (generic_smooth_duty_action_at
                        if args.generic_gait_shape == "smooth-duty"
                        else generic_coupled_duty_action_at)
            action = shape_fn(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, coupling_ratio=args.generic_coupling_ratio,
                duty_factor=args.generic_duty_factor,
                clearance_ratio=args.generic_clearance_ratio,
                pairing=args.generic_trot_pairing,
                thigh_sign_layout=args.generic_thigh_sign_layout,
                noise=args.noise, rng=rng,
                hip_ratio=args.generic_hip_ratio,
                hip_bias=(np.asarray(args.generic_hip_bias) if args.generic_hip_bias else None),
                hip_sign_layout=args.generic_hip_sign_layout,
                hip_phase=args.generic_hip_phase,
                hip_clock=args.generic_hip_clock,
                stance_calf_bias=args.generic_stance_calf_bias)
        elif args.cpg_mode == "generic":
            action = generic_cpg_action_at(step * dt, noise=args.noise, rng=rng,
                                           **generic_parameters)
        else:
            action = babble_action_at(step * dt, args.freq, args.thigh_amp, args.calf_amp,
                                      args.phase_lag, args.bias, args.turn_bias, args.strafe_amp,
                                      args.pivot_amp, args.noise, rng)
        target = clip_sdk_targets(il_to_sdk(DEFAULT_IL + ACTION_SCALE * action))
        for handle, value in zip(joints, target):
            sim.setJointTargetPosition(handle, float(value))
        sim.step()

        pos = np.asarray(sim.getObjectPosition(root, sim.handle_world), dtype=float)
        qx, qy, qz, qw = sim.getObjectQuaternion(root, sim.handle_world)
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        actual = np.asarray([sim.getJointPosition(h) for h in joints])
        touch_sdk = np.asarray([np.linalg.norm(sim.readForceSensor(h)[1]) for h in feet])
        positions.append(pos)
        quaternions.append([qw, qx, qy, qz])
        uprights.append(upright)
        joint_errors.append(np.max(np.abs(target - actual)))
        actions.append(action.copy())
        targets.append(target.copy())
        actuals.append(actual.copy())
        contacts.append((touch_sdk[TOUCH_SDK_TO_IL] > 1.0).astype(np.float32))
        foot_positions.append(np.asarray([
            sim.getObjectPosition(h, sim.handle_world) for h in feet
        ], dtype=np.float32)[TOUCH_SDK_TO_IL])
        if cam is not None:
            sim.handleVisionSensor(cam)
            buf, res = sim.getVisionSensorImg(cam)
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)
            frames.append(np.flipud(frame).copy())
        if step % 20 == 0:
            print(f"step={step:4d} xyz=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}) "
                  f"up.z={upright:.3f} max|qerr|={joint_errors[-1]:.3f}", flush=True)

    sim.stopSimulation()
    settle(sim)

    positions = np.asarray(positions)
    quaternions = np.asarray(quaternions)
    uprights = np.asarray(uprights)
    fell = positions[:, 2].min() < args.fall_height or uprights.min() < 0.5
    displacement = positions[-1] - positions[0]
    print(f"duration={args.steps * dt:.2f}s displacement="
          f"({displacement[0]:+.3f},{displacement[1]:+.3f},{displacement[2]:+.3f})m")
    print(f"z: min={positions[:, 2].min():.3f} final={positions[-1, 2]:.3f}; "
          f"up.z: min={uprights.min():.3f}; worst_joint_error={max(joint_errors):.3f}rad; "
          f"{'FELL' if fell else 'UPRIGHT'}")
    body_v = body_velocity(positions, quaternions, dt, "b1")
    body_yaw = yaw_rate(quaternions, dt, "b1", float(np.median(positions[:, 2])))
    body_motion = np.concatenate([body_v, body_yaw], axis=1)
    interior = slice(max(1, int(round(0.5 / dt))), -max(1, int(round(0.5 / dt))))
    mean_motion = body_motion[interior].mean(0)
    quality = motion_quality_metrics(actions, targets, actuals, contacts, foot_positions, dt)
    print(f"body-frame mean Froude: forward={mean_motion[0]:+.4f} "
          f"lateral={mean_motion[1]:+.4f} yaw={mean_motion[2]:+.4f}")
    print("motion quality: "
          f"joint_acc_rms={quality['joint_acc_rms']:.3f} "
          f"jerk_rms={quality['joint_jerk_rms']:.3f} "
          f"contact_switches/s={quality['contact_switches_per_second']:.2f} "
          f"support_mean={quality['support_count_mean']:.2f} "
          f"support_<=1={quality['support_frac_0_or_1_feet']:.2f}")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    config = vars(args).copy()
    config.update({
        "scene": "sim/env/b1_flat_convex.ttt",
        "physics_engine": "Bullet",
        "dt": dt,
        "status": "fell" if fell else "upright",
        "min_height": float(positions[:, 2].min()),
        "min_upright": float(uprights.min()),
        "displacement_m": [float(v) for v in displacement],
        "mean_body_froude": [float(v) for v in mean_motion],
        "dominant_family": ("forward", "lateral", "yaw")[int(np.argmax(np.abs(mean_motion)))],
        "view": "egocentric" if args.ego else "allocentric",
        "claim_scope": "designed_cpg_primitives_not_undirected_babble",
        "motion_quality": quality,
    })
    if args.cpg_mode == "generic":
        config["claim_scope"] = "generic_quadruped_cpg_plus_per_step_motor_noise"
        config["retention_policy"] = "retain_every_rollout_including_falls; no_froude_or_outcome_selection"
        config["sampled_cpg"] = {
            "frequency": generic_parameters["frequency"],
            "base_amplitude": base_amplitude,
            "amplitudes": [float(v) for v in generic_parameters["amplitudes"]],
            "phases": [float(v) for v in generic_parameters["phases"]],
        }
    yaml_path = str(Path(out).with_suffix(".yaml"))
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    print(f"saved run record: {yaml_path}")
    if fell and args.cpg_mode != "generic":
        print("  FELL -- rollout NPZ discarded; YAML retained for survival accounting")
        return
    if args.screen_only:
        print("  SCREEN ONLY -- outcome recorded; rerun useful config with real frames")
        return

    np.savez_compressed(
        out, frames=np.asarray(frames), base_pos=positions.astype(np.float32),
        base_quat=quaternions.astype(np.float32), uprights=uprights,
        joint_errors=np.asarray(joint_errors), action=np.asarray(actions),
        body_motion=body_motion, foot_contact=np.asarray(contacts),
        foot_pos=np.asarray(foot_positions),
        joint_targets=np.asarray(targets), joint_pos=np.asarray(actuals), dt=dt,
        motion_quality=np.asarray(yaml.safe_dump(quality, sort_keys=False)),
        expert_episode=np.int64(0),
        condition=(f"coppelia_generic_cpg_s{args.seed}" if args.cpg_mode == "generic" else
                   f"coppelia_babble_f{args.freq}_n{args.noise}_tb{args.turn_bias}"
                   f"_st{args.strafe_amp}_pv{args.pivot_amp}_s{args.seed}"),
        joint_order_sdk=np.asarray(JOINT_ALIASES_SDK),
        view=np.asarray("egocentric" if args.ego else "allocentric"),
        ego_seed=np.int64(args.ego_seed), fell=np.bool_(fell))
    print(f"saved -> {out}")

    if args.video:
        import imageio.v2 as imageio
        video = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(video), exist_ok=True)
        imageio.mimwrite(video, frames, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"saved video: {video}")


if __name__ == "__main__":
    main()
