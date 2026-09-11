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
from collect_b1_cpg_babble import ACTION_SCALE, DEFAULT_IL, PHASE, il_to_sdk  # noqa: E402
from b1_coppelia_cpg_controller import (  # noqa: E402
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


def generic_trot_sine_action_at(t, frequency, amplitude, hip_ratio, calf_ratio,
                                calf_phase, noise, rng):
    """One structured generic quadruped sine CPG: hip < thigh < swing-only calf.

    All four legs receive exactly the same three-joint waveform, shifted only by the fixed
    diagonal leg phase.  There is no velocity/turn command or per-leg amplitude adjustment.
    """
    action = np.zeros(12, np.float32)
    leg_phase = np.asarray([0.0, np.pi, np.pi, 0.0])  # FL, FR, RL, RR
    ramp = min(1.0, max(0.0, t / 1.0))
    for li, offset in enumerate(leg_phase):
        phase = 2.0 * np.pi * frequency * t + offset
        action[li] = ramp * hip_ratio * amplitude * np.sin(phase)
        action[4 + li] = ramp * -amplitude * np.sin(phase)
        action[8 + li] = ramp * calf_ratio * amplitude * max(0.0, np.sin(phase + calf_phase))
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


def generic_duty_cycle_action_at(t, frequency, amplitude, calf_ratio, duty_factor,
                                 swing_thigh_lift, noise, rng):
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
    leg_phase = np.asarray([0.0, 0.5, 0.5, 0.0])  # FL, FR, RL, RR, in cycles
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
        action[4 + li] = ramp * thigh
        action[8 + li] = ramp * calf
    action += rng.normal(0.0, noise, size=12).astype(np.float32)
    return action


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
    ap.add_argument("--generic-gait-shape", choices=("sine", "stance-swing", "trot-sine", "duty-cycle"),
                    default="sine")
    ap.add_argument("--generic-calf-ratio", type=float, default=1.5)
    ap.add_argument("--generic-hip-ratio", type=float, default=0.05,
                    help="hip/thigh amplitude ratio for the generic trot-sine CPG")
    ap.add_argument("--generic-calf-phase", type=float, default=-np.pi / 2.0,
                    help="calf phase relative to thigh for the generic trot-sine CPG")
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

    neutral = il_to_sdk(DEFAULT_IL)
    for alias, handle, target in zip(JOINT_ALIASES_SDK, joints, neutral):
        segment = alias.split("_")[1]
        sim.setJointMode(handle, sim.jointmode_dynamic, 0)
        sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setObjectInt32Param(handle, sim.jointintparam_motor_enabled, 1)
        sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_p, args.pid_p)
        sim.setObjectFloatParam(handle, sim.jointfloatparam_pid_d, args.pid_d)
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
                noise=args.noise, rng=rng)
        elif args.cpg_mode == "generic" and args.generic_gait_shape == "duty-cycle":
            action = generic_duty_cycle_action_at(
                step * dt, frequency=generic_parameters["frequency"],
                amplitude=base_amplitude, calf_ratio=args.generic_calf_ratio,
                duty_factor=args.generic_duty_factor,
                swing_thigh_lift=args.generic_swing_thigh_lift,
                noise=args.noise, rng=rng)
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
    print(f"body-frame mean Froude: forward={mean_motion[0]:+.4f} "
          f"lateral={mean_motion[1]:+.4f} yaw={mean_motion[2]:+.4f}")

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
        print("  SCREEN ONLY -- stable outcome recorded; rerun selected config with real frames")
        return

    np.savez_compressed(
        out, frames=np.asarray(frames), base_pos=positions.astype(np.float32),
        base_quat=quaternions.astype(np.float32), uprights=uprights,
        joint_errors=np.asarray(joint_errors), action=np.asarray(actions),
        body_motion=body_motion, foot_contact=np.asarray(contacts),
        foot_pos=np.asarray(foot_positions),
        joint_targets=np.asarray(targets), joint_pos=np.asarray(actuals), dt=dt,
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
