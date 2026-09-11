"""Randomized, designed-primitive CPG pilot for B1 under live CoppeliaSim Bullet dynamics.

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

**Scope, stated honestly.** A diagonal trot plus separate strafe and pivot mechanisms injects
locomotion knowledge for each requested family. This can test whether simple designed primitives
make a useful candidate library; it cannot establish that undirected, no-prior-knowledge babble
does so. Before expanding a pilot into a dataset, tune collection effort toward the goal-source
Froude vocabulary (currently hexapod forward [0.12, 0.19], lateral [-0.12, +0.07], and no truly
yaw-dominant goal), reject weak near-zero/barely-dominant clips, then fit/reuse a checkpoint and
check cross-family nearest neighbours in *predicted* Froude space. True-Froude coverage alone is
not an acceptance test.

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
    cam = sim.getObject(SENSOR_ALIAS)
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
    positions, quaternions, uprights, joint_errors = [], [], [], []
    actions, targets, actuals, contacts, frames = [], [], [], [], []
    for step in range(args.steps):
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
    yaml_path = str(Path(out).with_suffix(".yaml"))
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    print(f"saved run record: {yaml_path}")
    if fell:
        print("  FELL -- rollout NPZ discarded; YAML retained for survival accounting")
        return

    np.savez_compressed(
        out, frames=np.asarray(frames), base_pos=positions.astype(np.float32),
        base_quat=quaternions.astype(np.float32), uprights=uprights,
        joint_errors=np.asarray(joint_errors), action=np.asarray(actions),
        body_motion=body_motion, foot_contact=np.asarray(contacts),
        joint_targets=np.asarray(targets), joint_pos=np.asarray(actuals), dt=dt,
        expert_episode=np.int64(0),
        condition=f"coppelia_babble_f{args.freq}_n{args.noise}_tb{args.turn_bias}"
                  f"_st{args.strafe_amp}_pv{args.pivot_amp}_s{args.seed}",
        joint_order_sdk=np.asarray(JOINT_ALIASES_SDK),
        view=np.asarray("egocentric" if args.ego else "allocentric"),
        ego_seed=np.int64(args.ego_seed))
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
