"""Run a hand-designed diagonal-trot CPG under live CoppeliaSim Bullet dynamics.

This is controller plumbing for future CoppeliaSim-native RL work, not a transplant of either
MuJoCo-trained B1 policy.  It uses the same neutral pose and joint-order conversion as the B1
assets, initializes the actual joint state before starting dynamics, then commands bounded
sinusoidal thigh motion and swing-only calf retraction.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/b1_coppelia_cpg_controller.py
"""
import argparse
import os
import sys
import time

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import body_velocity, heading, yaw_rate  # noqa: E402

# **Inlined rather than imported, deliberately -- same reasoning as `wm/policy/b1_coppelia_env.py`.**
# These were pulled from `collect_b1_cpg_babble.py`, which broke this script's import once already
# when a parallel session moved it to `sim/collect/_archive/`.
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
IL_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])
PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}


def il_to_sdk(v):
    out = np.empty(12)
    out[IL_TO_SDK] = np.asarray(v)
    return out

LEGS_IL = ("FL", "FR", "RL", "RR")
LEFT_LEGS = {"FL", "RL"}
FRONT_LEGS = {"FL", "FR"}
SIDE_PHASE = {"FL": 0.0, "RL": 0.0, "FR": 0.5, "RR": 0.5}
JOINT_ALIASES_SDK = [f"{leg}_{seg}_joint"
                     for leg in ("FR", "FL", "RR", "RL")
                     for seg in ("hip", "thigh", "calf")]
ROOT_ALIAS = "trunk_respondable"
SENSOR_ALIAS = "/vjepa_cam"
FOOT_ALIASES_SDK = [f"{leg}_foot_fixed" for leg in ("FR", "FL", "RR", "RL")]
TOUCH_SDK_TO_IL = np.array([1, 0, 3, 2])
JOINT_LIMITS = {
    "hip": (-0.75, 0.75),
    "thigh": (-1.0, 3.5),
    "calf": (-2.6, -0.6),
}
MAX_FORCE = {"hip": 91.0, "thigh": 93.0, "calf": 140.0}


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation()
        time.sleep(0.1)


def action_at(t, behavior, direction, freq, thigh_amp, calf_amp, phase_lag,
              strafe_amp, pivot_amp, turn_bias, hip_phase_lag, lateral_layout, yaw_layout):
    action = np.zeros(12, np.float32)
    for li, leg in enumerate(LEGS_IL):
        phase = SIDE_PHASE[leg] if behavior == "lateral" else PHASE[leg]
        theta = 2.0 * np.pi * (phase + freq * t)
        global_theta = 2.0 * np.pi * freq * t
        side = 1.0 if leg in LEFT_LEGS else -1.0
        drive_scale = (max(0.0, 1.0 + direction * turn_bias * side)
                       if behavior == "yaw" and yaw_layout == "drive-diff" else 1.0)
        action[4 + li] = -thigh_amp * drive_scale * np.sin(theta)
        action[8 + li] = max(0.0, calf_amp * drive_scale * np.sin(theta + phase_lag))
        if behavior == "lateral":
            mirror = side if lateral_layout == "mirrored" else 1.0
            action[li] = direction * strafe_amp * mirror * np.sin(theta + hip_phase_lag)
        elif behavior == "yaw" and yaw_layout == "same-sign-wave":
            action[li] = -direction * strafe_amp * np.sin(theta + hip_phase_lag)
        elif behavior == "yaw" and yaw_layout != "drive-diff":
            front_rear = 1.0 if leg in FRONT_LEGS else -1.0
            one_side = yaw_layout in ("one-side", "static-one")
            enabled = leg in LEFT_LEGS if one_side else True
            mirror = side if yaw_layout in ("symmetric-mirror", "static-mirror") else 1.0
            if enabled:
                wave = (1.0 if yaw_layout.startswith("static-")
                        else -np.sin(global_theta + hip_phase_lag))
                action[li] = direction * pivot_amp * thigh_amp * front_rear * mirror * wave
    return action


def clip_sdk_targets(target):
    target = np.asarray(target).copy()
    for i, alias in enumerate(JOINT_ALIASES_SDK):
        target[i] = np.clip(target[i], *JOINT_LIMITS[alias.split("_")[1]])
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160, help="controlled steps after warmup")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--freq", type=float, default=0.5, help="CPG frequency in Hz")
    ap.add_argument("--thigh-amp", type=float, default=None,
                    help="amplitude in action units; defaults to the validated behavior preset")
    ap.add_argument("--calf-amp", type=float, default=None,
                    help="swing retraction; defaults to the validated behavior preset")
    ap.add_argument("--phase-lag", type=float, default=-np.pi / 2.0)
    ap.add_argument("--behavior", choices=("forward", "lateral", "yaw"), default="forward")
    ap.add_argument("--direction", type=float, choices=(-1.0, 1.0), default=1.0,
                    help="sign of lateral/yaw command")
    ap.add_argument("--strafe-amp", type=float, default=None,
                    help="hip oscillation; defaults to the validated lateral/yaw preset")
    ap.add_argument("--pivot-amp", type=float, default=1.5,
                    help="yaw hip differential as a multiple of thigh amplitude")
    ap.add_argument("--turn-bias", type=float, default=0.6,
                    help="left/right drive asymmetry for yaw-layout=drive-diff")
    ap.add_argument("--hip-phase-lag", type=float, default=0.0,
                    help="phase offset of lateral/yaw hip motion relative to the CPG")
    ap.add_argument("--lateral-layout", choices=("mirrored", "same-sign"), default="mirrored",
                    help="mirror hip commands because B1's left/right joint axes are opposite")
    ap.add_argument("--yaw-layout",
                    choices=("drive-diff", "one-side", "symmetric", "symmetric-mirror",
                             "static-one", "static-symmetric", "static-mirror",
                             "same-sign-wave"),
                    default=None, help="spatial hip pattern; defaults to the validated preset")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--pid-p", type=float, default=300.0,
                    help="Coppelia position-controller proportional coefficient")
    ap.add_argument("--pid-d", type=float, default=5.0,
                    help="Coppelia position-controller derivative coefficient")
    ap.add_argument("--out", default="",
                    help="optional .npz path; stores rendered frames and controller telemetry")
    ap.add_argument("--video", default="",
                    help="optional .mp4 path rendered from /vjepa_cam")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="output video rate; CoppeliaSim steps remain unchanged")
    ap.add_argument("--cam-fov", type=float, default=24.0,
                    help="camera FOV in degrees; 24 matches corrected B1 dataset rendering")
    ap.add_argument("--floor-scale", type=float, default=3.0,
                    help="floor scale used by corrected B1 dataset rendering")
    ap.add_argument("--ego", action="store_true",
                    help="mount /vjepa_cam using the exact egocentric dataset convention")
    ap.add_argument("--ego-seed", type=int, default=0,
                    help="paired room/ground appearance seed for --ego")
    args = ap.parse_args()
    presets = {
        "forward": dict(thigh_amp=0.23, calf_amp=0.25, strafe_amp=0.0,
                        yaw_layout="same-sign-wave"),
        "lateral": dict(thigh_amp=0.0, calf_amp=0.18, strafe_amp=1.10,
                        yaw_layout="same-sign-wave"),
        "yaw": dict(thigh_amp=0.0, calf_amp=0.18, strafe_amp=0.40,
                    yaw_layout="same-sign-wave"),
    }[args.behavior]
    for name, value in presets.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.require("sim")
    settle(sim)
    sim.loadScene(os.path.join(ROOT, "sim/env/b1_flat_convex.ttt"))
    settle(sim)
    if sim.getInt32Param(sim.intparam_dynamic_engine) != 0:
        raise RuntimeError("b1_flat_convex.ttt must use Bullet (engine 0)")

    joints_by_name = {
        sim.getObjectAlias(h): h
        for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)
    }
    joints = [joints_by_name[name] for name in JOINT_ALIASES_SDK]
    shapes_by_name = {
        sim.getObjectAlias(h): h
        for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
    }
    root = shapes_by_name[ROOT_ALIAS]
    force_sensors = {
        sim.getObjectAlias(h): h
        for h in sim.getObjectsInTree(sim.handle_scene, sim.object_forcesensor_type)
    }
    feet = [force_sensors[name] for name in FOOT_ALIASES_SDK]
    cam = sim.getObject(SENSOR_ALIAS) if args.out or args.video else None
    if cam is not None:
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                float(np.deg2rad(args.cam_fov)))
    ego_room = None
    if args.ego:
        if cam is None:
            raise ValueError("--ego requires --out or --video")
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
        sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode,
                                sim.jointdynctrl_position)
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
    positions, quaternions, uprights, joint_errors = [], [], [], []
    actions, targets, actuals, contacts, frames = [], [], [], [], []
    for step in range(args.steps):
        action = action_at(step * dt, args.behavior, args.direction, args.freq,
                           args.thigh_amp, args.calf_amp, args.phase_lag,
                           args.strafe_amp, args.pivot_amp, args.turn_bias, args.hip_phase_lag,
                           args.lateral_layout,
                           args.yaw_layout)
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
    displacement = positions[-1] - positions[0]
    body_v = body_velocity(positions, quaternions, dt, "b1")
    body_yaw = yaw_rate(quaternions, dt, "b1", float(np.median(positions[:, 2])))
    body_motion = np.concatenate([body_v, body_yaw], axis=1)
    interior = slice(max(1, int(round(0.5 / dt))), -max(1, int(round(0.5 / dt))))
    mean_motion = body_motion[interior].mean(0)
    yaw_change = np.degrees(np.unwrap(heading(quaternions, "b1"))[-1]
                            - np.unwrap(heading(quaternions, "b1"))[0])
    fell = positions[:, 2].min() < 0.35 or uprights.min() < 0.5
    print(f"duration={args.steps * dt:.2f}s displacement="
          f"({displacement[0]:+.3f},{displacement[1]:+.3f},{displacement[2]:+.3f})m")
    print(f"z: min={positions[:, 2].min():.3f} final={positions[-1, 2]:.3f}; "
          f"up.z: min={uprights.min():.3f} final={uprights[-1]:.3f}; "
          f"worst_joint_error={max(joint_errors):.3f}rad; {'FELL' if fell else 'UPRIGHT'}")
    print(f"body-frame mean Froude: forward={mean_motion[0]:+.4f} "
          f"lateral={mean_motion[1]:+.4f} yaw={mean_motion[2]:+.4f}; "
          f"yaw_change={yaw_change:+.1f}deg")

    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, frames=np.asarray(frames), positions=positions,
                            base_pos=positions.astype(np.float32),
                            base_quat=quaternions.astype(np.float32),
                            uprights=uprights, joint_errors=np.asarray(joint_errors),
                            actions=np.asarray(actions), action=np.asarray(actions),
                            body_motion=body_motion, foot_contact=np.asarray(contacts),
                            joint_targets=np.asarray(targets),
                            joint_positions=np.asarray(actuals), joint_pos=np.asarray(actuals),
                            dt=dt, expert_episode=np.int64(0),
                            condition=f"coppelia_{args.behavior}_d{args.direction:+.0f}",
                            joint_order_sdk=np.asarray(JOINT_ALIASES_SDK))
        print(f"saved rollout: {out}")
    if args.video:
        import imageio.v2 as imageio
        video = os.path.abspath(args.video)
        os.makedirs(os.path.dirname(video), exist_ok=True)
        imageio.mimwrite(video, frames, fps=args.fps, codec="libx264", quality=8,
                         macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"saved video: {video}")


if __name__ == "__main__":
    main()
