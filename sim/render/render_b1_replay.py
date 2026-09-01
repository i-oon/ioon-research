"""Render a native-MuJoCo B1 trajectory in CoppeliaSim (kinematic replay) -> dataset .npz.

No physics: each frame sets the base pose + 12 joint angles from the MuJoCo rollout and
grabs the fixed vjepa_cam image. Same renderer + camera as the stick insect, so the
cross-embodiment vision comparison isn't confounded by render style. Motion and
proprioception come from the MuJoCo rollout (where the policy actually walks).

  python3 sim/collect/rollout_b1_mujoco.py --vx 0.4 --steps 300 --out /tmp/b1_traj/walk.npz
  python3 sim/render/render_b1_replay.py --scene sim/env/b1_flat.ttt \
      --traj /tmp/b1_traj/walk.npz --out data/b1_v1 --preview
"""
# **The three camera flags default to the corrected setup, not to the scene as authored.** Rendered
# the scene's own way, the B1 touches an image edge in 61% of frames against the insect's 0%, every
# clip carries its own background because the camera is never pinned, and a 24-degree view reaches
# the far edge of the 15 m floor. `--cam_fov 24 --spawn 0 0 --floor_scale 3` is what
# `data/allocentric/beh12_b1_flat` was built with, and anything rendered differently cannot be mixed with it
# (F113). Pass `--cam_fov 15 --floor_scale 0` to reproduce the old, defective framing.
import argparse
import os
import time

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SENSOR = "vjepa_cam"
JOINT_ALIASES_SDK = [f"{leg}_{seg}_joint"
                     for leg in ("FR", "FL", "RR", "RL")
                     for seg in ("hip", "thigh", "calf")]
ROOT_ALIAS = "base_visual"           # articulation root in the imported B1


def settle(sim):
    while sim.getSimulationState() != 0:
        sim.stopSimulation(); time.sleep(0.1)


def capture(sim, cam):
    sim.handleVisionSensor(cam)
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--cam_dx", type=float, default=0.0, help="shift the fixed camera along world x")
    ap.add_argument("--cam_dy", type=float, default=0.0, help="shift the fixed camera along world y")
    ap.add_argument("--floor_scale", type=float, default=3.0,
                    help="scale the floor about the origin so a wider view does not reach its far "
                         "edge. **`sim.scaleObjects` grows a box without moving its centre**, so a "
                         "3x floor lifts its surface from z=0.000 to z=+0.200 and the robot stands "
                         "20 cm below ground with its feet buried -- which is what the first "
                         "attempt did. The surface is put back at z=0 here (F113).")
    ap.add_argument("--cam_back", type=float, default=1.0,
                    help="multiply the camera's distance from the robot, keeping the scene's "
                         "authored angle. **Kept because it was tried and does not work, and "
                         "the reason is worth not rediscovering**: the B1 sits at image y=0.35 "
                         "where the insect sits at 0.49, so it is framed high and clips the top; "
                         "moving along the optical axis shrinks the robot without moving it in "
                         "frame, and the sideways clips stayed 100%% clipped at 1.7x. Widening "
                         "the angle is the only motion that adds room on the side the robot is "
                         "leaving (F113). Left at 1.0 for every collected set.")
    ap.add_argument("--cam_fov", type=float, default=24.0,
                    help="perspective angle in degrees, overriding the scene's. **The two scenes ship "
                         "identical 15-deg cameras, and that is not the same as an identical view.** "
                         "The field is 2.11 m wide at the robot; the B1 is 1.29 m across and travels "
                         "up to 1.56 m, needing 2.85 m, while the insect needs 1.75 m and fits. "
                         "Matched camera parameters produced a quadruped clipped in 36-100%% of "
                         "frames beside an insect clipped in none (F113). What has to match is that "
                         "both robots stay whole, not that the two numbers agree.")
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="replay from this world x y; use the same value as the insect collector")
    ap.add_argument("--ego", action="store_true",
                    help="mount the camera on the base and look forward; the egocentric de-risk "
                         "gate. **Look at a frame before trusting the orientation default**")
    ap.add_argument("--ego_forward", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="override the measured direction of travel")
    ap.add_argument("--ego_offset", type=float, nargs=3, default=None, metavar=("R", "U", "F"),
                    help="(right, up, forward) in the camera's own basis, metres")
    ap.add_argument("--align_yaw", action="store_true",
                    help="rotate the clip so it starts facing +x, matching the insect's fixed "
                         "spawn. **Required for a paired cross-embodiment set**: without it a slot "
                         "differs in start pose as well as in body")
    ap.add_argument("--ego_seed", type=int, default=0,
                    help="**appearance seed, and it must be PAIRED with the insect's.** The same "
                         "integer produces the same room on either robot, so a cross-embodiment "
                         "test run on matched seeds differs by body and by nothing else. Unmatched "
                         "seeds make Q2 unreadable: 'the coordinate does not transfer' and 'the two "
                         "sets were collected in different-looking rooms' become the same number")
    ap.add_argument("--ego_box", type=float, default=0.0,   # 0 = scaled from the insect's by height
                    help="build a textured room this many metres across; an untextured world "
                         "carries no optical flow and the egocentric view would see nothing")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="stop after this many frames, so every condition yields the same clip "
                         "length regardless of how fast the robot happens to walk")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="sample frames at this rate instead of one per rollout step. The MuJoCo "
                         "rollout logs at 50 Hz (DECIMATION 4 x 5 ms) while the insect collector "
                         "records at 20 Hz, so rendering every step stores a transition worth 20 ms "
                         "on this robot against 50 ms on the other -- the ITM is handed (e_t, e_t+1) "
                         "and 'the transition' then means 2.5x different amounts of time on the two "
                         "sides. Pass 20 to match the insect. Physics stays at 50 Hz; only the "
                         "frames are subsampled, so the policy is untouched.")
    ap.add_argument("--travel", type=float, default=0.0,
                    help="stop once the body has moved this far (m); keeps it inside the fixed frame")
    args = ap.parse_args()

    T = np.load(args.traj)
    base_pos, base_quat, jpos = T["base_pos"], T["base_quat"], T["joint_pos"]
    n = len(base_pos)

    c = RemoteAPIClient("localhost", port=args.port)
    sim = c.require("sim")
    settle(sim)
    sim.loadScene(os.path.abspath(args.scene))
    settle(sim)

    jm = {sim.getObjectAlias(h): h
          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [jm[a] for a in JOINT_ALIASES_SDK]
    sm = {sim.getObjectAlias(h): h
          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root = sm[ROOT_ALIAS]
    cam = sim.getObject("/" + SENSOR)

    # Authored camera offset, read before anything moves. Same convention as the insect
    # collector: the camera is pinned relative to the body's START pose, not re-aimed at the
    # path midpoint, or the body enters and leaves an otherwise static shot.
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world))
    root0 = np.array(sim.getObjectPosition(root, sim.handle_world))
    off_xy, cam_z = cam0[:2] - root0[:2], cam0[2]

    # Replay at the same world point as the insect. Without this B1 stands on a different
    # patch of a 5 m floor and its background differs from the insect's across ~27% of pixels,
    # which a probe would read as an embodiment difference.
    if args.align_yaw:
        # **Every clip starts facing the same way, so a slot pairs across bodies.** The insect
        # spawns identically every time -- [0, 0.01] and 178 degrees in its own aft-referenced
        # convention, which is +x physically -- while each B1 rollout begins wherever MuJoCo left
        # it, from -21.5 to 0 degrees. Without this the two bodies' clips differ in start pose as
        # well as in body, and Q2 cannot tell which caused a coordinate not to transfer.
        #
        # A rigid rotation of the whole trajectory about its own first frame: the physics is
        # untouched because the replay is kinematic, and the room is randomised per seed anyway.
        import sys as _s3
        _s3.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from wm.data.embodiment import heading as _h   # noqa: E402
        psi = float(_h(base_quat[:1], "b1")[0])
        c, sn = np.cos(-psi), np.sin(-psi)
        p0 = base_pos[0, :2].copy()
        d = base_pos[:, :2] - p0
        base_pos = base_pos.copy()
        base_pos[:, 0] = p0[0] + c * d[:, 0] - sn * d[:, 1]
        base_pos[:, 1] = p0[1] + sn * d[:, 0] + c * d[:, 1]
        # rotate the orientation by the same amount, in (w,x,y,z)
        qz = np.array([np.cos(-psi / 2), 0.0, 0.0, np.sin(-psi / 2)])
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        base_quat = np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                              w1*x2 + x1*w2 + y1*z2 - z1*y2,
                              w1*y2 - x1*z2 + y1*w2 + z1*x2,
                              w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)
        print(f"    yaw aligned: start heading {np.degrees(psi):+.1f} -> "
              f"{np.degrees(_h(base_quat[:1], 'b1')[0]):+.1f} deg")

    if args.spawn is not None:
        base_pos = base_pos.copy()
        base_pos[:, 0] += args.spawn[0] - base_pos[0, 0]
        base_pos[:, 1] += args.spawn[1] - base_pos[0, 1]

    if args.ego:
        # **Mounted on the base, and the room it looks at built first.** The replay teleports
        # `root` every frame, so a parented sensor rides it for free; a world-positioned one would
        # have to be recomputed from the same pose and is one convention error away from silently
        # looking somewhere else.
        import sys as _s
        ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _s.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scene"))
        from ego_camera import (attach_ego, build_texture_box,   # noqa: E402
                                randomise_ground, room_for, WALK_PITCH)
        R = room_for(sim.getObjectPosition(root, sim.handle_world)[2])
        if args.ego_box > 0:
            R["size"] = args.ego_box
        build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"],
                          seed=args.ego_seed,
                          centre=(float(base_pos[0, 0]), float(base_pos[0, 1])))
        # ground texture is applied AFTER --floor_scale, further down: `sim.scaleObjects` stretches
        # whatever texture is already on the shape, so texturing first made the B1's floor three
        # times coarser than the insect's -- measured 1.67 against 0.60 of horizontal detail at
        # matched image rows, a ratio of 2.8 against a floor_scale of 3.
        # **Where the body faces, not where it travels.** Deriving forward from the direction of
        # motion was wrong and only showed up on the sideways clips: a crabbing robot travels along
        # -y while facing +x, so the camera swung 90 degrees off the body axis and the B1's
        # sideways view was a different experiment from the insect's, which mounts along
        # `head - abdomen` and therefore always looks where the body points.
        #
        # `heading()` carries the B1's (w,x,y,z)-with-x-forward convention, which is not guessable
        # and has cost this project a week before (F71, F117).
        if args.ego_forward is not None:
            fwd = args.ego_forward
        else:
            import sys as _s2
            _s2.path.insert(0, ROOTDIR)
            from wm.data.embodiment import heading as _heading   # noqa: E402
            psi = float(_heading(base_quat[:1], "b1")[0])
            fwd = [float(np.cos(psi)), float(np.sin(psi)), 0.0]
        _cam_info = attach_ego(sim, cam, root, fwd,
                               args.ego_offset or (0, 0, 0),
                               offset_frac=None if args.ego_offset else R["offset_frac"],
                               pitch_comp=WALK_PITCH["b1"])
        print(f"    ego camera: {_cam_info}")
    else:
        sim.setObjectPosition(cam, sim.handle_world,
                              [base_pos[0, 0] + off_xy[0] + args.cam_dx,
                               base_pos[0, 1] + off_xy[1] + args.cam_dy, cam_z])
    if args.floor_scale > 0:
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                  if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            top = sim.getObject("/Floor")
            def surface():
                q = sim.getObjectPosition(top, sim.handle_world)
                bb = sim.getShapeBB(top)
                return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
            before = surface()
            sim.scaleObjects(floors, float(args.floor_scale), False)
            drop = surface() - before
            for h in floors:                      # put the walking surface back where it was
                q = sim.getObjectPosition(h, sim.handle_world)
                sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - drop])
            print(f"    floor x{args.floor_scale}: surface {before:+.3f} -> {surface():+.3f}")
    if args.ego:
        randomise_ground(sim, seed=args.ego_seed, uv=R["ground_uv"])
    if args.cam_back != 1.0:
        # push the camera away along the line it already looks down, so the framing widens without
        # the lens changing -- the insect's shot, taken from further back
        cam_now = np.array(sim.getObjectPosition(cam, sim.handle_world))
        target = np.array([base_pos[0, 0], base_pos[0, 1], float(base_pos[0, 2])])
        sim.setObjectPosition(cam, sim.handle_world,
                              [float(v) for v in target + (cam_now - target) * args.cam_back])
    if args.cam_fov > 0:
        sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle,
                                float(np.deg2rad(args.cam_fov)))

    # Which rollout steps become frames. Round rather than slice: 50 Hz to 20 Hz is a stride of
    # 2.5, and an integer slice would give 25 Hz and silently leave a 25% timing error in place.
    step = 1.0 / (args.fps * float(T["dt"])) if args.fps > 0 else 1.0
    keep = np.unique(np.round(np.arange(0, n, step)).astype(int))
    keep = keep[keep < n]

    frames = []
    kept_index = []
    for k in keep:
        # B1 covers 1.3-3.1 m depending on commanded speed while the camera sees about 2.1 m,
        # so the walk has to be gated or the body leaves the frame.
        if args.travel > 0 and float(np.linalg.norm(base_pos[k, :2] - base_pos[0, :2])) >= args.travel:
            break
        if args.max_frames and len(frames) >= args.max_frames:
            break
        q = base_quat[k]                                    # (w, x, y, z)
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in base_pos[k]])
        sim.setObjectQuaternion(root, sim.handle_world,
                                [float(q[1]), float(q[2]), float(q[3]), float(q[0])])  # -> (x,y,z,w)
        kept_index.append(int(k))
        for h, a in zip(joints, jpos[k]):
            sim.setJointPosition(h, float(a))
        frames.append(capture(sim, cam))

    frames = np.asarray(frames, np.uint8)
    n = len(frames)   # the travel gate can stop early; everything saved must match
    print(f"rendered {frames.shape} from {len(base_pos)} traj steps  mean px {frames.mean():.1f}")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idx = np.linspace(0, n - 1, 5).astype(int)
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        for i, t in enumerate(idx):
            ax[i].imshow(frames[t]); ax[i].set_title(f"step {t}"); ax[i].axis("off")
        plt.tight_layout(); plt.savefig("/tmp/b1_replay_frames.png", dpi=90)
        print("preview -> /tmp/b1_replay_frames.png")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.traj))[0]
        # Carry the rollout's proprio/command straight through (physics-correct from MuJoCo),
        # **indexed by the steps that were actually rendered**. Slicing `[:n]` instead would take
        # the first n rollout steps, which is the same thing only when every step became a frame --
        # under --fps it silently pairs frame i with the proprioception of a different moment.
        idx = np.asarray(kept_index, int)
        extra = {k: T[k][idx] for k in ("joint_pos", "joint_vel", "action", "command",
                                        "foot_contact", "base_pos", "base_quat")
                 if k in T.files}
        np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                            frames=frames, joint_order_sdk=T["joint_order_sdk"],
                            # from the requested rate, not from idx[1]-idx[0]: a 2.5-step stride
                            # rounds to gaps of 2 and 3 alternately, and reading the first one
                            # reports 0.04 s where the average interval is 0.05
                            dt=(1.0 / args.fps) if args.fps > 0 else float(T["dt"]),
                            fps=float(args.fps) if args.fps > 0 else 1.0 / float(T["dt"]),
                            **extra)
        print(f"saved -> {os.path.join(args.out, tag + '.npz')}")


if __name__ == "__main__":
    main()
