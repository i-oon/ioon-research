"""Render a native-MuJoCo B1 trajectory in CoppeliaSim (kinematic replay) -> dataset .npz.

No physics: each frame sets the base pose + 12 joint angles from the MuJoCo rollout and
grabs the fixed vjepa_cam image. Same renderer + camera as the stick insect, so the
cross-embodiment vision comparison isn't confounded by render style. Motion and
proprioception come from the MuJoCo rollout (where the policy actually walks).

  python3 sim/collect/rollout_b1_mujoco.py --vx 0.4 --steps 300 --out /tmp/b1_traj/walk.npz
  python3 sim/render/render_b1_replay.py --scene sim/env/b1_flat.ttt \
      --traj /tmp/b1_traj/walk.npz --out data/b1_v1 --preview
"""
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
    ap.add_argument("--floor_scale", type=float, default=0.0,
                    help="scale the floor about the world origin. The scene ships 15 m of floor, and a 25-deg view reaches its far edge -- the straight band across the upper third of every wide B1 frame, which the insect's 15-deg view never reaches. Raising the lights does not remove it; only more floor does.")
    ap.add_argument("--light_z", type=float, default=0.0,
                    help="raise the four default lights to this height. The scene ships them at 2.5 m, which lights a pool narrower than a 25-deg view, so the floor beyond it reads as a hard band across the top of every frame -- the thing that makes the wide shot look unlike the insect's. Widening the view without this trades a clipped robot for a visible lighting edge.")
    ap.add_argument("--cam_fov", type=float, default=0.0,
                    help="perspective angle in degrees, overriding the scene's. **The two scenes ship "
                         "identical 15-deg cameras, and that is not the same as an identical view.** "
                         "The field is 2.11 m wide at the robot; the B1 is 1.29 m across and travels "
                         "up to 1.56 m, needing 2.85 m, while the insect needs 1.75 m and fits. "
                         "Matched camera parameters produced a quadruped clipped in 36-100% of "
                         "frames beside an insect clipped in none (F113). What has to match is that "
                         "both robots stay whole, not that the two numbers agree.")
    ap.add_argument("--spawn", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="replay from this world x y; use the same value as the insect collector")
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
    if args.spawn is not None:
        base_pos = base_pos.copy()
        base_pos[:, 0] += args.spawn[0] - base_pos[0, 0]
        base_pos[:, 1] += args.spawn[1] - base_pos[0, 1]

    sim.setObjectPosition(cam, sim.handle_world,
                          [base_pos[0, 0] + off_xy[0] + args.cam_dx,
                           base_pos[0, 1] + off_xy[1] + args.cam_dy, cam_z])
    if args.floor_scale > 0:
        floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
                  if sim.getObjectAlias(h, 1).startswith("/Floor")]
        if floors:
            sim.scaleObjects(floors, float(args.floor_scale), False)
    if args.light_z > 0:
        for h in sim.getObjectsInTree(sim.handle_scene):
            if sim.getObjectType(h) == sim.object_light_type:
                q = sim.getObjectPosition(h, sim.handle_world)
                sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], float(args.light_z)])
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
