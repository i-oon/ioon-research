"""Render a native-MuJoCo B1 trajectory in CoppeliaSim (kinematic replay) -> dataset .npz.

No physics: each frame sets the base pose + 12 joint angles from the MuJoCo rollout and
grabs the fixed vjepa_cam image. Same renderer + camera as the stick insect, so the
cross-embodiment vision comparison isn't confounded by render style. Motion and
proprioception come from the MuJoCo rollout (where the policy actually walks).

  python3 sim/rollout_b1_mujoco.py --vx 0.4 --steps 300 --out /tmp/b1_traj/walk.npz
  python3 sim/render_b1_replay.py --scene sim/env/b1_flat.ttt \
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
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    T = np.load(args.traj)
    base_pos, base_quat, jpos = T["base_pos"], T["base_quat"], T["joint_pos"]
    n = len(base_pos)

    c = RemoteAPIClient("localhost", port=23000)
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

    # Re-aim the fixed camera at the trajectory's mid-x so the whole walk stays in frame
    # (matches the insect's fixed world-frame framing: the body travels through a static shot).
    mid = base_pos[n // 2]
    cur = sim.getObjectPosition(cam, sim.handle_world)
    # keep the camera's existing offset, just recentre its aim on the path midpoint in x
    dx = float(mid[0] - base_pos[0, 0])
    sim.setObjectPosition(cam, sim.handle_world, [cur[0] + dx, cur[1], cur[2]])

    frames = []
    for k in range(n):
        q = base_quat[k]                                    # (w, x, y, z)
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in base_pos[k]])
        sim.setObjectQuaternion(root, sim.handle_world,
                                [float(q[1]), float(q[2]), float(q[3]), float(q[0])])  # -> (x,y,z,w)
        for h, a in zip(joints, jpos[k]):
            sim.setJointPosition(h, float(a))
        frames.append(capture(sim, cam))

    frames = np.asarray(frames, np.uint8)
    print(f"rendered {frames.shape} from {n} traj steps  mean px {frames.mean():.1f}")

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
        # carry the rollout's proprio/command straight through (physics-correct from MuJoCo)
        extra = {k: T[k] for k in ("joint_pos", "joint_vel", "action", "command",
                                   "foot_contact", "base_pos", "base_quat")
                 if k in T.files}
        np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                            frames=frames, joint_order_sdk=T["joint_order_sdk"], **extra)
        print(f"saved -> {os.path.join(args.out, tag + '.npz')}")


if __name__ == "__main__":
    main()
