"""Is the hexapod's and the B1's egocentric RENDERING matched, independent of how each body moves?

Renders both scenes from the SAME camera poses with no body in view and no walking: the room, floor
and camera height are built exactly as each collector builds them (room_for scaled by the scene's own
mount height; scale_floor; randomise_ground, the B1 with the matched floor-texture density x1.5), the
same room seeds on both sides, the same poses in room-scaled units (position, heading), camera level at
90 deg FOV. Then the V-JEPA2 body-identity probe (as body_id_render_check.py) on hexapod-scene vs
B1-scene frames, grouped by room.

  probe ~0.5  -> the scenes render alike; any separation in real data comes from the bodies
  probe high  -> a rendering difference remains (reported per image region)

    .venv/bin/python3 scripts/diagnostics/egocentric_view/static_scene_render_check.py
Needs CoppeliaSim on port 23000; V-JEPA2 on the CPU by default (float32).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "sim/scene"),
                os.path.join(ROOT, "scripts/diagnostics/egocentric_view")]
from ego_camera import (_basis, build_texture_box, randomise_ground, room_for, scale_floor,  # noqa: E402
                        set_ego_fov)
from body_id_render_check import probe  # noqa: E402

SCENES = {"hexapod": ("sim/env/medauroidea_c10f10t10.ttt", "/head", 1.0),
          "b1": ("sim/env/b1_flat.ttt", "/base_visual", 1.5)}   # (scene, mount object, ground-uv mult)


def grab(sim, cam):
    buf, res = sim.getVisionSensorImg(cam)
    return np.flipud(np.frombuffer(buf, dtype=np.uint8).reshape(res[1], res[0], 3)).copy()


def render_scene(sim, name, rooms, poses):
    scene, mount_name, uv_mult = SCENES[name]
    frames = []
    for seed in range(rooms):
        sim.loadScene(os.path.join(ROOT, scene))
        mz = float(sim.getObjectPosition(sim.getObject(mount_name), sim.handle_world)[2])
        R = room_for(mz)
        build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed, centre=(0.0, 0.0))
        scale_floor(sim, R["size"])
        randomise_ground(sim, seed=seed, uv=R["ground_uv"] * uv_mult)
        # hide every shape that is not the room or the floor, so no body part can enter the view
        for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type):
            a = sim.getObjectAlias(h, 1)
            if not (a.startswith("/Floor") or a.startswith("/egoWall")):
                sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        cam = sim.getObject("/vjepa_cam")
        sim.setObjectParent(cam, -1, True)
        set_ego_fov(sim, cam)
        cam_h = mz * (1.0 + R["offset_frac"][1])
        sim.startSimulation()
        set_ego_fov(sim, cam)                      # startSimulation restores the authored lens (F256)
        for px, py, yaw in poses:
            f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            x, y, z = _basis(f)
            p = np.array([px * R["size"], py * R["size"], cam_h])
            sim.setObjectMatrix(cam, sim.handle_world,
                                [float(x[0]), float(y[0]), float(z[0]), float(p[0]),
                                 float(x[1]), float(y[1]), float(z[1]), float(p[1]),
                                 float(x[2]), float(y[2]), float(z[2]), float(p[2])])
            sim.handleVisionSensor(cam)
            frames.append((seed, grab(sim, cam)))
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            pass
        print(f"  {name} room {seed + 1}/{rooms}: mount {mz:.3f} m, room {R['size']:.1f} m", flush=True)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=16)
    ap.add_argument("--poses", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/cache/static_scene_render.npz")
    ap.add_argument("--b1_uv_mult", type=float, default=1.5, help="B1 floor-texture multiplier (1.0 = pure scaling)")
    args = ap.parse_args()
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
    rng = np.random.default_rng(0)
    # positions as fractions of the room size (inside the middle half), heading anywhere
    poses = [(rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(-np.pi, np.pi))
             for _ in range(args.poses)]
    SCENES["b1"] = SCENES["b1"][:2] + (args.b1_uv_mult,)
    out = {name: render_scene(sim, name, args.rooms, poses) for name in SCENES}
    np.savez_compressed(os.path.join(ROOT, args.out),
                        **{f"{n}_frames": np.stack([f for _, f in v]) for n, v in out.items()},
                        **{f"{n}_room": np.array([s for s, _ in v]) for n, v in out.items()})
    from vjepa2_encoder import VJEPA2FrameEncoder
    from wm.evaluate import encode_clip
    enc = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    grid = lambda e: e.float().reshape(-1, 16, 16, e.shape[-1])
    groups = {}
    for n, v in out.items():
        frames = np.stack([f for _, f in v]); rooms = np.array([s for s, _ in v])
        E = grid(encode_clip(enc, frames, 4).cpu())
        groups[n] = [E[rooms == r] for r in range(args.rooms)]    # one group per room
        print(f"  encoded {n}: {len(frames)} frames", flush=True)
    r = probe(groups["hexapod"], groups["b1"])
    print("static scenes, hexapod vs B1: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    print("STATIC_SCENE_CHECK_DONE")


if __name__ == "__main__":
    main()
