"""Hexapod THIRD-PERSON replay of a recorded clip or physics-loop run (links mode: every shape set to its recorded world
pose), with the scene's own authored camera behind the robot -- the view used for the goal / robot panels of the weekly
physics videos (`weekly_visuals.physics_compare`, `b1_physics_clip`).

History: moved from a scratch script 2026-10-07, hard-coded to two goal clips of the (since deleted) random-room data;
generalised 2026-10-12 to any recorded file with `state_link_pose` and either hexapod body.

Camera: the scene's `vjepa_cam` moved to (first head position + its authored head offset - 0.6 m in x), height kept, as
`collect_ik --view allocentric` (cam_dx -0.6); the clip is re-centred so its first head position is (0, 0); plain floor.

    .venv/bin/python3 scripts/figures/render_hex_allo_goal.py --port P --scene medauroidea_c10f10t10.ttt \\
        --rec a.npz b.npz --out_dir DIR [--tag T]  # -> DIR/allo_[T_]<basename>.npz (frames)
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim/render"))
from render_hex_replay import ENV, SCENE, capture, settle  # noqa: E402


def render_allo(sim, rec_path, scene=SCENE, cam_dx=-0.6):
    rec = np.load(rec_path, allow_pickle=True)
    settle(sim); sim.loadScene(os.path.join(ENV, scene)); settle(sim)
    cam, track = sim.getObject("/vjepa_cam"), sim.getObject("/head")
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world)); trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off, cz = cam0[:2] - trk0[:2], cam0[2]
    lp = np.asarray(rec["state_link_pose"], float).copy()
    shift = -np.asarray(rec["head"][0, :2], float)
    lp[:, :, :2] += shift
    lh = [sim.getObject(str(n)) for n in rec["state_link_names"]]
    h0 = rec["head"][0, :2] + shift
    sim.setObjectPosition(cam, sim.handle_world, [float(h0[0] + off[0] + cam_dx), float(h0[1] + off[1]), float(cz)])
    fr = []
    for t in range(len(lp)):
        for h, p in zip(lh, lp[t]):
            sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        fr.append(capture(sim, cam))
    return np.asarray(fr, np.uint8), shift


def main():
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--scene", default=SCENE, help="medauroidea_c10f10t10.ttt (c10) or medauroidea_c08f09t09.ttt (c08)")
    ap.add_argument("--rec", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="", help="prefix of the output name (e.g. direct / rollout: runs share file names)")
    a = ap.parse_args()
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.rec:
        fr, shift = render_allo(sim, p, a.scene)
        out = os.path.join(a.out_dir, "allo_" + (a.tag + "_" if a.tag else "") + os.path.basename(p))
        np.savez_compressed(out, frames=fr, offset_xy=shift, cam_dx=-0.6, source=np.array(p))
        print(os.path.basename(p), fr.shape, "->", out, flush=True)


if __name__ == "__main__":
    main()
