# Goal hexapod third-person view in the old allocentric setup (collect_ik default camera, plain floor), for weekly_visuals.b1_physics_clip.
# Moved from a scratch script, 2026-10-07.
# allocentric (collect_ik --view allocentric default: authored scene camera, spawn head (0,0), cam_dx -0.6) replay
import sys, numpy as np, os
sys.path.insert(0, "sim/render")
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from render_hex_replay import settle, capture, ENV, SCENE
sim = RemoteAPIClient("localhost", port=25700).require("sim")
for ep in ("ep40110", "ep40030"):
    rec = np.load(f"data/counterfactual_walks/rr_c10_clips_heldout/hexapod_{ep}.npz", allow_pickle=True)
    settle(sim); sim.loadScene(os.path.join(ENV, SCENE)); settle(sim)
    cam, track = sim.getObject("/vjepa_cam"), sim.getObject("/head")
    cam0 = np.array(sim.getObjectPosition(cam, sim.handle_world)); trk0 = np.array(sim.getObjectPosition(track, sim.handle_world))
    off, cz = cam0[:2] - trk0[:2], cam0[2]
    lp = np.asarray(rec["state_link_pose"], float).copy()
    shift = -np.asarray(rec["head"][0, :2], float)
    lp[:, :, :2] += shift
    lh = [sim.getObject(str(n)) for n in rec["state_link_names"]]
    h0 = rec["head"][0, :2] + shift
    sim.setObjectPosition(cam, sim.handle_world, [float(h0[0] + off[0] - 0.6), float(h0[1] + off[1]), float(cz)])
    fr = []
    for t in range(len(lp)):
        for h, p in zip(lh, lp[t]): sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        fr.append(capture(sim, cam))
    fr = np.asarray(fr, np.uint8)
    out = f"results/wm/closed_loop_rr/physics/joint_rr/b1_allo/goal_allo_hexapod_{ep}.npz"
    np.savez_compressed(out, frames=fr, offset_xy=shift, cam_dx=-0.6)
    print(ep, fr.shape, fr.mean(), "->", out)
