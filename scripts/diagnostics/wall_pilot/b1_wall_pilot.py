"""B1 wall pilot (2026-10-09, Ajan Go week-19 direction: predictive wall avoidance).

Question: (1) can the frozen V-JEPA2 embedding read the distance to the wall ahead, and (2) does the current forward model
(FTM) keep that information after 1/2/4/8 predicted steps? Only if (1) holds is a rollout "about to hit the wall" score
possible at all; (2) says whether the existing model can be used or needs training on wall data.

Data: windows of B1's own long forward walks (`data/counterfactual_walks/b1_walks/speed_*.npz`, exact MuJoCo poses, policy
walking), replayed (render only, physics unchanged) in the ORIGINAL room recipe (sized to the body, 17.65 m, room seed per
clip; render_shift_heldout.b1_render without override). Each window is rotated so the robot faces +x at its first frame and
the room centre is placed so the front wall's inner face is D_end metres ahead of the camera at the LAST frame. B1's physics
has no walls (it would walk through them): windows are chosen to end before the wall. Labels per frame: distance from the
camera to the front wall along +x (`wall_dist`), from the recorded camera pose.

    PY=.venv/bin/python3; S=scripts/diagnostics/wall_pilot/b1_wall_pilot.py
    $PY $S render --port P          # -> data/wall_pilot/b1/*.npz (+ manifest)
    $PY $S sheet                    # -> results/check/wall_pilot/sheet_b1.png, video_b1.mp4
    $PY $S probe [--ckpt ...]       # -> results/check/wall_pilot/probe_b1.txt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/dataset", "sim/collect", "sim/scene", "sim/render"):
    sys.path.insert(0, os.path.join(ROOT, p))

OUT = os.path.join(ROOT, "data/wall_pilot/b1")
CHK = os.path.join(ROOT, "results/check/wall_pilot")
WALKS = ("speed_c5.8", "speed_c7.1", "speed_c8.15", "speed_c8.8")
D_END = (1.0, 2.0, 4.0, 6.0)      # camera-to-wall distance at the last frame (m); below 1 m the wall fills the 90 deg view
EP = 66
WALL_T = 0.05                     # wall thickness (ego_camera.build_texture_box default)


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def jobs():
    J = []
    for wi, w in enumerate(WALKS):
        W = np.load(os.path.join(ROOT, f"data/counterfactual_walks/b1_walks/{w}.npz"), allow_pickle=True)
        n = len(W["base_pos"])
        starts = np.linspace(100, n - EP - 1, 6).astype(int)          # 6 windows per walk, spread over the walk
        for k, s in enumerate(starts):
            d_end = D_END[k % len(D_END)]
            seed = 400 + 6 * wi + k                                     # rooms 400-423, unused elsewhere
            J.append(dict(walk=w, start=int(s), d_end=float(d_end), room_seed=int(seed),
                          name=f"b1wall_{w}_s{int(s):04d}_d{d_end:.1f}"))
    return J


def do_render(a):
    import render_shift_heldout as RS
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from wm.data.embodiment import heading
    os.makedirs(OUT, exist_ok=True)
    sim = RemoteAPIClient("localhost", port=a.port).getObject("sim")
    man = []
    for j in jobs():
        dst = os.path.join(OUT, j["name"] + ".npz")
        W = np.load(os.path.join(ROOT, f"data/counterfactual_walks/b1_walks/{j['walk']}.npz"), allow_pickle=True)
        sl = slice(j["start"], j["start"] + EP)
        pos, quat, jp = (np.asarray(W[k][sl], float) for k in ("base_pos", "base_quat", "joint_pos"))
        psi0 = float(heading(quat[:1], "b1")[0])
        c, s = np.cos(-psi0), np.sin(-psi0)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p0 = pos[0] * [1, 1, 0]
        pos_r = (pos - p0) @ R.T
        qz = np.array([np.cos(-psi0 / 2), 0, 0, np.sin(-psi0 / 2)])
        quat_r = np.stack([qmul(qz, q) for q in quat])
        d = dict(base_pos=pos_r, base_quat=quat_r, joint_pos=jp, room_seed=np.int64(j["room_seed"]))
        # first pass with the room centred on the start, to read the camera's x at the last frame and the room size
        fr, cp, Rm = RS.b1_render(sim, d, np.zeros(2))
        half = float(Rm["size"]) / 2 - WALL_T / 2
        cam_x_end = float(cp[-1][0]) if np.asarray(cp).ndim == 2 else float(np.asarray(cp)[-1, 0, 3])
        cx = cam_x_end + j["d_end"] - half                              # front wall inner face at cam_x_end + d_end
        fr, cp, Rm = RS.b1_render(sim, d, np.array([cx, 0.0]))
        cp = np.asarray(cp, float)
        cam_x = cp[:, 0] if cp.ndim == 2 else cp[:, 0, 3]
        wall = cx + half - cam_x
        np.savez_compressed(dst[:-4] + ".tmp.npz", frames=fr, base_pos=pos_r, base_quat=quat_r, joint_pos=jp,
                            joint_vel=np.asarray(W["joint_vel"][sl]), action=np.asarray(W["action"][sl]),
                            command=np.asarray(W["command"][sl]), com_pos=(np.asarray(W["com_pos"][sl], float) - p0) @ R.T,
                            foot_contact=np.asarray(W["foot_contact"][sl]), cam_pose=cp, wall_dist=wall.astype(np.float32),
                            room_seed=np.int64(j["room_seed"]), room_size=np.float64(Rm["size"]), room_centre=np.array([cx, 0.0]),
                            source_walk=np.array(j["walk"]), window_start=np.int64(j["start"]), condition=np.array(j["walk"]),
                            dt=np.float32(0.05), embodiment=np.array("b1"))
        os.replace(dst[:-4] + ".tmp.npz", dst)
        man.append(dict(j, wall_first=float(wall[0]), wall_last=float(wall[-1])))
        print(f"{j['name']}: wall {wall[0]:.2f} -> {wall[-1]:.2f} m, room {Rm['size']:.2f} m", flush=True)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


def do_sheet(a):
    from PIL import Image, ImageDraw
    os.makedirs(CHK, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    rows = []
    for f in fs[::2][:8]:
        d = np.load(f, allow_pickle=True)
        tiles = []
        for t in (0, 22, 44, 65):
            im = Image.fromarray(d["frames"][t]).resize((192, 192))
            ImageDraw.Draw(im).text((4, 4), f"{d['wall_dist'][t]:.2f} m", fill=(255, 255, 0))
            tiles.append(np.asarray(im))
        rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(os.path.join(CHK, "sheet_b1.png"))
    import imageio.v2 as imageio
    wr = imageio.get_writer(os.path.join(CHK, "video_b1.mp4"), fps=20)
    for f in fs[::3]:
        d = np.load(f, allow_pickle=True)
        for t in range(len(d["frames"])):
            im = Image.fromarray(d["frames"][t]).resize((384, 384))
            ImageDraw.Draw(im).text((6, 6), f"{os.path.basename(f)[7:-4]}  wall {d['wall_dist'][t]:.2f} m", fill=(255, 255, 0))
            wr.append_data(np.asarray(im))
    wr.close()
    print("->", CHK)


def do_probe(a):
    import torch
    from sklearn.linear_model import RidgeCV
    from vjepa2_encoder import VJEPA2FrameEncoder
    from wm.evaluate import encode_clip
    from wm.config import from_checkpoint
    from wm.models.ftm import ForwardTransitionModel
    from wm.models.itm import InverseTransitionModel
    from wm.data.strided import stride_of
    fs = sorted(glob.glob(os.path.join(OUT, "*.npz")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    E, Y, G = [], [], []
    for gi, f in enumerate(fs):
        d = np.load(f, allow_pickle=True)
        e = encode_clip(enc, d["frames"], 4).float().cpu()          # (T', tokens, D) or (T', D)
        E.append(e); Y.append(np.asarray(d["wall_dist"], float)); G.append(gi)
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    k = stride_of(cfg)
    itm = InverseTransitionModel(cfg).to(dev).eval(); itm.load_state_dict(ck["itm"], strict=False)
    ftm = ForwardTransitionModel(cfg).to(dev).eval(); ftm.load_state_dict(ck["ftm"], strict=False)

    def pool(x):
        return x.flatten(1) if x.dim() == 2 else x.mean(1)
    n_fr = [len(y) for y in Y]
    T = [len(e) for e in E]
    print(f"{len(fs)} clips, embeddings per clip {T[0]} (frames {n_fr[0]}), stride {k}")
    # frame index of embedding i: encode_clip maps tubelets of 2 frames -> embedding i covers frames 2i, 2i+1 (checked below)
    fpe = n_fr[0] / T[0]
    lab = [np.array([y[min(int(round(i * fpe)), len(y) - 1)] for i in range(len(e))]) for e, y in zip(E, Y)]
    folds = np.arange(len(fs)) % 4
    res = {}
    def rolled(i, h, mode):
        """embeddings standing for frame t + h*k, from frame t: real (the true future), copy (frame t itself,
        persistence), ftm (FTM rolled h steps with the real transition's ITM z)"""
        e = E[i].to(dev)
        n = len(e) - h * k
        if n <= 0:
            return None, None
        if mode == "real":
            x = e[h * k:h * k + n]
        elif mode == "copy":
            x = e[:n]
        else:
            x = e[:n]
            with torch.no_grad():
                for step in range(h):
                    t0 = np.arange(n) + step * k
                    x = ftm(x, itm(e[t0], e[t0 + k]))
        return pool(x.float().cpu()).numpy(), lab[i][h * k:h * k + n]

    for h in (0, 1, 2, 4, 8):
        for mode, fit_on in (("copy", "real"), ("ftm", "real"), ("ftm", "ftm")):
            if h == 0 and mode != "copy":
                continue
            pred, true = [], []
            for fo in range(4):
                tr = [i for i in range(len(fs)) if folds[i] != fo]
                te = [i for i in range(len(fs)) if folds[i] == fo]
                X, Yl = zip(*[rolled(i, h if fit_on == "ftm" else 0, fit_on) for i in tr])
                X = [x for x in X if x is not None]; Yl = [y for y in Yl if y is not None]
                reg = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(np.concatenate(X), np.concatenate(Yl))
                for i in te:
                    x, y = rolled(i, h, mode)
                    if x is not None:
                        pred.append(reg.predict(x)); true.append(y)
            p, t = np.concatenate(pred), np.concatenate(true)
            r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            mae = np.abs(p - t).mean()
            near = t < 2.0
            tag = f"h={h} ({h * k * 0.05:.2f} s) {mode:4s} probe fit on {fit_on:4s}"
            res[tag] = (r2, mae, np.abs(p - t)[near].mean() if near.any() else np.nan, len(t))
            print(f"{tag}: R2 {r2:+.3f}  MAE {mae:.2f} m  MAE within 2 m {res[tag][2]:.2f} m  (n={len(t)})", flush=True)
    os.makedirs(CHK, exist_ok=True)
    with open(os.path.join(CHK, "probe_b1.txt"), "w") as fh:
        fh.write(f"ckpt {a.ckpt}; ridge on mean-pooled frozen V-JEPA2 embeddings -> camera-to-front-wall distance; "
                 f"4 folds grouped by clip; horizon h = FTM rolled h steps with the real transition's ITM z\n")
        fh.write("copy = frame t's own embedding used for t+h (persistence); ftm = FTM rolled h steps with the real ITM z; "
                 "'fit on ftm' = probe refit on FTM-predicted embeddings of the training folds\n")
        for tag, v in res.items():
            fh.write(f"{tag}: R2 {v[0]:+.3f} MAE {v[1]:.2f} m, MAE within 2 m {v[2]:.2f} m, n {v[3]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("render", "sheet", "probe"))
    ap.add_argument("--port", type=int, default=25704)
    ap.add_argument("--ckpt", default="results/eval/round1_branches_s0/ckpt/b1.pt")
    a = ap.parse_args()
    {"render": do_render, "sheet": do_sheet, "probe": do_probe}[a.cmd](a)
