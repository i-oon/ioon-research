"""B1 version of counterfactual_readout_hex.py, in the v3 MATCHED rendering (data/egocentric_v3).

Same protocol as the hexapod script: start states = 8 library clips x t in --steps; from each, every
candidate's own recorded motion is applied KINEMATICALLY for K = stride (5) steps (planar base
displacement + heading change composed onto the running pose; height, tilt (candidate's quaternion
re-yawed) and 12 joint angles from the candidate) and the end frame rendered in CoppeliaSim.

The scene reproduces `render_b1_replay.py --ego --match_floor --ground_uv_mult 1.0 --ego_seed i` exactly
(what scripts/dataset/rerender_b1_ego_matched.py ran): scene's vjepa_cam, room_for / build_texture_box
(seed, centred on the spawn 0,0) / attach_ego on base_visual along the clip's initial heading with
WALK_PITCH["b1"] / scale_floor(room size) / randomise_ground(seed, ground_uv x1.0) / 90 deg. Poses are the
clip's base_pos shifted so step 0 is at (0, 0) (spawn), no yaw alignment. The room seed is read from the
clip's `render_note` (seed = the clip's index in its source directory), and every start state is
rendered in its own clip's room. A replay of real clip poses is checked against the real v3 frames first.

Read-outs (Pearson r across the 24 actions vs true Froude over [t, t+5), mean over start states):
  own / real cf / FTM cf / direct -- identical definitions to the hexapod script.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_readout_b1.py --device cpu \\
        --ckpt s0=wm/runs/fmd_beh24_s0/b1_v3_new/ckpt.pt --ckpt s1=wm/runs/fmd_beh24_s1/b1_v3_new/ckpt.pt
Needs CoppeliaSim on --port.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/render", "sim/scene", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from close_loop_direct_froude import qz, quat_mul  # noqa: E402  (w, x, y, z)
from ego_camera import (WALK_PITCH, attach_ego, build_texture_box, check_ego_view,  # noqa: E402
                        randomise_ground, room_for, scale_floor)
from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle  # noqa: E402
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, heading, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at, load_candidates  # noqa: E402


def build_b1v3_scene(sim, scene, seed, fwd_psi, spawn=(0.0, 0.0)):
    """render_b1_replay.py --ego --match_floor --ground_uv_mult 1.0, step for step. Returns pose(pos, quat_wxyz, j)."""
    settle(sim)
    sim.loadScene(os.path.abspath(os.path.join(ROOT, scene)))
    settle(sim)
    jm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    joints, root = [jm[a] for a in JOINT_ALIASES_SDK], sm[ROOT_ALIAS]
    cam = sim.getObject("/" + SENSOR)
    R = room_for(sim.getObjectPosition(root, sim.handle_world)[2])
    build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed,
                      centre=(float(spawn[0]), float(spawn[1])))
    attach_ego(sim, cam, root, [float(np.cos(fwd_psi)), float(np.sin(fwd_psi)), 0.0], (0, 0, 0),
               offset_frac=R["offset_frac"], pitch_comp=WALK_PITCH["b1"])
    scale_floor(sim, R["size"])
    randomise_ground(sim, seed=seed, uv=R["ground_uv"] * 1.0)
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(90.0)))

    def pose(pos, q, jangles):
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in pos])
        sim.setObjectQuaternion(root, sim.handle_world, [float(q[1]), float(q[2]), float(q[3]), float(q[0])])
        for h, a in zip(joints, jangles):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)
    return pose


def counterfactual_pose(pos_s, q_s, clip_a, t, K):
    """Candidate a's steps t..t+K-1 composed onto start pose (pos_s, q_s wxyz); returns pose at t+K."""
    pa, qa, psia = clip_a["pos"], clip_a["quat"], clip_a["psi"]
    pos, psi = np.array(pos_s, float), float(heading(np.asarray(q_s)[None], "b1")[0])
    quat = np.asarray(q_s, float)
    for j in range(t, t + K):
        d = pa[j + 1, :2] - pa[j, :2]
        rot = psi - psia[j]
        pos = np.array([pos[0] + np.cos(rot) * d[0] - np.sin(rot) * d[1],
                        pos[1] + np.sin(rot) * d[0] + np.cos(rot) * d[1], pa[j + 1, 2]])
        dpsi = psia[j + 1] - psia[j]
        psi += float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
        quat = quat_mul(qz(psi - psia[j + 1]), qa[j + 1])
    return pos, quat / np.linalg.norm(quat), clip_a["jpos"][t + K]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--candidates_dir", default="data/egocentric_v3/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--cache", default="results/wm/cache/counterfactual_readout_b1v3.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev, K = args.device, args.K

    spec = REGISTRY["b1"]
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    clips = []
    for c in cands:
        d = np.load(c["path"], allow_pickle=True)
        q = d["base_quat"].astype(np.float64)
        pos = d["base_pos"].astype(np.float64).copy()
        pos[:, :2] -= pos[0, :2]                      # --spawn 0 0
        seed = int(re.search(r"seed (\d+)", str(d["render_note"])).group(1))
        clips.append({"pos": pos, "quat": q, "psi": np.unwrap(heading(q, "b1")),
                      "jpos": d["joint_pos"].astype(np.float64), "seed": seed})
    bm = [np.asarray(load(c["path"], spec)["body_motion"])[:, :3] for c in cands]
    S = [int(v) for v in np.linspace(0, n - 1, args.states).round()]

    cache = os.path.join(ROOT, args.cache)
    if os.path.exists(cache):
        E = torch.load(cache)
        print(f"embeddings from cache: {cache}")
    else:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        from vjepa2_encoder import VJEPA2FrameEncoder
        sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
        try:
            ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
            pix, prof, prof_lib, rendered = [], [], [], {}
            for s in S:
                cs = clips[s]
                pose = build_b1v3_scene(sim, args.scene, cs["seed"], float(heading(cs["quat"][:1], "b1")[0]))
                fr = np.load(cands[s]["path"], allow_pickle=True)["frames"]
                for t in [0] + args.steps + [t_ + K for t_ in args.steps]:
                    img = pose(cs["pos"][t], cs["quat"][t], cs["jpos"][t])
                    pix.append(np.corrcoef(img.ravel().astype(float), fr[t].ravel().astype(float))[0, 1])
                    prof.append(check_ego_view(img, [fr[t]], min_corr=0.0))
                    prof_lib.append(check_ego_view(img, ref, min_corr=0.0))
                for t in args.steps:
                    fs = [pose(cs["pos"][t], cs["quat"][t], cs["jpos"][t])]
                    for a in range(n):
                        fs.append(pose(*counterfactual_pose(cs["pos"][t], cs["quat"][t], clips[a], t, K)))
                    rendered[(t, s)] = np.stack(fs)
                print(f"  rendered s={s} (seed {cs['seed']}); pixel corr so far min {np.min(pix):.3f}", flush=True)
            print(f"view check (replayed real poses vs the same clip's real v3 frames): pixel corr mean "
                  f"{np.mean(pix):.3f} min {np.min(pix):.3f}; row-profile corr mean {np.mean(prof):.3f} "
                  f"min {np.min(prof):.3f}; row-profile vs library frame 0 mean {np.mean(prof_lib):.3f} "
                  f"min {np.min(prof_lib):.3f}", flush=True)
            if np.min(prof) < 0.9 or np.mean(pix) < 0.95:
                raise SystemExit("view check failed")
            keys = sorted(rendered)
            frames = [f for k in keys for f in rendered[k]]
        finally:
            sim.stopSimulation()
        enc = VJEPA2FrameEncoder(device="cpu", dtype=torch.float32)
        emb = encode_clip(enc, np.stack(frames), 8).float().half()
        E = {"cf": {}, "own": {}}
        for i, key in enumerate(keys):
            blk = emb[i * (n + 1):(i + 1) * (n + 1)]
            E["cf"][key] = {"e_s": blk[:1], "e_f": blk[1:]}
        for t in args.steps:
            real = np.stack([np.load(c["path"], allow_pickle=True)["frames"][[t, t + K]] for c in cands])
            e = encode_clip(enc, real.reshape(-1, *real.shape[2:]), 8).float().half()
            E["own"][t] = e.reshape(n, 2, *e.shape[1:])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(E, cache)
        del enc

    for spec_ in args.ckpt:
        name, path = spec_.split("=", 1)
        m = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        if m.stride != K:
            raise SystemExit(f"{name}: stride {m.stride} != K {K}")
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        rd = lambda z: m.md.body(None, z).cpu().numpy() * std + mean  # noqa: E731
        off = None if m.offset is None else m.offset.float().to(dev)
        fix = (lambda e: e) if off is None else (lambda e: e - off.reshape(e.shape[1:]))  # noqa: E731
        R = {k: [] for k in ("own", "real cf", "FTM cf", "direct")}
        for (t, s), rec in sorted(E["cf"].items()):
            truth = np.stack([b[t:t + K].mean(0) for b in bm])
            e_s = fix(rec["e_s"].float().to(dev)).expand(n, -1, -1)
            e_f = fix(rec["e_f"].float().to(dev))
            own = E["own"][t].float().to(dev)
            chunk = np.stack([action_chunk_at(c["actions"], t + m.action_lag, K) for c in cands])
            z = m.proj(torch.as_tensor(chunk, device=dev), "b1")
            reads = {"own": rd(m.itm(fix(own[:, 0]), fix(own[:, 1]))),
                     "real cf": rd(m.itm(e_s, e_f)),
                     "FTM cf": rd(m.itm(e_s, m.ftm_step(e_s, z))),
                     "direct": rd(z)}
            for k, f in reads.items():
                R[k].append([corr(f[:, j], truth[:, j]) for j in range(3)])
        print(f"\n=== {name}: {len(E['cf'])} start states x {n} actions, K={K}; r fwd / lat / yaw")
        for k, v in R.items():
            print(f"  {k:<8}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)), flush=True)


if __name__ == "__main__":
    main()
