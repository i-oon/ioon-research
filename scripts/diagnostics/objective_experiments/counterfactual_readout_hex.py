"""Hexapod version of the kinematic counterfactual read-out test (B1: counterfactual_horizon_check.py,
counterfactual_prediction_k.py, physics_counterfactual_check.py).

From library start state (clip s, step t), each of the 24 library candidates' own recorded motion is
applied KINEMATICALLY for K = stride (5) steps -- per step the candidate's planar head displacement and
heading change are composed onto the running pose, while height, tilt (the candidate's own abdomen
quaternion re-yawed) and the 18 joint angles come from the candidate -- and the end frame is rendered in
CoppeliaSim with the collector's exact egocentric scene (collect_ik.drive_and_record's ego block:
room_for / scale_floor / randomise_ground(seed 0) / attach_ego on /head with WALK_PITCH / respawn at the
origin / build_texture_box / 90 deg). The simulation is never started: poses are set with the sim
stopped (abdomen pose + setJointPosition), exactly as the B1 build_scene does.

Hexapod clips store only `head` (world position of /head), `body_quat` (abdomen, x y z w) and `actions`
(18 joint targets). The abdomen position is recovered as head - R(body_quat) @ h_off, where h_off is
/head's position in the /abdomen frame at the scene's rest configuration (body joints B1..B4 at 0); the
joint angles are the recorded targets. A replay of real clip poses is checked against the real frames
(pixel and row-profile correlation) before anything is scored.

Per model, Pearson r across the 24 actions vs the actions' true Froude over [t, t+5) (fwd / lat / yaw),
mean over start states:
  own         Froude head(ITM(e_a[t], e_a[t+5]))                 candidate's own recorded transition
  real cf     Froude head(ITM(e_s, e*_5(s, a)))                  real (rendered) counterfactual future
  FTM cf      Froude head(ITM(e_s, FTM(e_s, proj(chunk_a))))     predicted future
  direct      Froude head(proj(chunk_a))                        chunk = a's 5 commands at t + lag

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_readout_hex.py --device cpu \\
        --ckpt s0=wm/runs/fmd_beh24_s0/c08_zeroshot/ckpt_lib_zeroshot.pt \\
        --ckpt s1=wm/runs/fmd_beh24_s1/c08_zeroshot/ckpt_lib_zeroshot.pt
Needs CoppeliaSim on --port.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/collect", "sim/scene", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from collect_ik import LEGS, SEG, capture, settle  # noqa: E402
from ego_camera import (EGO_FOV_DEG, WALK_PITCH, attach_ego, build_texture_box, check_ego_view,  # noqa: E402
                        insect_forward, randomise_ground, room_for, scale_floor, set_ego_fov)
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, heading, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at, load_candidates  # noqa: E402

CH = ("forward", "lateral", "yaw")


def qmul(a, b):
    """Hamilton product, quaternions as (x, y, z, w)."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def qz(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def qrot(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def build_hex_scene(sim, scene, ego_seed=0):
    """The collector's egocentric scene, without starting the simulation. Returns pose(head, quat, joints)."""
    sim.loadScene(os.path.join(ROOT, "sim", "env", scene))
    settle(sim)
    cam = sim.getObject("/vjepa_cam")
    track, body = sim.getObject("/head"), sim.getObject("/abdomen")
    joints = [sim.getObject(f"/{jn}_{leg}") for leg in LEGS for jn in SEG]
    bjoints = [sim.getObject(f"/B{i}_joint") for i in range(1, 5)]
    R = room_for(sim.getObjectPosition(track, sim.handle_world)[2])
    scale_floor(sim, R["size"])
    randomise_ground(sim, seed=ego_seed, uv=R["ground_uv"])
    attach_ego(sim, cam, track, insect_forward(sim), (0, 0, 0), offset_frac=R["offset_frac"],
               pitch_comp=WALK_PITCH["hexapod"])
    pos = sim.getObjectPosition(body, sim.handle_world)
    head = sim.getObjectPosition(track, sim.handle_world)
    sim.setObjectPosition(body, sim.handle_world, [pos[0] - head[0], pos[1] - head[1], pos[2]])
    here = np.array(sim.getObjectPosition(track, sim.handle_world))
    build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=ego_seed,
                      centre=(float(here[0]), float(here[1])))
    set_ego_fov(sim, cam, EGO_FOV_DEG)
    for h in bjoints:
        sim.setJointPosition(h, 0.0)
    h_off = np.array(sim.getObjectPosition(track, body))           # /head in the /abdomen frame

    def pose(head_pos, quat, jangles):
        q = np.asarray(quat, float)
        q = q / np.linalg.norm(q)
        abd = np.asarray(head_pos, float) - qrot(q, h_off)
        sim.setObjectQuaternion(body, sim.handle_world, [float(v) for v in q])
        sim.setObjectPosition(body, sim.handle_world, [float(v) for v in abd])
        for h, a in zip(joints, jangles):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)
    return pose


def counterfactual_pose(head_s, q_s, clip_a, t, K):
    """Candidate a's steps t..t+K-1 composed onto start pose (head_s, q_s); returns pose at t+K."""
    ha, qa, psia = clip_a["head"], clip_a["quat"], clip_a["psi"]
    pos, psi = np.array(head_s, float), float(heading(np.asarray(q_s)[None], "hexapod")[0])
    quat = np.asarray(q_s, float)
    for j in range(t, t + K):
        d = ha[j + 1, :2] - ha[j, :2]
        rot = psi - psia[j]
        pos = np.array([pos[0] + np.cos(rot) * d[0] - np.sin(rot) * d[1],
                        pos[1] + np.sin(rot) * d[0] + np.cos(rot) * d[1], ha[j + 1, 2]])
        dpsi = psia[j + 1] - psia[j]
        psi += float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
        quat = qmul(qz(psi - psia[j + 1]), qa[j + 1])
    return pos, quat / np.linalg.norm(quat), clip_a["act"][t + K]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_c10f10t10_ego_flat_cleantrain")
    ap.add_argument("--scene", default="medauroidea_c10f10t10.ttt")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--cache", default="results/wm/cache/counterfactual_readout_hex_c10.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev, K = args.device, args.K

    spec = REGISTRY["hexapod"]
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "hexapod", per_condition=999)
    n = len(cands)
    clips = []
    for c in cands:
        d = np.load(c["path"], allow_pickle=True)
        q = d["body_quat"].astype(np.float64)
        clips.append({"head": d["head"].astype(np.float64), "quat": q,
                      "psi": np.unwrap(heading(q, "hexapod")), "act": d["actions"].astype(np.float64)})
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
            # the collector draws the room (ground + wall textures) with ego_seed = the clip's repeat index,
            # so every start state is rendered in its own clip's room
            ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
            seeds = {s: int(np.load(cands[s]["path"], allow_pickle=True)["repeat"]) for s in S}
            pix, prof, prof_lib, rendered = [], [], [], {}
            for seed in sorted(set(seeds.values())):
                pose = build_hex_scene(sim, args.scene, ego_seed=seed)
                for s in [s for s in S if seeds[s] == seed]:
                    # 1. view check: replay REAL clip poses and compare with the real frames
                    fr = np.load(cands[s]["path"], allow_pickle=True)["frames"]
                    for t in [0] + args.steps + [t_ + K for t_ in args.steps]:
                        img = pose(clips[s]["head"][t], clips[s]["quat"][t], clips[s]["act"][t])
                        pix.append(np.corrcoef(img.ravel().astype(float), fr[t].ravel().astype(float))[0, 1])
                        prof.append(check_ego_view(img, [fr[t]], min_corr=0.0))
                        prof_lib.append(check_ego_view(img, ref, min_corr=0.0))
                    for t in args.steps:
                        fs = [pose(clips[s]["head"][t], clips[s]["quat"][t], clips[s]["act"][t])]
                        for a in range(n):
                            fs.append(pose(*counterfactual_pose(clips[s]["head"][t], clips[s]["quat"][t],
                                                                clips[a], t, K)))
                        rendered[(t, s)] = np.stack(fs)
                    print(f"  rendered s={s} (seed {seed})", flush=True)
            print(f"view check (replayed real poses vs the same clip's real frames): pixel corr mean "
                  f"{np.mean(pix):.3f} min {np.min(pix):.3f}; row-profile corr mean {np.mean(prof):.3f} "
                  f"min {np.min(prof):.3f}; row-profile vs library frame 0 mean {np.mean(prof_lib):.3f} "
                  f"min {np.min(prof_lib):.3f}", flush=True)
            if np.min(prof) < 0.9:
                raise SystemExit("view check failed (< 0.9)")
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
        for t in args.steps:          # own recorded transition, real clip frames at t and t+K
            real = np.stack([np.load(c["path"], allow_pickle=True)["frames"][[t, t + K]] for c in cands])
            e = encode_clip(enc, real.reshape(-1, *real.shape[2:]), 8).float().half()
            E["own"][t] = e.reshape(n, 2, *e.shape[1:])
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(E, cache)
        del enc

    for spec_ in args.ckpt:
        name, path = spec_.split("=", 1)
        m = Models(os.path.join(ROOT, path), "hexapod", 18, dev)
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
            z = m.proj(torch.as_tensor(chunk, device=dev), "hexapod")
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
