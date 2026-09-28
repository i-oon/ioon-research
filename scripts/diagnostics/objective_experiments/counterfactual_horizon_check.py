"""How much does the ACTION decide the real future, as a function of how far ahead we look?

F258 found a real one-step counterfactual frame barely distinguishable by action (read as the action's
Froude at r 0.14-0.19): over 50 ms the state dominates, so a 1-step FTM is right to ignore z. This
renders real counterfactual TRAJECTORIES in CoppeliaSim -- from library state s at step t, candidate
a's own motion applied for K consecutive steps (planar delta + heading composed onto s, height/pitch/
roll and joints from a; the kinematic loop's execution) -- and measures, for k in --ks, how separable
the outcomes e*_k(s, a) are by action. No FTM anywhere; nothing here is a model's prediction.

  act/state     embedding-space spread of outcomes across actions (fixed s) over spread across states
                (fixed a): how much the action, rather than where you started, decides the outcome
  probe R2      model-free: ridge from the pooled real change e*_k - e_s to the action's true Froude
                over [t, t+k), leave-one-state-out -- can the change be read at all
  itm r         `body(ITM)` averaged over the k real one-step pairs along the trajectory, correlated
                across actions with the truth (the w=k rollout read-out, on real frames)

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_horizon_check.py
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/render", "sim/scene", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from counterfactual_truth_check import build_scene  # noqa: E402
from close_loop_direct_froude import load_motion, qz, quat_mul  # noqa: E402
from ego_camera import check_ego_view  # noqa: E402
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import REGISTRY, heading, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import load_candidates  # noqa: E402

CH = ("forward", "lateral", "yaw")


def pool(x, q=2):
    n = x.shape[0]
    x = x.reshape(n, 16, 16, -1)
    s = 16 // q
    return torch.cat([x[:, i * s:(i + 1) * s, j * s:(j + 1) * s].mean((1, 2)) for i in range(q) for j in range(q)], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh24_ft_ctrl_fz/b1_lora_c3/ckpt_lib_insample.pt",
                    help="only its ITM/body head, for the `itm r` column")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--K", type=int, default=11)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 11])
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--cache", default="results/wm/cache/counterfactual_horizon.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    mot = [load_motion(c["path"]) for c in cands]
    bm = [np.asarray(load(c["path"], REGISTRY["b1"])["body_motion"])[:, :3] for c in cands]
    n = len(cands)
    S = [int(v) for v in np.linspace(0, n - 1, args.states).round()]

    cache = os.path.join(ROOT, args.cache)
    if os.path.exists(cache):
        E = torch.load(cache)
        print(f"trajectories from cache: {cache}")
    else:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        from vjepa2_encoder import VJEPA2FrameEncoder
        sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
        pose = build_scene(sim, args.scene)
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
        E = {}
        for t in args.steps:
            for s in S:
                ms = mot[s]
                c0, s0 = np.cos(-ms["psi"][0]), np.sin(-ms["psi"][0])
                d0 = ms["pos"][t, :2] - ms["pos"][0, :2]
                p_s = np.array([c0 * d0[0] - s0 * d0[1], s0 * d0[0] + c0 * d0[1], ms["pos"][t, 2]])
                q_s = quat_mul(qz(-float(ms["psi"][0])), ms["quat"][t])
                psi_s = float(heading(q_s[None], "b1")[0])
                img_s = pose(p_s, q_s, ms["jpos"][t])
                if not E:
                    print(f"view check: corr {check_ego_view(img_s, ref, min_corr=0.9):.3f}", flush=True)
                traj = []
                for a in range(n):
                    ma, pos, psi, frames = mot[a], p_s.copy(), psi_s, []
                    for j in range(t, t + args.K):
                        d = ma["pos"][j + 1, :2] - ma["pos"][j, :2]
                        rot = psi - float(ma["psi"][j])
                        pos = np.array([pos[0] + np.cos(rot) * d[0] - np.sin(rot) * d[1],
                                        pos[1] + np.sin(rot) * d[0] + np.cos(rot) * d[1], ma["pos"][j + 1, 2]])
                        dpsi = float(ma["psi"][j + 1] - ma["psi"][j])
                        psi += float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
                        quat = quat_mul(qz(psi - float(ma["psi"][j + 1])), ma["quat"][j + 1])
                        frames.append(pose(pos, quat / np.linalg.norm(quat), ma["jpos"][j + 1]))
                    traj.append(frames)
                allf = np.stack([img_s] + [f for fr in traj for f in fr])
                emb = encode_clip(enc, allf, 2).float().cpu().half()
                E[(t, s)] = {"e_s": emb[:1], "traj": emb[1:].reshape(n, args.K, *emb.shape[1:])}
                print(f"  rendered t={t} s={s}", flush=True)
        torch.save(E, cache)
        del enc
        torch.cuda.empty_cache()

    m = Models(os.path.join(ROOT, args.ckpt), "b1", cands[0]["actions"].shape[1], dev)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
    keys = sorted(E)
    print(f"\n{len(keys)} state-steps x {n} actions; k = steps of real counterfactual motion\n")
    print(f"{'k':>3} {'act/state':>10} " + " ".join(f"{'probe ' + c:>13}" for c in CH)
          + " " + " ".join(f"{'itm r ' + c:>13}" for c in CH))
    for k in args.ks:
        # 1. spread by action vs spread by start state, raw embedding space
        O = {key: (E[key]["traj"][:, k - 1].float().flatten(1)) for key in keys}   # (n, D)
        act = np.mean([((o - o.mean(0)) ** 2).sum(1).mean().item() for o in O.values()])
        state = []
        for t in args.steps:
            grid = torch.stack([O[(t, s)] for s in S])                               # (states, n, D)
            state.append(((grid - grid.mean(0)) ** 2).sum(-1).mean().item())
        ratio = act / np.mean(state)
        # 2. model-free probe on the real change, leave-one-state-out
        X, Y, G = [], [], []
        for (t, s) in keys:
            X.append(pool(E[(t, s)]["traj"][:, k - 1].float() - E[(t, s)]["e_s"].float()).numpy())
            Y.append(np.stack([b[t:t + k].mean(0) for b in bm]))
            G += [s] * n
        X, Y, G = np.concatenate(X), np.concatenate(Y), np.asarray(G)
        pred = np.zeros_like(Y)
        for g in np.unique(G):
            tr, te = G != g, G == g
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            pred[te] = RidgeCV(alphas=np.logspace(0, 6, 13)).fit((X[tr] - mu) / sd, Y[tr]).predict((X[te] - mu) / sd)
        probe = 1 - ((pred - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        # 3. the w=k read-out on REAL frames
        rs = []
        with torch.no_grad():
            for (t, s) in keys:
                tr_ = E[(t, s)]["traj"].float()
                seq = torch.cat([E[(t, s)]["e_s"].float().expand(n, -1, -1)[:, None], tr_[:, :k]], 1).to(dev)
                if m.offset is not None:
                    seq = seq - m.offset.float().reshape(seq.shape[2:]).to(dev)
                zs = torch.stack([m.itm(seq[:, j], seq[:, j + 1]) for j in range(k)], 1)
                f = m.md.body(None, zs.reshape(n * k, -1)).reshape(n, k, -1).mean(1).cpu().numpy() * std + mean
                truth = np.stack([b[t:t + k].mean(0) for b in bm])
                rs.append([corr(f[:, j], truth[:, j]) for j in range(3)])
        itm_r = np.nanmean(np.asarray(rs), 0)
        print(f"{k:>3} {ratio:>10.3f} " + " ".join(f"{v:>13.3f}" for v in probe) + " "
              + " ".join(f"{v:>13.3f}" for v in itm_r), flush=True)


if __name__ == "__main__":
    main()
