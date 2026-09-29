"""Same start state, different actions: do the model's predicted futures differ the way the real ones do?

Renders ONE B1 start state and, for every one of the 24 library candidates, its own motion applied
from that state for 5 steps (the counterfactual construction of counterfactual_horizon_check.py,
same scene and room for all actions), and encodes the frames. The FTM predicts embeddings, not
pixels, so each model's prediction for action a is shown as the REAL future it is closest to: the
action-specific component (mean over the 24 actions subtracted) of the prediction vs of each real
future, by cosine similarity -- the "top-1 retrieval" criterion. A model that ignores the action maps
every action to the same few real futures.

    .venv/bin/python3 scripts/figures/action_future_strip.py \\
        --model "start of period=wm/runs/beh24_hinge_cleansplit/b1_adapt_beh24_lora_c3/ckpt_lib_s4.pt" \\
        --model "current=wm/runs/fmd_beh24_s0/b1_lora_c3/ckpt_lib_s4.pt"
Stride-1 models are rolled 5 FTM steps with the per-step commands; stride-5 models take one step.
Needs CoppeliaSim on port 23000.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "sim/control", "sim/scene", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from counterfactual_truth_check import build_scene  # noqa: E402
from close_loop_direct_froude import load_motion, qz, quat_mul  # noqa: E402
from ego_camera import check_ego_view  # noqa: E402
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.embodiment import heading  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at, condition_of, load_candidates  # noqa: E402

K = 5


def render(cands, s, t, port, scene):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    sim = RemoteAPIClient("localhost", port=port).getObject("sim")
    pose = build_scene(sim, scene)
    mot = [load_motion(c["path"]) for c in cands]
    ms = mot[s]
    c0, s0 = np.cos(-ms["psi"][0]), np.sin(-ms["psi"][0])
    d0 = ms["pos"][t, :2] - ms["pos"][0, :2]
    p_s = np.array([c0 * d0[0] - s0 * d0[1], s0 * d0[0] + c0 * d0[1], ms["pos"][t, 2]])
    q_s = quat_mul(qz(-float(ms["psi"][0])), ms["quat"][t])
    psi_s = float(heading(q_s[None], "b1")[0])
    img_s = pose(p_s, q_s, ms["jpos"][t])
    ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
    print(f"view check: corr {check_ego_view(img_s, ref, min_corr=0.9):.3f}", flush=True)
    futures = []
    for a in range(len(cands)):
        ma, pos, psi = mot[a], p_s.copy(), psi_s
        for j in range(t, t + K):
            d = ma["pos"][j + 1, :2] - ma["pos"][j, :2]
            rot = psi - float(ma["psi"][j])
            pos = np.array([pos[0] + np.cos(rot) * d[0] - np.sin(rot) * d[1],
                            pos[1] + np.sin(rot) * d[0] + np.cos(rot) * d[1], ma["pos"][j + 1, 2]])
            dpsi = float(ma["psi"][j + 1] - ma["psi"][j])
            psi += float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
            quat = quat_mul(qz(psi - float(ma["psi"][j + 1])), ma["quat"][j + 1])
            if j == t + K - 1:
                futures.append(pose(pos, quat / np.linalg.norm(quat), ma["jpos"][j + 1]))
    return img_s, np.stack(futures)


@torch.no_grad()
def predict(m, cands, e_s, t):
    n = len(cands)
    e = e_s.expand(n, -1, -1)
    if m.stride >= K:
        a = np.stack([action_chunk_at(c["actions"], t + m.action_lag, m.stride) for c in cands])
        return m.ftm_step(e, m.proj(torch.as_tensor(a, device=e.device), "b1"))
    for j in range(K):
        a = np.stack([c["actions"][t + m.action_lag + j] for c in cands])
        e = m.ftm_step(e, m.proj(torch.as_tensor(a, device=e.device), "b1"))
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=CKPT")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--state", default="0,30", help="s,t: library clip index and frame of the start state")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--out", default="results/deck/weekly/action_future_strip.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = "cuda"
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    s, t = (int(x) for x in args.state.split(","))
    img_s, fut = render(cands, s, t, args.port, args.scene)
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    emb = encode_clip(enc, np.concatenate([img_s[None], fut]), 2).float()
    del enc
    torch.cuda.empty_cache()
    e_s, real = emb[:1].to(dev), emb[1:].to(dev)
    spec = lambda x: F.normalize((x - x.mean(0, keepdim=True)).flatten(1), dim=1)
    rows = []
    for lab, path in (m.split("=", 1) for m in args.model):
        mdl = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        pred = predict(mdl, cands, e_s, t).float()
        match = (spec(pred) @ spec(real).T).argmax(1).cpu().numpy()
        top1 = float((match == np.arange(n)).mean())
        rows.append((lab, match, top1, len(set(match.tolist()))))
        print(f"{lab}: top-1 {top1:.3f} (chance {1 / n:.3f}), distinct real futures matched "
              f"{len(set(match.tolist()))} / {n}", flush=True)
        del mdl
        torch.cuda.empty_cache()

    names = [condition_of(c["path"]) for c in cands]
    pick = [names.index(c) for c in ("speed_vx0.50", "turn_w0.075", "side_L_lvl1", "side_R_lvl1",
                                    "speed_vx0.30", "turn_w0.024") if c in names][:6]
    R = 2 + len(rows)
    fig, ax = plt.subplots(R, len(pick), figsize=(2.1 * len(pick), 2.25 * R))
    for col, i in enumerate(pick):
        ax[0, col].imshow(img_s)
        ax[0, col].set_title(names[i], fontsize=9)
        ax[1, col].imshow(fut[i])
        for r, (lab, match, top1, nd) in enumerate(rows, start=2):
            ax[r, col].imshow(fut[match[i]])
            ok = match[i] == i
            for sp in ax[r, col].spines.values():
                sp.set_edgecolor("#2e7d32" if ok else "#c62828"); sp.set_linewidth(3)
            ax[r, col].set_xlabel(("= " if ok else "looks like ") + names[match[i]], fontsize=7)
        for r in range(R):
            ax[r, col].set_xticks([]); ax[r, col].set_yticks([])
    ax[0, 0].set_ylabel("start frame\n(same for all)", fontsize=9)
    ax[1, 0].set_ylabel("real future\nafter 5 steps", fontsize=9)
    for r, (lab, match, top1, nd) in enumerate(rows, start=2):
        ax[r, 0].set_ylabel(f"{lab}:\npredicted future\nlooks most like", fontsize=9)
    fig.suptitle("Same start state, different actions (B1). Green border = the prediction matches this "
                 "action's real future; red = it matches another action's.\n" +
                 "   ".join(f"{lab}: correct {top1 * n:.0f}/{n} actions, {nd} distinct futures"
                            for lab, _, top1, nd in rows), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, args.out), dpi=130)
    print("->", args.out)


if __name__ == "__main__":
    main()
