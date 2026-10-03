"""Can the ITM + Froude head read a counterfactual future when it comes from PHYSICS, not kinematic posing?

The kinematic counterfactuals (counterfactual_horizon_check.py) pose each candidate's recorded motion from
a shared start state, so at the first step the body jumps to that candidate's recorded posture and
height -- a transition no training clip contains. Reading those real futures gave Pearson r
0.26 / 0.41 / 0.52 (current pipeline, B1), against 0.83 / 0.90 / 0.87 for direct. Here the branches
are physical: from one exact MuJoCo state (mjSTATE_INTEGRATION + the policy's last action and gait
clock, as branch_b1_mujoco.py), the B1's own policy executes each of the 24 library candidates'
recorded commands for 5 rendered frames. The truth is the Froude the simulated body actually achieved.

  per start state, across the 24 branches, Pearson r vs achieved Froude (fwd / lat / yaw) of:
    direct      Froude head(proj(executed action chunk))
    read real   Froude head(ITM(e_start, e_future))           the physical future = a perfect FTM
    read pred   Froude head(ITM(e_start, FTM(e_start, proj(chunk))))   what rollout reads
If "read real" rises well above the kinematic 0.26 / 0.41 / 0.52, the kinematic posing was the cause;
if it stays low, the read-out itself cannot read an action applied from a state it did not come from.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/physics_counterfactual_check.py \\
        --ckpt D=wm/runs/fmd_beh24_s0/b1_lora_c3/ckpt_lib_s4.pt
Needs CoppeliaSim on port 23000 (ego render) and the GPU for V-JEPA2 (run when no training holds it).
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/control", "sim/scene", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from b1_policy_runtime import B1Walker  # noqa: E402
from counterfactual_truth_check import build_scene  # noqa: E402
from ego_camera import check_ego_view  # noqa: E402
from rollout_state_action_anova import Models, corr  # noqa: E402
from wm.data.embodiment import body_velocity, yaw_rate  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import load_candidates  # noqa: E402

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
K = 5                      # rendered frames per branch (= the stride-5 model step)
PER_FRAME = 2.5            # policy steps per rendered frame (50 Hz policy, 20 Hz frames)


def snapshot(w):
    # B1Walker.snapshot: integration state + sensordata + last action + gait clock + settings. (Before
    # 2026-10-01 sensordata was recomputed by mj_forward on restore; the policy reads the PRE-step sensor
    # values, so every branch started ~1e-3 rad off the uninterrupted run within 20 frames.)
    return w.snapshot()


def restore(w, snap):
    w.restore(snap)


def clip_policy(path):
    """The policy the start-state candidate was recorded with: its `policy` field, else the exact replay
    (build_b1_cf_branches.locate). Raises if neither determines it -- never a default walker."""
    with np.load(path, allow_pickle=True) as f:
        if "policy" in f.files:
            return str(f["policy"])
    sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
    import build_b1_cf_branches as B
    try:
        return B.locate(path)["policy"]
    except RuntimeError as e:
        raise SystemExit(f"{path}: recording policy unknown (no `policy` field, replay does not "
                         f"reproduce it): {e}")


def run_frames(w, cmds, render):
    """Execute one command per rendered frame; return frames, base positions / quats, per-frame action."""
    frames, pos, quat, act, acc = [], [], [], [], 0.0
    for cmd in cmds:
        acc += PER_FRAME
        n = int(acc); acc -= n
        for _ in range(n):
            a = w.policy_step(cmd)
        p, q, j, _ = w.state()
        frames.append(render(p, q, j)); pos.append(p); quat.append(q); act.append(a)
    return frames, np.asarray(pos), np.asarray(quat), np.asarray(act, np.float32)


def achieved_froude(pos, quat):
    """Mean Froude (fwd, lat, yaw) over the branch, from the simulated body's own poses."""
    v = body_velocity(pos, quat, 0.05, "b1")
    h = float(np.median(pos[:, 2]))
    w = np.asarray(yaw_rate(quat, 0.05, "b1", h)).ravel()
    return np.array([v[1:, 0].mean(), v[1:, 1].mean(), w[1:].mean()])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=CKPT")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--states", type=int, default=8, help="start states (library clips, spread)")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30], help="frames walked before the branch")
    ap.add_argument("--warmup", type=int, default=45, help="policy steps before recording, as the physics loop")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--cache", default="results/wm/cache/physics_counterfactual_b1_v2.pt",
                    help="v2: start-state policy = the candidate's own, sensordata restored (2026-10-01)")
    args = ap.parse_args()
    dev = "cuda"
    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    n = len(cands)
    cmds = [np.asarray(np.load(c["path"], allow_pickle=True)["command"], np.float32) for c in cands]
    S = [int(v) for v in np.linspace(0, n - 1, args.states).round()]
    cache = os.path.join(ROOT, args.cache)

    if os.path.exists(cache):
        D = torch.load(cache)
        print(f"branches from cache: {cache}")
    else:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        from vjepa2_encoder import VJEPA2FrameEncoder
        sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
        render = build_scene(sim, args.scene)
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
        D = {}
        for s in S:
            for t in args.steps:
                # the start state is walked with candidate s's recorded commands, so by candidate s's own
                # policy (heading gains need not match: the recorded yaw command is replayed, no PI here)
                w = B1Walker(policy=clip_policy(cands[s]["path"]))
                for _ in range(args.warmup):
                    w.policy_step(cmds[s][0])
                w.d.qpos[0:2] -= w.d.qpos[0:2].copy()       # centre in the room, as the physics loop
                mujoco.mj_forward(w.m, w.d)
                _, pre_pos, pre_quat, _ = run_frames(w, cmds[s][:t], render)
                p, q, j, _ = w.state()
                img_s = render(p, q, j)
                if not D:
                    print(f"view check: corr {check_ego_view(img_s, ref, min_corr=0.97):.3f}", flush=True)
                snap = snapshot(w)
                futs, truth, chunks = [], [], []
                for a in range(n):
                    restore(w, snap)
                    ca = cmds[a][min(t, len(cmds[a]) - K):][:K]
                    fr, pos, quat, act = run_frames(w, ca, render)
                    futs.append(fr[-1])
                    truth.append(achieved_froude(np.concatenate([p[None], pos]), np.concatenate([q[None], quat])))
                    chunks.append(act)
                emb = encode_clip(enc, np.stack([img_s] + futs), 2).float().cpu().half()
                D[(t, s)] = {"e_s": emb[:1], "e_f": emb[1:], "truth": np.stack(truth), "chunk": np.stack(chunks)}
                print(f"  branched t={t} s={s}", flush=True)
        torch.save(D, cache)
        del enc
        torch.cuda.empty_cache()

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        rd = lambda z: m.md.body(None, z).cpu().numpy() * std + mean
        R = {k: [] for k in ("direct", "read real", "read pred")}
        for key, rec in sorted(D.items()):
            e_s = rec["e_s"].float().to(dev).expand(n, -1, -1)
            e_f = rec["e_f"].float().to(dev)
            z = m.proj(torch.as_tensor(rec["chunk"], device=dev), "b1")
            reads = {"direct": rd(z), "read real": rd(m.itm(e_s, e_f)),
                     "read pred": rd(m.itm(e_s, m.ftm_step(e_s, z)))}
            for k, f in reads.items():
                R[k].append([corr(f[:, j], rec["truth"][:, j]) for j in range(3)])
        print(f"\n=== {name}: physics counterfactuals, {len(D)} start states x {n} branches; "
              f"r vs achieved Froude, fwd / lat / yaw")
        for k, v in R.items():
            print(f"  {k:<10}" + " / ".join(f"{x:.2f}" for x in np.nanmean(np.asarray(v), 0)))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
