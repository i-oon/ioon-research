"""F186 corrected: 68% on recorded conditions is not near-ceiling. Where does each scorer lose?

    .venv/bin/python3 scripts/diagnostics/planning/condition_confusion.py

**The correction this diagnoses.** F186 read `state`'s 3.95x signal/floor on recorded conditions as
confirming candidate separation was the whole wall. It wasn't the whole story: 68% win rate is far
from the ~100% a scorer with no error of its own should get when candidates are already 4x above the
noise floor apart. Two layers are tangled in that 68% -- `perturb` stacks candidate-similarity AND
scorer-inaccuracy; `conditions` isolates scorer-inaccuracy alone, since the candidates are no longer
the limiting factor. This measures that layer directly, without touching candidate generation and
without executing anything in the simulator beyond reaching each branch point once.

**Ground truth for "which condition is actually best" does not need a fresh rollout.** Each of the
twelve recorded conditions has its own clip, and that clip's own whole-clip-average body motion (the
same quantity `body_goal` uses to define the goal itself) is the honest answer to "what does this
condition actually do" -- no model, no prediction, read directly off recorded telemetry. Comparing
that fixed ranking against what each scorer *predicts* isolates scorer error from execution noise.

**Magnitude vs direction.** Each condition's motion is a 3-vector (forward, lateral, yaw). A scorer
that picks the wrong condition but in the same *direction* (e.g. confuses two forward speeds) is
making a finer, arguably more forgivable error than one that picks the wrong *direction* entirely
(e.g. confuses forward with turning). Reported separately, per scorer, on the same branch points, so
`state`/`ridge`/`direct` can be compared on the identical mistakes rather than aggregate rates alone.

Diagnosis only. No tuning, no retraining, no new checkpoints.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from teacher_student_insect import Student, body_goal, load_teacher, pooled  # noqa: E402
from wm.adapt3 import gather  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


def family(cond):
    return "side" if cond.startswith("side") else cond.split("_")[0]


def gram(a, b, device, chunk=64, bchunk=256):
    out = torch.empty(len(a), len(b), dtype=torch.float64)
    for i in range(0, len(a), chunk):
        q = a[i:i + chunk].to(device).float()
        for j in range(0, len(b), bchunk):
            out[i:i + chunk, j:j + bchunk] = (
                q @ b[j:j + bchunk].to(device).float().T).double().cpu()
        del q
        torch.cuda.empty_cache()
    return out.numpy()


def fit_ridge(ck, cfg, itm, channels, train_data, cache_path, train_stride, device):
    lag = max(1, cfg.action_lag)
    cache = torch.load(os.path.join(ROOT, cache_path), map_location="cpu", mmap=True)
    tr = gather(os.path.join(ROOT, train_data), "hexapod", None, ck, cache, 2, lag, device)
    paths = sorted(glob.glob(os.path.join(ROOT, train_data, "*.npz")))

    E, Z, Y, cid = [], [], [], []
    with torch.no_grad():
        for ci, (c, p) in enumerate(zip(tr, paths)):
            bm = np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, channels]
            e = c["e"].float()
            for t in range(1, min(len(e) - 2, len(bm)), train_stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                Z.append(itm(e_t, e1)[0].float().cpu())
                E.append(e[t].flatten().half())
                Y.append(torch.tensor(bm[t], dtype=torch.float64))
                cid.append(ci)
    E, Z = torch.stack(E), torch.stack(Z)
    Y = torch.stack(Y).numpy()
    cid = np.array(cid)

    mean_e = E.float().mean(0, keepdim=True)
    for i in range(0, len(E), 256):
        E[i:i + 256] = (E[i:i + 256].float() - mean_e).half()
    mean_z = Z.numpy().astype(np.float64).mean(0, keepdims=True)
    Zc = Z.numpy().astype(np.float64) - mean_z

    print("  fitting ridge: building Gram matrices", flush=True)
    raw = gram(E, E, device); s_e = max(np.mean(np.diag(raw)), 1e-12)
    Kee = raw / s_e
    Kzz_raw = Zc @ Zc.T; s_z = max(np.mean(np.diag(Kzz_raw)), 1e-12)
    K = Kee + Kzz_raw / s_z

    clips = sorted(set(cid.tolist()))
    val = set(clips[1::3])
    va = np.array([c in val for c in cid]); fit = ~va
    mu, sd = Y[fit].mean(0), Y[fit].std(0) + 1e-9
    Ys = (Y - mu) / sd

    best_a, best_v = ALPHAS[0], -1e9
    for a in ALPHAS:
        w = np.linalg.solve(K[np.ix_(fit, fit)] + a * np.eye(fit.sum()), Ys[fit])
        pred = K[np.ix_(va, fit)] @ w
        ss = ((pred - Ys[va]) ** 2).sum()
        r2 = 1 - ss / max(((Ys[va] - Ys[fit].mean(0)) ** 2).sum(), 1e-12)
        if r2 > best_v:
            best_v, best_a = r2, a
    w_full = np.linalg.solve(K + best_a * np.eye(len(K)), Ys)
    print(f"  ridge fit: alpha {best_a:.4g}, held-out-clip R2 {best_v:.3f}\n")

    return {"E": E.to(device).float(), "Zc": torch.tensor(Zc, dtype=torch.float64, device=device),
           "mean_e": mean_e.to(device), "s_e": s_e,
           "mean_z": torch.tensor(mean_z, dtype=torch.float64, device=device), "s_z": s_z,
           "w": torch.tensor(w_full, dtype=torch.float64, device=device), "mu": mu, "sd": sd}


def score_ridge(fitted, e_t, z_batch, device):
    with torch.no_grad():
        ec = e_t.float().flatten() - fitted["mean_e"].flatten()
        k_e = (fitted["E"] @ ec) / fitted["s_e"]
        zc = z_batch.double() - fitted["mean_z"]
        k_z = (zc @ fitted["Zc"].T) / fitted["s_z"]
        K_row = k_e.double().unsqueeze(0) + k_z
        pred = K_row @ fitted["w"]
    return pred.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_state/teacher_state.pt")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--goal_clip",
                    default="data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="medauroidea_c08f09t09.ttt")
    ap.add_argument("--ego_seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

    fitted = fit_ridge(ck, cfg, itm, channels, args.train_data, args.cache, 2, device)

    names = tuple(s.split("=", 1)[0] for s in cfg.sources) or ("default",)
    state_model = StateHead(cfg, len(channels), names).to(device).eval()
    state_model.load_state_dict(ck["state"])

    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", channels)
    goal_std = (goal - mean_s) / std_s
    goal_std_t = torch.tensor(goal_std, dtype=torch.float32, device=device)

    # ---- the fixed ground truth: each condition's own whole-clip average motion ------------------
    cand, seen = {}, set()
    for p_ in sorted(glob.glob(os.path.join(ROOT, args.train_data, "*.npz"))):
        with np.load(p_, allow_pickle=True) as z_:
            c = str(z_["condition"])
        if c not in seen:
            seen.add(c); cand[c] = p_
    conds = sorted(cand)
    acts = {c: torch.tensor(np.asarray(load(cand[c], REGISTRY["hexapod"])["actions"]),
                            dtype=torch.float32) for c in conds}
    true_motion = {c: body_goal(cand[c], "hexapod", channels) for c in conds}
    true_std = {c: (true_motion[c] - mean_s) / std_s for c in conds}
    true_dist = {c: float(np.linalg.norm(true_std[c] - goal_std)) for c in conds}
    true_best = min(conds, key=lambda c: true_dist[c])
    print(f"goal {np.round(goal, 4)}")
    print(f"true best condition by recorded motion: {true_best} "
          f"(dist {true_dist[true_best]:.4f})")
    print("true ranking:", ", ".join(f"{c}:{true_dist[c]:.3f}" for c in
                                     sorted(conds, key=lambda c: true_dist[c])))
    print()

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])
    seed = np.asarray(clip["actions"], np.float32)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    branch = np.linspace(6, min(len(seed) - args.horizon - 1, 50), args.states).astype(int)
    EGO_CAM = dict(ego=True, cam_fov=90.0, ego_box=8.0)

    def run_to(t_branch):
        def policy(observation, t):
            return seed[t]
        return drive_and_record(sim, args.scene, seed[:t_branch + args.horizon], 0.0, 20,
                                cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy,
                                **dict(EGO_CAM, ego_seed=args.ego_seed))

    scorers = ("f179", "direct", "ridge", "state")
    picks = {s: [] for s in scorers}
    true_ranks = {c: r for r, c in enumerate(sorted(conds, key=lambda c: true_dist[c]))}

    for bt in branch:
        frames, _a, _f, _h, _o = run_to(int(bt))
        obs = frames[int(bt) - 1]
        with torch.no_grad():
            e_full = encode_clip(encoder, np.asarray(obs)[None], 1).float().to(device)
            a = torch.stack([acts[c][min(int(bt), len(acts[c]) - 1)] for c in conds]).to(device)
            z = proj(a, "hexapod")

            roll = e_full.expand(len(conds), -1, -1)
            for _ in range(args.horizon):
                roll = ftm(roll, z)
            m_f179 = md.body(None, itm(e_full.expand(len(conds), -1, -1), roll))
            m_direct = md.body(None, z)
            m_ridge = score_ridge(fitted, e_full[0], z, device)
            one_step = ftm(e_full.expand(len(conds), -1, -1), z)
            m_state = state_model(one_step - e_full.expand(len(conds), -1, -1), z, "hexapod")

            for name, m in (("f179", m_f179), ("direct", m_direct),
                           ("ridge", m_ridge), ("state", m_state)):
                if m.dim() == 1:
                    m = m.unsqueeze(-1)
                k = min(m.shape[-1], len(channels))
                pick = int((m[:, :k] - goal_std_t[:k]).pow(2).mean(-1).argmin())
                picks[name].append(conds[pick])
        print(f"  t={bt:3d}  " + "  ".join(f"{s}:{picks[s][-1]}" for s in scorers), flush=True)

    print(f"\n{'scorer':>8}{'accuracy':>12}{'mean true-rank of pick':>26}")
    for name in scorers:
        acc = np.mean([p == true_best for p in picks[name]])
        mean_rank = np.mean([true_ranks[p] for p in picks[name]])
        print(f"{name:>8}{acc:>12.0%}{mean_rank:>26.2f}   (0 = always picks the true best, "
              f"{len(conds) - 1} = always picks the true worst)")

    # ---- confusion: what gets picked when the true best is NOT what's picked ---------------------
    print("\nconfusion (true best -> what was actually picked, when wrong):")
    for name in scorers:
        wrong = [(true_best, p) for p in picks[name] if p != true_best]
        if not wrong:
            print(f"  {name}: never wrong")
            continue
        from collections import Counter
        cnt = Counter(wrong)
        same_family = sum(1 for (t, p) in wrong if family(t) == family(p))
        print(f"  {name}: {len(wrong)}/{len(picks[name])} wrong, "
              f"{same_family}/{len(wrong)} within the same family ({family(true_best)})")
        for (t, p), n in cnt.most_common(5):
            print(f"    {t} mistaken for {p}: {n}x  (true dist {true_dist[t]:.3f} vs "
                  f"picked dist {true_dist[p]:.3f})")

    # ---- magnitude vs direction, on the SAME picks, comparable across scorers ---------------------
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    true_dir = unit(true_std[true_best])   # direction of the true-best condition's own motion
    goal_dir = unit(goal_std)
    print(f"\ncosine(true-best direction, goal direction): {float(true_dir @ goal_dir):.3f}"
          "  (ceiling for any scorer reading direction alone)")
    print("\nmagnitude vs direction of each scorer's mistake (only when wrong):")
    for name in scorers:
        cos_errs, mag_errs = [], []
        for p in picks[name]:
            if p == true_best:
                continue
            p_dir = unit(true_std[p])
            cos_errs.append(float(p_dir @ true_dir))
            mag_errs.append(abs(np.linalg.norm(true_std[p]) - np.linalg.norm(true_std[true_best])))
        if not cos_errs:
            continue
        print(f"  {name}: mean cosine(picked, true-best) = {np.mean(cos_errs):.3f}  "
              f"(1.0 = right direction, wrong speed only)   "
              f"mean |magnitude gap| = {np.mean(mag_errs):.3f}")


if __name__ == "__main__":
    main()
