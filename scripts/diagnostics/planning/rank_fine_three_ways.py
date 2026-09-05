"""F179's local arm, three scorers side by side: embedding rollout, state-blind body head, ridge.

    .venv/bin/python3 scripts/diagnostics/planning/rank_fine_three_ways.py

Closes the one thing `rank_on_delta_state.py` could not: **accuracy**, not separation, on fine
perturbations, judged in the simulator exactly as F179 was.

    f179       proj(a) -> FTM rollout -> ITM -> body head      state-aware, embedding-mediated
    direct     proj(a) -> body head                            state-BLIND by construction
    ridge      [e_t, proj(a)] -> body motion, kernel ridge      state-aware, Delta-state-direct

**`ridge` is the untested arm, and it is not the rebuild.** The FTM(e_t,z)->body-motion network
does not exist; this is a closed-form linear ridge over `[e_t, z]` in kernel form, fitted offline on
cached embeddings the same way `target_action_share.py` measured 0.404 R2. It is a weaker function
class than a trained network, so a positive result here is a lower bound on what a real rebuild could
do, not the rebuild's own number. It is fit with `z = ITM(e_t, e_t+1)` -- the same regime the
existing body head was trained under -- and evaluated with `z = proj(a)` at ranking time, exactly
paralleling how the body head is normally used. **This is the fair comparison**: same target, same
kind of z-shift between fit and use, different function class and different aggregation of the
inputs.

**`direct` is included on purpose, not as a straw man.** It is state-blind: `proj(a) -> body head`
never reads `e_t`, so it scores what an action does *on average*, and `rank_on_delta_state.py` found
it separates and coarse-ranks well precisely because it bypasses the world model. If it also wins
here, the result is F184 again -- a win that argues against the world model's contribution, not for
Delta-state as a target.

All three share the branch points, the perturbation draw, `repeat_control`'s noise floor, and the
realizability checks, so a difference between them is the scorer and nothing else.

**Trains nothing beyond the offline ridge fit, which touches no checkpoint file.**
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
from wm.data.embodiment import REGISTRY, body_velocity, load, yaw_rate  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


def family(cond):
    return "side" if cond.startswith("side") else cond.split("_")[0]


def channels_of(pos, quat, dt="0.05"):
    dt = float(dt)
    v = body_velocity(pos, quat, dt, "hexapod")
    w = yaw_rate(quat, dt, "hexapod", float(np.median(pos[:, 2])))
    return np.concatenate([v, np.asarray(w).reshape(len(v), 1)], axis=1)


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
    """[e_t, z=ITM(e_t,e_t+1)] -> standardised body motion, dual kernel ridge.

    Returns everything `score_ridge` needs: the training features (kept on GPU), the block scales,
    the dual weights, and the offline R2 as a sanity check against `target_action_share.py`.
    """
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
    w_full = np.linalg.solve(K + best_a * np.eye(len(K)), Ys)     # refit on everything for use
    print(f"  ridge fit: alpha {best_a:.4g}, held-out-clip R2 {best_v:.3f} "
          f"(target_action_share.py measured 0.404 with a different validation split)\n")

    return {"E": E.to(device).float(),                 # kept resident: ~1.1 GB, reused every call
           "Zc": torch.tensor(Zc, dtype=torch.float64, device=device),
           "mean_e": mean_e.to(device), "s_e": s_e,
           "mean_z": torch.tensor(mean_z, dtype=torch.float64, device=device), "s_z": s_z,
           "w": torch.tensor(w_full, dtype=torch.float64, device=device), "mu": mu, "sd": sd}


def score_ridge(fitted, e_t, z_batch, device):
    """Predicted standardised body motion for a batch of candidate `z`, at a fixed `e_t`."""
    with torch.no_grad():
        ec = e_t.float().flatten() - fitted["mean_e"].flatten()
        k_e = (fitted["E"] @ ec) / fitted["s_e"]                       # (n_train,)
        zc = z_batch.double() - fitted["mean_z"]
        k_z = (zc @ fitted["Zc"].T) / fitted["s_z"]                    # (batch, n_train)
        K_row = k_e.double().unsqueeze(0) + k_z
        pred = K_row @ fitted["w"]                                     # (batch, 3) standardised
    return pred.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--student", default="wm/runs/students/insect_bc_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--train_data", default="data/egocentric/beh12_c10f10t10_ego_flat")
    ap.add_argument("--cache", default="results/wm/cache/ego_hex.pt")
    ap.add_argument("--goal_clip",
                    default="data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep0.npz")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--states", type=int, default=15)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--candidates", choices=("perturb", "conditions"), default="perturb",
                    help="**the ceiling check.** `perturb` is F179's setting -- Gaussian noise "
                         "around the student's guess, 2.5% separation, none of the four scorers "
                         "beat a coin on it (F185). `conditions` swaps in the twelve recorded "
                         "behaviours from --train_data as candidates instead: on-manifold, large "
                         "separation, F143 read the old f179-style scorer at 85-90% on this. If a "
                         "scorer can't clear this either, it cannot rank at all, independent of "
                         "candidate generation; if it clears this but not `perturb`, the wall really "
                         "is separation, not the scorer.")
    ap.add_argument("--repeat_control", type=int, default=4)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="medauroidea_c08f09t09.ttt")
    ap.add_argument("--ego_seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

    fitted = fit_ridge(ck, cfg, itm, channels, args.train_data, args.cache, 2, device)

    # **The fourth, real arm.** `f179`/`direct`/`ridge` above predate `StateHead` -- none of them
    # calls it, so a checkpoint that has one was never actually put through this test. `ridge` is
    # explicitly a lower bound on what a trained network could do (see module docstring); this is
    # the trained network itself. Same z, same horizon-rolled `delta` the f179 arm already computes,
    # so the only thing that differs from f179 is how body motion gets read off it.
    state_model = None
    if "state" in ck:
        names = tuple(s.split("=", 1)[0] for s in cfg.sources) or ("default",)
        state_model = StateHead(cfg, len(channels), names).to(device).eval()
        state_model.load_state_dict(ck["state"])
        print("state head loaded from checkpoint -- scoring with it too\n")
    else:
        print("checkpoint has no state head -- skipping that arm\n")

    st = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(st["token_dim"], st["goal_dim"], st["action_dim"]).to(device).eval()
    student.load_state_dict(st["student"])

    # **The ceiling pool.** `--train_data`, not the held-out `--data` -- the twelve conditions have
    # to come from the body the checkpoint actually trained on, or a bad score here would measure
    # an unseen body instead of a scorer's ceiling (the same trap `--exclude` guards against in
    # `wm.fit_projector`).
    conds_acts = {}
    if args.candidates == "conditions":
        cand, seen = {}, set()
        for p_ in sorted(glob.glob(os.path.join(ROOT, args.train_data, "*.npz"))):
            with np.load(p_, allow_pickle=True) as z_:
                c = str(z_["condition"])
            if c not in seen:
                seen.add(c); cand[c] = p_
        conds_acts = {c: torch.tensor(np.asarray(load(cand[c], REGISTRY["hexapod"])["actions"]),
                                      dtype=torch.float32) for c in sorted(cand)}
        print(f"ceiling pool: {len(conds_acts)} recorded conditions from {args.train_data}\n")

    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", channels)
    goal_std = torch.tensor((goal - mean_s) / std_s, dtype=torch.float32, device=device)
    with np.load(os.path.join(ROOT, args.goal_clip), allow_pickle=True) as z_:
        goal_cond = str(z_["condition"])
    print(f"goal {goal_cond} {np.round(goal, 4)}\n")

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])
    seed = np.asarray(clip["actions"], np.float32)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    branch = np.linspace(6, min(len(seed) - args.horizon - 1, 50), args.states).astype(int)
    EGO_CAM = dict(ego=True, cam_fov=90.0, ego_box=8.0)

    def run_to(t_branch, tail):
        def policy(observation, t):
            return seed[t] if t < t_branch else tail
        return drive_and_record(sim, args.scene, seed[:t_branch + args.horizon], 0.0, 20,
                                cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy,
                                **dict(EGO_CAM, ego_seed=args.ego_seed))

    allacts = np.concatenate([np.asarray(load(p, REGISTRY["hexapod"])["actions"], np.float64)
                              for p in sorted(glob.glob(os.path.join(ROOT, args.train_data,
                                                                     "*.npz")))])
    joint_sd = torch.tensor(allacts.std(0), dtype=torch.float32, device=device)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    scorers = ("f179", "direct", "ridge") + (("state",) if state_model is not None else ())
    stats = {s: dict(wins=0, ties=0, rows=[]) for s in scorers}
    repeats = []

    for bt in branch:
        frames, _a, _f, heads, oris = run_to(int(bt), seed[int(bt)])
        obs = frames[int(bt) - 1]
        with torch.no_grad():
            e_full = encode_clip(encoder, np.asarray(obs)[None], 1).float().to(device)
            g_t = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
            base = student.act(pooled(e_full), g_t)
            if args.candidates == "conditions":
                # index 0 stays the student's own action, so `pick == 0` keeps its meaning in
                # both modes -- only what fills the rest of the pool changes
                pool = torch.stack([conds_acts[c][min(int(bt), len(conds_acts[c]) - 1)]
                                    for c in conds_acts]).to(device)
                allc = torch.cat([base, pool])
            else:
                pert = base + (args.sigma * student.std.to(device)) * torch.randn(
                    args.samples, base.shape[-1], generator=g).to(device)
                allc = torch.cat([base, pert])
            z = proj(allc, "hexapod")

            roll = e_full.expand(len(allc), -1, -1)
            for _ in range(args.horizon):
                roll = ftm(roll, z)
            m_f179 = md.body(None, itm(e_full.expand(len(allc), -1, -1), roll))
            m_direct = md.body(None, z)
            m_ridge = score_ridge(fitted, e_full[0], z, device)
            scored = [("f179", m_f179), ("direct", m_direct), ("ridge", m_ridge)]
            if state_model is not None:
                # single FTM application, matching both StateHead's training regime (one-step
                # delta) and how `ridge` itself is evaluated here -- not the horizon-rolled `roll`
                # f179 uses, which would hand the head a 3x-compounded delta it never trained on
                one_step = ftm(e_full.expand(len(allc), -1, -1), z)
                delta = one_step - e_full.expand(len(allc), -1, -1)
                scored.append(("state", state_model(delta, z, "hexapod")))

            picks = {}
            for name, m in scored:
                if m.dim() == 1:
                    m = m.unsqueeze(-1)
                k = min(m.shape[-1], len(channels))
                picks[name] = int((m[:, :k] - goal_std[:k]).pow(2).mean(-1).argmin())

        line = f"  t={bt:3d}"
        for name in scorers:
            pick = picks[name]
            a = allc[pick]
            _fr, _ac, _fo, hh, oo = run_to(int(bt), a.cpu().numpy())
            hh, oo = np.asarray(hh, np.float64), np.asarray(oo, np.float64)
            got = channels_of(hh[-args.horizon - 1:], oo[-args.horizon - 1:])[:, channels].mean(0)
            d_t = float(np.linalg.norm(got - goal))
            if name == "f179":
                _frs, _acs, _fos, hhs, oos = run_to(int(bt), base[0].cpu().numpy())
                hhs, oos = np.asarray(hhs, np.float64), np.asarray(oos, np.float64)
                gots = channels_of(hhs[-args.horizon - 1:],
                                   oos[-args.horizon - 1:])[:, channels].mean(0)
                d_s = float(np.linalg.norm(gots - goal))
                stats["_d_s"] = stats.get("_d_s", []) + [d_s]
                if len(repeats) < args.repeat_control:
                    _f2, _a2, _o2, hh2, oo2 = run_to(int(bt), base[0].cpu().numpy())
                    hh2, oo2 = np.asarray(hh2, np.float64), np.asarray(oo2, np.float64)
                    again = channels_of(hh2[-args.horizon - 1:],
                                        oo2[-args.horizon - 1:])[:, channels].mean(0)
                    repeats.append(abs(float(np.linalg.norm(again - goal)) - d_s))
            d_s = stats["_d_s"][-1]
            stats[name]["wins"] += d_t < d_s
            stats[name]["ties"] += pick == 0
            stats[name]["rows"].append((d_s, d_t))
            line += f"  {name}:{'W' if d_t < d_s else '.'}({d_t:.3f})"
        print(line, flush=True)

    n = len(branch)
    print(f"\n{'scorer':>10}{'wins':>8}{'rate':>8}{'kept student':>14}{'mean d_teacher':>16}")
    d_s_mean = np.mean(stats["_d_s"])
    print(f"{'student':>10}{'':>8}{'':>8}{'':>14}{d_s_mean:>16.4f}")
    for name in scorers:
        s = stats[name]
        d_t_mean = np.mean([r[1] for r in s["rows"]])
        print(f"{name:>10}{s['wins']:>8}{s['wins'] / n:>8.0%}{s['ties']:>14}{d_t_mean:>16.4f}")
    if repeats:
        floor = float(np.mean(repeats))
        print(f"\nsimulator noise floor (same action twice, {len(repeats)} states): {floor:.4f}")
        for name in scorers:
            gap = np.mean([r[0] - r[1] for r in stats[name]["rows"]])
            print(f"  {name}: mean(student-teacher) {gap:+.4f}, signal/floor {gap / max(floor, 1e-9):.2f}x")
    print("\na coin is 50%. F179 measured f179-scorer at 47% under this exact protocol before.")


if __name__ == "__main__":
    main()
