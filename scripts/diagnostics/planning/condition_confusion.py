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
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.state_head import StateHead  # noqa: E402

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)

# Must match scripts/diagnostics/objective_experiments/rssm_stage1_gate.py exactly.
RSSM_H_DIM, RSSM_Z_DIM, RSSM_POOL_DIM, RSSM_ACTION_DIM, RSSM_BODY_DIM = 256, 32, 1408, 18, 3


class _RSSM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRUCell(RSSM_Z_DIM + RSSM_ACTION_DIM, RSSM_H_DIM)
        self.post = torch.nn.Sequential(torch.nn.Linear(RSSM_H_DIM + RSSM_POOL_DIM, 256),
                                        torch.nn.GELU(), torch.nn.Linear(256, 2 * RSSM_Z_DIM))
        self.prior = torch.nn.Sequential(torch.nn.Linear(RSSM_H_DIM, 128), torch.nn.GELU(),
                                         torch.nn.Linear(128, 2 * RSSM_Z_DIM))
        self.froude = torch.nn.Sequential(torch.nn.Linear(RSSM_H_DIM + RSSM_Z_DIM, 128),
                                          torch.nn.GELU(), torch.nn.Linear(128, RSSM_BODY_DIM))

    def step_posterior(self, h, z_prev, a_prev, e_t):
        h = self.cell(torch.cat([z_prev, a_prev], -1), h)
        mu_q, _ = self.post(torch.cat([h, e_t], -1)).chunk(2, -1)
        return h, mu_q   # mean, not sampled -- deterministic at eval time

    def step_prior(self, h, z_prev, a_prev):
        h = self.cell(torch.cat([z_prev, a_prev], -1), h)
        mu_p, _ = self.prior(h).chunk(2, -1)
        pred = self.froude(torch.cat([h, mu_p], -1))
        return h, mu_p, pred


# Must match scripts/diagnostics/objective_experiments/ftm_froude_stopgrad_probe.py exactly.
FROUDE_POOL_DIM, FROUDE_HIDDEN = 1408, 128


class _CleanFroudeHead(torch.nn.Module):
    """Reads pool(FTM(e_t,z)) -- FTM's OWN predicted next-embedding -- with no delta, no z_proj,
    trained isolated (stop-gradient) from FTM's own training. The untested cell: does the world
    model's rollout, read directly in Froude space, rank candidates -- as opposed to `direct`
    (no FTM at all) or `f179` (multi-step embedding rollout decoded via ITM)."""
    def __init__(self, n_channels):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(FROUDE_POOL_DIM),
                                       torch.nn.Linear(FROUDE_POOL_DIM, FROUDE_HIDDEN),
                                       torch.nn.GELU(), torch.nn.Linear(FROUDE_HIDDEN, n_channels))

    def forward(self, pooled_ftm_output):
        return self.net(pooled_ftm_output)


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


def fit_ridge(ck, cfg, itm, channels, train_data, cache_path, train_stride, device, encoder=None):
    lag = max(1, cfg.action_lag)
    cache = torch.load(os.path.join(ROOT, cache_path), map_location="cpu", mmap=True)
    # a real encoder, not None: the default cache (96 entries) only covers the original 48-clip
    # datasets. Pointing --train_data at a larger set (e.g. the 240-clip beh12_c10f10t10_more)
    # means most clips are cache misses, and gather() needs something to encode them with.
    if encoder is None:
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    before = len(cache)
    paths = sorted(glob.glob(os.path.join(ROOT, train_data, "*.npz")))

    # One clip at a time, not wm.adapt3.gather's full-list return: that holds every clip's whole
    # embedding in RAM simultaneously (240 clips x ~47MB half-precision = ~11GB), which on top of
    # the embedding cache OOM-killed this on a 31GB box with no swap. Only the per-timestep E/Z/Y
    # rows need to survive past one clip -- the full per-clip tensor does not.
    E, Z, Y, cid = [], [], [], []
    with torch.no_grad():
        for ci, p in enumerate(paths):
            clip = load(p, REGISTRY["hexapod"])
            if p not in cache:
                cache[p] = encode_clip(encoder, clip["frames"], 2).cpu().half()
            e = cache[p].float()
            off = offset_for(ck, "hexapod")
            if off is not None:
                e = e - off.cpu()
            bm = np.asarray(clip["body_motion"])[:, channels]
            for t in range(1, min(len(e) - 2, len(bm)), train_stride):
                e_t, e1 = e[t:t + 1].to(device), e[t + 1:t + 2].to(device)
                Z.append(itm(e_t, e1)[0].float().cpu())
                E.append(e[t].flatten().half())
                Y.append(torch.tensor(bm[t], dtype=torch.float64))
                cid.append(ci)
            del e
    if len(cache) > before:
        # write to a temp file and rename over the original -- the loaded `cache` dict still holds
        # mmap'd tensors backed by that same file, so overwriting it in place while those are live
        # would corrupt the mapping
        full_path = os.path.join(ROOT, cache_path)
        tmp_path = full_path + ".tmp"
        torch.save(cache, tmp_path)
        os.replace(tmp_path, full_path)
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

    # E stays half-precision on GPU (~5.4GB for the 240-clip set): score_ridge only ever does a
    # matrix-VECTOR product against it, so upcasting the whole matrix to float32 (~10.2GB) was
    # never necessary and OOM'd this 10.57GB card once the dataset grew 5x.
    return {"E": E.to(device), "Zc": torch.tensor(Zc, dtype=torch.float64, device=device),
           "mean_e": mean_e.to(device), "s_e": s_e,
           "mean_z": torch.tensor(mean_z, dtype=torch.float64, device=device), "s_z": s_z,
           "w": torch.tensor(w_full, dtype=torch.float64, device=device), "mu": mu, "sd": sd}


def score_ridge(fitted, e_t, z_batch, device):
    with torch.no_grad():
        ec = e_t.float().flatten() - fitted["mean_e"].flatten()
        # the matvec itself in float32, but chunked over E's rows: casting all of E to float32 at
        # once (~10.2GB for 240 clips) is what OOM'd this 10.57GB card; doing the matvec in half
        # precision throughout instead (the first attempted fix) avoided the OOM but collapsed
        # ridge to a single constant answer regardless of goal (0% accuracy, 40/40 same wrong
        # pick) -- a real precision-loss bug, not a data effect. Chunking keeps both properties:
        # full float32 precision, bounded peak memory.
        E = fitted["E"]
        k_e = torch.empty(len(E), dtype=torch.float32, device=device)
        for i in range(0, len(E), 256):
            k_e[i:i + 256] = E[i:i + 256].float() @ ec
        k_e = k_e / fitted["s_e"]
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
    ap.add_argument("--rssm", default="", help="path to an rssm_stage1_gate.py checkpoint; "
                    "adds an 'rssm' scorer that rolls the RSSM's PRIOR forward --horizon steps "
                    "from a posterior built by teacher-forcing on real preceding frames 0..bt-1")
    ap.add_argument("--ftm_froude", default="", help="path to an ftm_froude_stopgrad_probe.py "
                    "checkpoint; adds an 'ftm_froude' scorer: FTM(e_t,z) -> pooled -> the clean, "
                    "stop-gradient-trained Froude head (uses the z=proj head, the control-relevant "
                    "one -- candidates only ever have an action, never a real next frame)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]

    fitted = fit_ridge(ck, cfg, itm, channels, args.train_data, args.cache, 2, device)
    # this card is only 10.57GB, and fitted["E"] alone is ~5.4GB resident for the rest of the run;
    # release whatever fit_ridge's own local encoder/activations left cached before the rollout
    # loop's own models and encoder claim their share
    torch.cuda.empty_cache()

    rssm = rssm_mean_s = rssm_std_s = rssm_channels = None
    if args.rssm:
        rssm_ck = torch.load(os.path.join(ROOT, args.rssm), map_location=device, weights_only=False)
        rssm = _RSSM().to(device).eval()
        rssm.load_state_dict(rssm_ck["model"])
        for p in rssm.parameters():
            p.requires_grad_(False)
        rssm_channels = rssm_ck["channels"]
        rssm_mean_s, rssm_std_s = rssm_ck["mean_s"], rssm_ck["std_s"]
        print(f"loaded rssm scorer from {args.rssm}")

    froude_head = None
    if args.ftm_froude:
        fh_ck = torch.load(os.path.join(ROOT, args.ftm_froude), map_location=device, weights_only=False)
        froude_head = _CleanFroudeHead(len(fh_ck["channels"])).to(device).eval()
        froude_head.load_state_dict(fh_ck["heads"]["proj"])   # control-relevant: candidates only have an action
        for p in froude_head.parameters():
            p.requires_grad_(False)
        print(f"loaded ftm_froude scorer from {args.ftm_froude}")

    names = tuple(s.split("=", 1)[0] for s in cfg.sources) or ("default",)
    # `state` may be absent -- checkpoints trained after L_state's retirement (2026-09-08,
    # stop-gradient fix on L_body) never build it. Skip that scorer rather than KeyError, so this
    # same script/protocol still runs the comparable scorers (direct/ridge/f179) on those.
    state_model = None
    if "state" in ck:
        state_model = StateHead(cfg, len(channels), names,
                                use_delta=getattr(cfg, "state_use_delta", True)).to(device).eval()
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
    # per-transition body_motion (NOT the whole-clip average) -- feeds the "oracle" scorer, which
    # asks: even with the true achieved motion at this exact instant (no model, no prediction),
    # does gait-phase noise alone cap ranking below 100% against the whole-clip-average ground
    # truth? Same logic as F190's per-transition vs clip-averaged oracle finding (39% vs 83.8%).
    bm_full = {c: np.asarray(load(cand[c], REGISTRY["hexapod"])["body_motion"])[:, channels]
              for c in conds}
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

    scorers = ("f179", "direct", "ridge", "oracle") + (("state",) if state_model is not None else ()) \
        + (("rssm",) if rssm is not None else ()) + (("ftm_froude",) if froude_head is not None else ())
    picks = {s: [] for s in scorers}
    true_ranks = {c: r for r, c in enumerate(sorted(conds, key=lambda c: true_dist[c]))}

    for bt in branch:
        frames, _a, _f, _h, _o = run_to(int(bt))
        obs = frames[int(bt) - 1]
        with torch.no_grad():
            e_full = encode_clip(encoder, np.asarray(obs)[None], 1).float().to(device)
            a = torch.stack([acts[c][min(int(bt), len(acts[c]) - 1)] for c in conds]).to(device)
            z = proj(a, "hexapod")

            # conditions in chunks of 4, not all 12 at once: this card is only 10.57GB and
            # fitted["E"] alone holds ~5.4GB of it resident for the whole rollout, so the full
            # 12-way batched itm/ftm attention call leaves too little headroom (observed failing
            # right at the edge, ~200MB short). Chunking the batch dim doesn't change results --
            # itm/ftm/state_model have no cross-batch-element interaction -- only peak memory.
            CHUNK = 4
            f179_parts, state_parts, froude_parts = [], [], []
            for i in range(0, len(conds), CHUNK):
                a_c, z_c = a[i:i + CHUNK], z[i:i + CHUNK]
                n_c = a_c.shape[0]
                roll_c = e_full.expand(n_c, -1, -1)
                for _ in range(args.horizon):
                    roll_c = ftm(roll_c, z_c)
                f179_parts.append(md.body(None, itm(e_full.expand(n_c, -1, -1), roll_c)))
                if state_model is not None or froude_head is not None:
                    one_step_c = ftm(e_full.expand(n_c, -1, -1), z_c)
                    if state_model is not None:
                        state_parts.append(state_model(one_step_c - e_full.expand(n_c, -1, -1), z_c,
                                                       "hexapod"))
                    if froude_head is not None:
                        # THE UNTESTED CELL: FTM's own single-step predicted embedding, pooled,
                        # read by the clean stop-gradient-trained Froude head -- no delta, no z_proj.
                        froude_parts.append(froude_head(one_step_c.mean(1)))
            m_f179 = torch.cat(f179_parts, dim=0)
            m_direct = md.body(None, z)
            m_ridge = score_ridge(fitted, e_full[0], z, device)
            # ORACLE: no model at all -- each candidate's TRUE recorded body_motion at this exact
            # branch timestep (standardised the same way), not the model's prediction. Tests
            # whether gait-phase noise alone caps ranking below 100% at this test's resolution.
            m_oracle_np = np.stack([(bm_full[c][min(int(bt), len(bm_full[c]) - 1)] - mean_s) / std_s
                                    for c in conds])
            m_oracle = torch.tensor(m_oracle_np, dtype=torch.float32, device=device)
            scored = [("f179", m_f179), ("direct", m_direct), ("ridge", m_ridge),
                     ("oracle", m_oracle)]
            if state_model is not None:
                scored.append(("state", torch.cat(state_parts, dim=0)))

            if rssm is not None:
                # build h at the branch point by teacher-forcing the REAL posterior over frames
                # 0..bt-1 with real actions 0..bt-1 (same per-step pairing convention training
                # used), THEN roll the PRIOR forward --horizon steps per candidate action --
                # exactly f179's rollout structure, using the RSSM's prior instead of the FTM.
                prefix = encode_clip(encoder, np.asarray(frames[:int(bt)]), 2).float().to(device)
                prefix_pooled = prefix.mean(1)                                   # [bt, 1408]
                a_prefix = torch.as_tensor(seed[:int(bt)], dtype=torch.float32, device=device)
                h = torch.zeros(1, RSSM_H_DIM, device=device)
                z = torch.zeros(1, RSSM_Z_DIM, device=device)
                for t in range(prefix_pooled.shape[0]):
                    h, z = rssm.step_posterior(h, z, a_prefix[t:t + 1], prefix_pooled[t:t + 1])
                r_mean_s = torch.tensor(rssm_mean_s, dtype=torch.float32, device=device)
                r_std_s = torch.tensor(rssm_std_s, dtype=torch.float32, device=device)
                rssm_preds = []
                for c in conds:
                    a_c = acts[c][min(int(bt), len(acts[c]) - 1):min(int(bt), len(acts[c]) - 1) + 1].to(device)
                    hc, zc = h.clone(), z.clone()
                    for _ in range(args.horizon):
                        hc, zc, pred = rssm.step_prior(hc, zc, a_c)
                    rssm_preds.append(pred[0])
                m_rssm_raw = torch.stack(rssm_preds)               # standardised on RSSM's own stats
                # rescore in the SAME standardised space condition_confusion.py uses (mean_s/std_s)
                m_rssm = (m_rssm_raw * r_std_s + r_mean_s - torch.tensor(mean_s, dtype=torch.float32,
                         device=device)) / torch.tensor(std_s, dtype=torch.float32, device=device)
                scored.append(("rssm", m_rssm))

            if froude_head is not None:
                # already in the SAME standardised space as mean_s/std_s -- ftm_froude_stopgrad_probe.py
                # used the identical checkpoint (teacher_stopgrad.pt)'s body_stats, no rescoring needed
                scored.append(("ftm_froude", torch.cat(froude_parts, dim=0)))

            for name, m in scored:
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
