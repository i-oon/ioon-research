"""Why does the FTM lose yaw in the counterfactual read-out (F308)? Dump stage.

Per group (24 branches, one per command, same state), averaged over P consecutive pairs b..b+P-1 (stride K):
  truth            mean CoM Froude over [b, b+P-1+K)
  F_real           body(ITM(e0, e1))                       real future
  F_direct         body(proj(chunk))
  F_ftm_proj       body(ITM(e0, FTM(e0, proj(chunk))))      as in counterfactual_readout.py
  F_ftm_true       body(ITM(e0, FTM(e0, ITM(e0, e1))))      FTM fed the latent it is trained on (isolates the FTM)
  z_real, z_proj   latents
  pooled deltas    (e1 - e0), (FTM_proj - e0), (FTM_true - e0), avg-pooled to a 4x4 token grid (16 x 1408)
  spread stats     on full tokens, centred across the 24 commands: total variance, and share of it linearly
                   explained by each truth channel, for real e1 / FTM_proj / FTM_true; cosine between the
                   centred prediction and the centred real future per command.
One group at a time; nothing but the small arrays kept. Output: results/check/ftm_yaw/<tag>.npz

    .venv/bin/python3 scripts/diagnostics/forward_model/ftm_yaw_dump.py --embodiment b1 \
        --cf_dir data/counterfactual_walks/b1_branches_heldout --ckpt B=results/eval/round1_branches_s0/ckpt/b1.pt \
        --groups 24 --P 11 --tag b1
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from counterfactual_readout import groups_of  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at  # noqa: E402


def pool4(d):  # (n, 256, C) -> (n, 16*C)
    n, T, C = d.shape
    g = int(round(T ** 0.5))
    x = d.reshape(n, g, g, C).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(x, 4).permute(0, 2, 3, 1).reshape(n, -1)


def spread(X, Y):  # X (24, D) centred future, Y (24, 3) truth -> total var, R2 share per channel
    Xc = X - X.mean(0, keepdim=True)
    tot = Xc.pow(2).sum().item()
    out = []
    for j in range(3):
        y = Y[:, j] - Y[:, j].mean()
        y = y / (y.norm() + 1e-12)
        out.append((y @ Xc).pow(2).sum().item() / (tot + 1e-12))
    return tot, out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", required=True, choices=("hexapod", "b1"))
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--P", type=int, default=11)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, emb = a.device, a.embodiment
    G = sorted(groups_of(a.cf_dir).items())
    idx = np.linspace(0, len(G) - 1, min(a.groups, len(G))).round().astype(int)
    G = [G[i] for i in idx]
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
    runs = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), emb, 18 if emb == "hexapod" else 12, dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        del ck
        off = None if m.offset is None else m.offset.float().to(dev)
        runs.append(dict(name=name, m=m, mean=mean, std=std, off=off, out={}))
    K = max(r["m"].stride for r in runs)
    P = a.P
    names, truths = [], []
    for gi, (key, paths) in enumerate(G):
        clips = [load(p, REGISTRY[emb], lazy_frames=True) for p in paths]
        order = np.argsort([int(np.load(p)["cf_command_index"]) for p in paths])
        clips = [clips[i] for i in order]
        paths = [paths[i] for i in order]
        names.append([str(np.load(p)["cf_command_name"]) for p in paths])
        bs = [int(c["first_pair"]) for c in clips]
        E = torch.stack([encode_clip(enc, np.asarray(c["frames"][b:b + P + K]), 8) for c, b in zip(clips, bs)])  # (24, P+K, T, C)
        truth = np.stack([np.asarray(c["body_motion"])[b:b + P - 1 + K, :3].mean(0) for c, b in zip(clips, bs)])
        truths.append(truth)
        for r in runs:
            m = r["m"]
            fix = (lambda e: e) if r["off"] is None else (lambda e, o=r["off"]: e - o.reshape(e.shape[-2:]))
            rd = lambda z: m.md.body(None, z).float().cpu().numpy() * r["std"] + r["mean"]  # noqa: E731
            acc = {k: [] for k in ("F_real", "F_direct", "F_ftm_proj", "F_ftm_true", "z_real", "z_proj")}
            fut = {k: 0 for k in ("real", "proj", "true")}
            pool = {k: 0 for k in ("real", "proj", "true")}
            for j in range(P):
                e0 = fix(E[:, j].to(dev)); e1 = fix(E[:, j + K].to(dev))
                ch = np.stack([action_chunk_at(c["actions"], b + j + m.action_lag, K) for c, b in zip(clips, bs)])
                zp = m.proj(torch.as_tensor(ch, device=dev), emb)
                zr = m.itm(e0, e1)
                fp = m.ftm_step(e0, zp); ft = m.ftm_step(e0, zr)
                acc["F_real"].append(rd(zr)); acc["F_direct"].append(rd(zp))
                acc["F_ftm_proj"].append(rd(m.itm(e0, fp))); acc["F_ftm_true"].append(rd(m.itm(e0, ft)))
                acc["z_real"].append(zr.float().cpu().numpy()); acc["z_proj"].append(zp.float().cpu().numpy())
                for k, x in (("real", e1), ("proj", fp), ("true", ft)):
                    fut[k] = fut[k] + x.float() / P
                    pool[k] = pool[k] + pool4((x - e0).float()).cpu() / P
            Y = torch.as_tensor(truth, dtype=torch.float32, device=dev)
            o = r["out"]
            for k, v in acc.items():
                o.setdefault(k, []).append(np.mean(v, 0))
            for k in fut:
                tot, sh = spread(fut[k].flatten(1), Y)
                o.setdefault("var_" + k, []).append(tot); o.setdefault("share_" + k, []).append(sh)
                o.setdefault("pool_" + k, []).append(pool[k].half().numpy())
            rc = fut["real"].flatten(1); rc = rc - rc.mean(0, keepdim=True)
            for k in ("proj", "true"):
                pc = fut[k].flatten(1); pc = pc - pc.mean(0, keepdim=True)
                o.setdefault("cos_" + k, []).append(F.cosine_similarity(pc, rc, dim=1).cpu().numpy())
                o.setdefault("norm_" + k, []).append((pc.norm(dim=1) / rc.norm(dim=1)).cpu().numpy())
        del clips, E
        print(f"{gi + 1}/{len(G)} groups", flush=True)
    for r in runs:
        np.savez_compressed(os.path.join(ROOT, "results/check/ftm_yaw", f"{a.tag}_{r['name']}.npz"),
                            truth=np.stack(truths), names=np.array(names), K=K, P=P,
                            **{k: np.stack(v) for k, v in r["out"].items()})


if __name__ == "__main__":
    main()
