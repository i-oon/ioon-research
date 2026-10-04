"""Does the FTM-predicted read-out recover yaw when the projector's z is brought onto the z distribution
the FTM was trained on? (z_proj's within-group spread is 4-8x z_real's on branches; ftm_yaw_dump.py.)

Variants of the z fed to FTM(e0, .), read by body(ITM(e0, FTM)), P-pair window as counterfactual_readout:
  proj         z_proj (baseline, = 'FTM cf')
  true         ITM(e0, e1)
  shrink0.5/0.3  group mean of z_proj + s * (z_proj - group mean)
  cal          affine map z_proj -> z_real, ridge-fitted on the EVEN groups of <tag>_<name>.npz (P-averaged z),
               evaluated here on ODD groups only (held out from the fit)
  cal_mean     cal, but only its group-mean part kept... (not used) -- see doc
Output: results/check/ftm_yaw/zfix_<tag>_<name>.txt
"""
import argparse, os, sys
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models, corr  # noqa: E402
from counterfactual_readout import groups_of  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.policy.planner import action_chunk_at  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--P", type=int, default=11)
    a = ap.parse_args()
    dev, emb, P = "cuda", a.embodiment, a.P
    G = sorted(groups_of(a.cf_dir).items())
    idx = np.linspace(0, len(G) - 1, 72).round().astype(int)       # same selection as the dump (72 groups)
    odd = idx[1::2][np.linspace(0, 35, min(a.groups, 36)).round().astype(int)]
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
    runs = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), emb, 18 if emb == "hexapod" else 12, dev)
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        del ck
        d = np.load(os.path.join(ROOT, "results/check/ftm_yaw", f"{a.tag}_{name}.npz"))
        zp, zr = d["z_proj"][0::2].reshape(-1, d["z_proj"].shape[-1]), d["z_real"][0::2].reshape(-1, d["z_real"].shape[-1])
        X = np.c_[zp, np.ones(len(zp))]
        W = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ zr)
        off = None if m.offset is None else m.offset.float().to(dev)
        runs.append(dict(name=name, m=m, mean=mean, std=std, off=off, W=torch.as_tensor(W, dtype=torch.float32, device=dev),
                         R={}))
    K = max(r["m"].stride for r in runs)
    for gi, gidx in enumerate(odd):
        key, paths = G[gidx]
        clips = [load(p, REGISTRY[emb], lazy_frames=True) for p in paths]
        bs = [int(c["first_pair"]) for c in clips]
        E = torch.stack([encode_clip(enc, np.asarray(c["frames"][b:b + P + K]), 8) for c, b in zip(clips, bs)])
        truth = np.stack([np.asarray(c["body_motion"])[b:b + P - 1 + K, :3].mean(0) for c, b in zip(clips, bs)])
        for r in runs:
            m = r["m"]
            fix = (lambda e: e) if r["off"] is None else (lambda e, o=r["off"]: e - o.reshape(e.shape[-2:]))
            rd = lambda z: m.md.body(None, z).float().cpu().numpy() * r["std"] + r["mean"]  # noqa: E731
            acc = {}
            for j in range(P):
                e0 = fix(E[:, j].to(dev)); e1 = fix(E[:, j + K].to(dev))
                ch = np.stack([action_chunk_at(c["actions"], b + j + m.action_lag, K) for c, b in zip(clips, bs)])
                zp = m.proj(torch.as_tensor(ch, device=dev), emb)
                zr = m.itm(e0, e1)
                mu = zp.mean(0, keepdim=True)
                zc = torch.cat([zp, torch.ones(len(zp), 1, device=dev)], 1) @ r["W"]
                Z = {"proj": zp, "true": zr, "shrink0.5": mu + 0.5 * (zp - mu), "shrink0.3": mu + 0.3 * (zp - mu),
                     "cal": zc, "cal_shrink0.5": zc.mean(0, keepdim=True) + 0.5 * (zc - zc.mean(0, keepdim=True)),
                     "proj_dev+real_mean": zp - mu + zr.mean(0, keepdim=True)}
                for k, z in Z.items():
                    acc.setdefault("ftm " + k, []).append(rd(m.itm(e0, m.ftm_step(e0, z))))
                    acc.setdefault("direct " + k, []).append(rd(z))
            for k, v in acc.items():
                f = np.mean(v, 0)
                r["R"].setdefault(k, []).append([corr(f[:, c], truth[:, c]) for c in range(3)])
        del clips, E
        print(f"{gi + 1}/{len(odd)}", flush=True)
    for r in runs:
        lines = [f"== {a.tag} {r['name']}: {len(odd)} held-out (odd) groups, P {P}; within-group r fwd / lat / yaw"]
        for k, v in r["R"].items():
            lines.append(f"  {k:<26}" + " / ".join(f"{x:.2f}" for x in np.nanmean(v, 0)))
        txt = "\n".join(lines); print(txt)
        open(os.path.join(ROOT, "results/check/ftm_yaw", f"zfix_{a.tag}_{r['name']}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
