"""Which direction of z does the FTM-predicted read use for each channel, and does the projector get it right?

For each pair: g_c = d body_c(ITM(e0, FTM(e0, z))) / dz at z = z_true (= ITM(e0, e1)), group-mean over the 24
commands; h_c = d body_c(z) / dz (the body head's own direction, = what 'direct' reads). Reported per channel:
  cos(g_c, h_c)                         does the FTM chain read channel c along the body head's direction
  r(z_true.g, truth), r(z_proj.g, truth)  the FTM-relevant coordinate in the real vs the projector latent
  r(z_proj.g, z_true.g)                 does the projector reproduce that coordinate across commands
  r(z_proj.h, truth)                    the body-head coordinate in the projector latent ('direct')
  std(z_proj.g) / std(z_true.g)
Within-group (24 commands) Pearson r, P-pair averaged, mean over groups. Output zgrad_<tag>_<name>.txt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", required=True)
    ap.add_argument("--cf_dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--P", type=int, default=11)
    a = ap.parse_args()
    dev, emb, P = "cuda", a.embodiment, a.P
    G = sorted(groups_of(a.cf_dir).items())
    G = [G[i] for i in np.linspace(0, len(G) - 1, a.groups).round().astype(int)]
    from vjepa2_encoder import VJEPA2FrameEncoder
    enc = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
    runs = []
    for spec in a.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), emb, 18 if emb == "hexapod" else 12, dev)
        for mod in (m.itm, m.ftm, m.md):
            for q in mod.parameters():
                q.requires_grad_(False)
        off = None if m.offset is None else m.offset.float().to(dev)
        runs.append(dict(name=name, m=m, off=off, R=[]))
    K = max(r["m"].stride for r in runs)
    for gi, (key, paths) in enumerate(G):
        clips = [load(p, REGISTRY[emb], lazy_frames=True) for p in paths]
        bs = [int(c["first_pair"]) for c in clips]
        with torch.no_grad():
            E = torch.stack([encode_clip(enc, np.asarray(c["frames"][b:b + P + K]), 8) for c, b in zip(clips, bs)])
        truth = np.stack([np.asarray(c["body_motion"])[b:b + P - 1 + K, :3].mean(0) for c, b in zip(clips, bs)])
        for r in runs:
            m = r["m"]
            fix = (lambda e: e) if r["off"] is None else (lambda e, o=r["off"]: e - o.reshape(e.shape[-2:]))
            sg_t, sg_p, sh_t, sh_p, cosgh = 0, 0, 0, 0, []
            hyb = {k: 0 for k in ("proj", "proj+yaw_dir", "proj+fwd_dir", "proj+rand_dir")}
            for j in range(P):
                with torch.no_grad():
                    e0 = fix(E[:, j].to(dev)); e1 = fix(E[:, j + K].to(dev))
                    ch = np.stack([action_chunk_at(c["actions"], b + j + m.action_lag, K) for c, b in zip(clips, bs)])
                    zp = m.proj(torch.as_tensor(ch, device=dev), emb)
                    zt = m.itm(e0, e1)
                g, h = [], []
                for c in range(3):
                    z = zt.clone().requires_grad_(True)
                    m.md.body(None, m.itm(e0, m.ftm_step(e0, z)))[:, c].sum().backward()
                    g.append(z.grad.mean(0))
                    z2 = zt.clone().requires_grad_(True)
                    m.md.body(None, z2)[:, c].sum().backward()
                    h.append(z2.grad.mean(0))
                g, h = torch.stack(g), torch.stack(h)                          # (3, zdim)
                cosgh.append(torch.nn.functional.cosine_similarity(g, h, dim=1).cpu().numpy())
                with torch.no_grad():
                    sg_t = sg_t + (zt @ g.T).cpu().numpy() / P; sg_p = sg_p + (zp @ g.T).cpu().numpy() / P
                    sh_t = sh_t + (zt @ h.T).cpu().numpy() / P; sh_p = sh_p + (zp @ h.T).cpu().numpy() / P
                    # rank-1 oracle: replace z_proj's coordinate along one direction by z_true's
                    gen = torch.Generator(device="cpu").manual_seed(j)
                    dirs = {"proj": None, "proj+yaw_dir": g[2], "proj+fwd_dir": g[0],
                            "proj+rand_dir": torch.randn(g.shape[1], generator=gen).to(dev)}
                    for k, u in dirs.items():
                        zz = zp if u is None else zp + torch.outer((zt - zp) @ (u / u.norm()), u / u.norm())
                        hyb[k] = hyb[k] + m.md.body(None, m.itm(e0, m.ftm_step(e0, zz))).cpu().numpy() / P
            row = []
            for c in range(3):
                row.append([np.mean(cosgh, 0)[c], corr(sg_t[:, c], truth[:, c]), corr(sg_p[:, c], truth[:, c]),
                            corr(sg_p[:, c], sg_t[:, c]), corr(sh_p[:, c], truth[:, c]),
                            sg_p[:, c].std() / (sg_t[:, c].std() + 1e-12)])
            r["R"].append(row)
            r.setdefault("H", []).append([[corr(v[:, c], truth[:, c]) for c in range(3)] for v in hyb.values()])
            r["hk"] = list(hyb)
        del clips, E
        print(f"{gi + 1}/{len(G)}", flush=True)
    hdr = ["cos(g,h)", "r(zt.g,y)", "r(zp.g,y)", "r(zp.g,zt.g)", "r(zp.h,y)", "std zp.g/zt.g"]
    for r in runs:
        R = np.nanmean(np.asarray(r["R"], dtype=float), 0)
        lines = [f"== {a.tag} {r['name']}: {len(G)} groups, P {P}", f"{'':<5}" + "".join(f"{x:>15}" for x in hdr)]
        for c, nm in enumerate(("fwd", "lat", "yaw")):
            lines.append(f"{nm:<5}" + "".join(f"{x:>15.2f}" for x in R[c]))
        Hm = np.nanmean(np.asarray(r["H"], dtype=float), 0)
        lines.append("FTM read r fwd / lat / yaw with z_proj, and z_proj with ONE coordinate (along g_c) set to z_true's:")
        for k, v in zip(r["hk"], Hm):
            lines.append(f"  {k:<15}" + " / ".join(f"{x:.2f}" for x in v))
        txt = "\n".join(lines); print(txt)
        open(os.path.join(ROOT, "results/check/ftm_yaw", f"zgrad_{a.tag}_{r['name']}.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
