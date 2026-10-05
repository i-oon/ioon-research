"""DIAGNOSTIC (F311 follow-up): does a STATE-AWARE projector z = P(a, e_t) recover FTM-predicted yaw on heldout branches?

Two projectors, trained identically (fit_projector's z-fit, then its rollout objective: read of proj z matched to read of
true z, channels standardised, + w_z * z-MSE), same seed / epochs, on c10 + B1 {branches_train subset, clips_train}:
  free   ActionProjector-shaped MLP on the standardised action chunk
  aware  same MLP on [action chunk, s(e_t)], s = shared per-cell linear (1408 -> 32) on 4x4-pooled tokens -> 512 -> 128
Clip-level split (branches grouped by cf_source). Evaluation = counterfactual_readout logic on {b1,c10}_branches_heldout
(group-wise r across the 24 commands, P = 1 / 11, mean over groups), tokens from the shared cf_tokens cache.

    .venv/bin/python3 scripts/diagnostics/forward_model/state_projector_diag.py [--n_branch 400]
Outputs results/check/state_projector/ (per-pair e_t is streamed to a raw fp16 file there and deleted at the end).
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from counterfactual_readout import groups_of, tokens  # noqa: E402
from rollout_state_action_anova import corr  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.data.strided import action_chunks, first_pair_of, stride_of  # noqa: E402
from wm.evaluate import offset_for  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from wm.policy.planner import action_chunk_at  # noqa: E402

OUT = os.path.join(ROOT, "results/check/state_projector")
CW = "data/counterfactual_walks"
BODIES = (("hexapod", "c10", 18), ("b1", "b1", 12))
NTOK, D = 256, 1408


def pool4(e):                      # (N, 256, 1408) -> (N, 16, 1408): 16x16 grid -> 4x4 mean
    return e.reshape(-1, 4, 4, 4, 4, D).mean((2, 4)).reshape(-1, 16, D)


class Proj(nn.Module):
    def __init__(self, dims, chunk, z_dim, width=512, state=False):
        super().__init__()
        self.chunk, self.state = chunk, state
        sd = 128 if state else 0
        self.nets = nn.ModuleDict({n: nn.Sequential(nn.Linear(d * chunk + sd, width), nn.GELU(), nn.Linear(width, width),
                                                    nn.GELU(), nn.Linear(width, z_dim)) for n, d in dims.items()})
        if state:
            self.cell = nn.Linear(D, 32)
            self.smap = nn.Sequential(nn.GELU(), nn.Linear(16 * 32, 128))
            self.register_buffer("pmean", torch.zeros(D)); self.register_buffer("pstd", torch.ones(D))
        for n, d in dims.items():
            self.register_buffer(f"mean_{n}", torch.zeros(d)); self.register_buffer(f"std_{n}", torch.ones(d))

    def forward(self, a, name, pooled=None):
        a = ((a - getattr(self, f"mean_{name}")) / getattr(self, f"std_{name}")).flatten(-2)
        if self.state:
            s = self.cell((pooled.float() - self.pmean) / self.pstd).flatten(-2)
            a = torch.cat([a, self.smap(s)], -1)
        return self.nets[name](a)


def load_models(dev):
    ck = torch.load(os.path.join(ROOT, "wm/runs/round1_branches_s0/best.pt"), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(dev).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(dev).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {n: d for n, _, d in BODIES}).to(dev).eval(); md.load_state_dict(ck["md"], strict=False)
    for m in (itm, ftm, md):
        for p in m.parameters():
            p.requires_grad_(False)
    mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
    offs = {n: offset_for(ck, n) for n, _, _ in BODIES}
    return cfg, itm, ftm, md, mean, std, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_branch", type=int, default=400)
    ap.add_argument("--n_clips", type=int, default=0, help="0 = all clips_train")
    ap.add_argument("--zfit_epochs", type=int, default=200)
    ap.add_argument("--rollout_epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval_groups", type=int, default=0, help="0 = all heldout groups")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--keep_bin", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    cfg, itm, ftm, md, bmean, bstd, offs = load_models(dev)
    K, lag = stride_of(cfg), max(1, cfg.action_lag)
    held = {}

    def encoder():
        if "enc" not in held:
            from vjepa2_encoder import VJEPA2FrameEncoder
            held["enc"] = VJEPA2FrameEncoder(dtype=torch.float32, device=dev)
        return held["enc"]

    def read(e, z, name):
        return md.body(None, itm(e, ftm(e, z, name)))

    # ---------------- gather: e_t streamed to disk, pooled e_t / z / action / target read in RAM ----------------
    binf = os.path.join(OUT, f"et_{a.tag}.bin")
    fh = open(binf, "wb")
    rows = 0
    data = {}
    for name, short, _ in BODIES:
        rng = np.random.default_rng(0)
        br = sorted(glob.glob(os.path.join(ROOT, CW, f"{short}_branches_train", "*.npz")))
        br = sorted(rng.choice(br, a.n_branch, replace=False).tolist()) if a.n_branch < len(br) else br
        cl = sorted(glob.glob(os.path.join(ROOT, CW, f"{short}_clips_train", "*.npz")))
        if a.n_clips:
            cl = cl[:a.n_clips]
        R, PL, Z, A, T, C, KIND = [], [], [], [], [], [], []
        cid = {}
        off = offs[name]
        for i, path in enumerate(br + cl):
            clip = load(path, REGISTRY[name], lazy_frames=True)
            s0 = first_pair_of(clip)
            nf = len(clip["body_motion"])
            e = tokens(path, clip, s0, nf - s0, encoder).float().to(dev)      # frames s0 ..
            if off is not None:
                e = e - off.to(dev).reshape(e.shape[1:])
            n = len(e) - K
            if n <= 0 or len(clip["actions"]) < s0 + n + lag + K - 1:
                continue
            act = torch.as_tensor(action_chunks(clip["actions"], lag, K, n, start=s0), dtype=torch.float32)
            with torch.no_grad():
                z = torch.cat([itm(e[t:min(t + 16, n)], e[t + K:min(t + 16, n) + K]) for t in range(0, n, 16)])
                tg = torch.cat([read(e[t:min(t + 16, n)], z[t:min(t + 16, n)], name) for t in range(0, n, 16)])
            et = e[:n].half().cpu()
            fh.write(et.numpy().tobytes())
            R.append(torch.arange(rows, rows + n)); rows += n
            PL.append(pool4(et.float()).half()); Z.append(z.cpu()); A.append(act); T.append(tg.cpu())
            src = str(np.load(path, allow_pickle=True)["cf_source"]) if path in br else path
            C.append(torch.full((n,), cid.setdefault(src, len(cid)))); KIND.append(torch.full((n,), int(path in br)))
            del e
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(br) + len(cl)} files, {rows} pairs", flush=True)
        data[name] = dict(row=torch.cat(R), pool=torch.cat(PL), z=torch.cat(Z), a=torch.cat(A), t=torch.cat(T),
                          c=torch.cat(C), kind=torch.cat(KIND))
        print(f"{name}: {len(br)} branch + {len(cl)} clip files, {len(data[name]['z'])} pairs, {len(cid)} clips/sources",
              flush=True)
    fh.close()
    held.clear(); torch.cuda.empty_cache()
    fd = os.open(binf, os.O_RDONLY)
    rowbytes = NTOK * D * 2

    def fetch(rows_):
        buf = np.empty((len(rows_), NTOK, D), np.float16)
        for j, r in enumerate(rows_.tolist()):
            buf[j] = np.frombuffer(os.pread(fd, rowbytes, r * rowbytes), np.float16).reshape(NTOK, D)
        return torch.from_numpy(buf).to(dev).float()

    # clip-level split (branches grouped by source clip), 20 % val, seed 0
    for name, d in data.items():
        ids = torch.unique(d["c"])
        val = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(0))[:max(1, int(0.2 * len(ids)))]]
        d["val"] = torch.isin(d["c"], val)
        print(f"{name}: {int(d['val'].sum())} / {len(d['val'])} pairs val", flush=True)

    s1 = torch.zeros(D, dtype=torch.float64); s2 = torch.zeros(D, dtype=torch.float64); cnt = 0
    for d in data.values():
        tp = d["pool"][~d["val"]]
        for i in range(0, len(tp), 1024):
            x = tp[i:i + 1024].double().reshape(-1, D)
            s1 += x.sum(0); s2 += (x ** 2).sum(0); cnt += len(x)
    pm = (s1 / cnt).float(); ps = (s2 / cnt - (s1 / cnt) ** 2).clamp_min(1e-12).sqrt().float().clamp_min(1e-6)
    scale = torch.cat([d["t"][~d["val"]] for d in data.values()]).std(0).clamp_min(1e-6).to(dev)
    print(f"read-of-true-z std {scale.tolist()}", flush=True)

    dims = {n: dd for n, _, dd in BODIES}
    projs = {}
    for kind in ("free", "aware"):
        torch.manual_seed(0)
        p = Proj(dims, K, cfg.z_dim, cfg.hidden, state=(kind == "aware")).to(dev)
        for name, d in data.items():
            pj = d["a"].reshape(-1, d["a"].shape[-1])
            getattr(p, f"mean_{name}").copy_(pj.mean(0)); getattr(p, f"std_{name}").copy_(pj.std(0).clamp_min(1e-6))
        if p.state:
            p.pmean.copy_(pm); p.pstd.copy_(ps)
        # phase 1: z fit (full batch, as fit_projector)
        G = {n: {k: d[k][~d["val"]].to(dev) for k in ("pool", "z", "a")} for n, d in data.items()}
        opt = torch.optim.Adam(p.parameters(), lr=1e-3)
        for ep in range(a.zfit_epochs):
            opt.zero_grad(); loss = 0.0
            for n, g in G.items():
                loss = loss + F.mse_loss(p(g["a"], n, g["pool"]), g["z"])
            loss.backward(); opt.step()
        print(f"[{kind}] z fit done, train z-MSE {loss.item():.4f}", flush=True)
        del G; torch.cuda.empty_cache()

        def errs(mask_fn):
            out = {}
            p.eval()
            with torch.no_grad():
                for n, d in data.items():
                    idx = torch.nonzero(mask_fn(d), as_tuple=True)[0]
                    se = torch.zeros(3, device=dev); zse = 0.0
                    for i in range(0, len(idx), 64):
                        sl = idx[i:i + 64]
                        zp = p(d["a"][sl].to(dev), n, d["pool"][sl].to(dev))
                        r = read(fetch(d["row"][sl]), zp, n)
                        se += (((r - d["t"][sl].to(dev)) / scale) ** 2).sum(0)
                        zse += ((zp - d["z"][sl].to(dev)) ** 2).mean(1).sum().item()
                    out[n] = [round(x, 4) for x in (se / len(idx)).tolist()] + [round(zse / len(idx), 4)]
            return out

        # phase 2: rollout objective (fit_projector.rollout_fit)
        opt = torch.optim.Adam(p.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(0)
        items = [(n, i) for n, d in data.items() for i in torch.nonzero(~d["val"], as_tuple=True)[0].tolist()]
        for ep in range(a.rollout_epochs):
            p.train()
            order = torch.randperm(len(items), generator=gen).tolist()
            tot = nb = 0
            for b in range(0, len(order), a.batch):
                batch = [items[j] for j in order[b:b + a.batch]]
                opt.zero_grad(); loss = 0.0
                for n, d in data.items():
                    sl = torch.tensor([i for nm, i in batch if nm == n], dtype=torch.long)
                    if not len(sl):
                        continue
                    zp = p(d["a"][sl].to(dev), n, d["pool"][sl].to(dev))
                    r = read(fetch(d["row"][sl]), zp, n)
                    w = len(sl) / len(batch)
                    loss = loss + (((r - d["t"][sl].to(dev)) / scale) ** 2).mean() * w \
                        + 0.1 * F.mse_loss(zp, d["z"][sl].to(dev)) * w
                loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            print(f"[{kind}] rollout epoch {ep + 1:2d} train {tot / nb:.4f}", flush=True)
        e_tr = errs(lambda d: ~d["val"] & (torch.arange(len(d["val"])) % 4 == 0))
        e_va = errs(lambda d: d["val"])
        print(f"[{kind}] read err (std. fwd/lat/yaw, z-MSE) train-subset {e_tr}  val {e_va}", flush=True)
        p.eval()
        projs[kind] = (p, e_tr, e_va)
        torch.save(p.state_dict(), os.path.join(OUT, f"proj_{kind}_{a.tag}.pt"))
    os.close(fd)
    if not a.keep_bin:
        os.remove(binf)
    for d in data.values():
        d.clear()

    # ---------------- evaluation: counterfactual_readout logic on heldout branches ----------------
    res = {}
    with torch.no_grad():
        for name, short, _ in BODIES:
            Gs = sorted(groups_of(f"{CW}/{short}_branches_heldout").items())
            if a.eval_groups:
                Gs = Gs[:a.eval_groups]
            off = offs[name]
            Pmax = 11
            reads = ("real", "ftm_true", "ftm_free", "ftm_aware", "dir_true", "dir_free", "dir_aware")
            Rr = {P: {k: [] for k in reads} for P in (1, 11)}
            zs = collections.defaultdict(lambda: collections.defaultdict(list))   # cmd -> kind -> [z]
            hrd = []      # standardised read error on heldout pair 0 (for the memorisation check)
            for gi, (key, paths) in enumerate(Gs):
                clips = [load(pp, REGISTRY[name], lazy_frames=True) for pp in paths]
                bs = [int(c["first_pair"]) for c in clips]
                E = [tokens(pp, c, bp, Pmax + K, encoder) for pp, c, bp in zip(paths, clips, bs)]
                for P in (1, 11):
                    truth, rd = [], {k: [] for k in reads}
                    for pp, c, bp, e in zip(paths, clips, bs, E):
                        e = e.float().to(dev)
                        if off is not None:
                            e = e - off.to(dev).reshape(e.shape[1:])
                        truth.append(np.asarray(c["body_motion"])[bp:bp + P - 1 + K, :3].mean(0))
                        e0, e1 = e[:P], e[K:K + P]
                        ch = torch.as_tensor(np.stack([action_chunk_at(c["actions"], bp + j + lag, K) for j in range(P)]),
                                             dtype=torch.float32, device=dev)
                        pl = pool4(e0)
                        zz = {"true": itm(e0, e1), "free": projs["free"][0](ch, name, pl),
                              "aware": projs["aware"][0](ch, name, pl)}
                        f = lambda z: (md.body(None, z).float().cpu().numpy() * bstd + bmean).mean(0)  # noqa: E731
                        rd["real"].append(f(zz["true"]))
                        for k2, z in zz.items():
                            rd[f"ftm_{k2}"].append(f(itm(e0, ftm(e0, z, name))))
                            rd[f"dir_{k2}"].append(f(z))
                        if P == 1:
                            with np.load(pp, allow_pickle=True) as nz:
                                ci = int(nz["cf_command_index"])
                            for k2, z in zz.items():
                                zs[ci][k2].append(z[0].cpu().numpy())
                            tr = md.body(None, itm(e0, ftm(e0, zz["true"], name)))
                            hrd.append(torch.stack([((md.body(None, itm(e0, ftm(e0, zz[k3], name))) - tr) / scale) ** 2
                                                    for k3 in ("free", "aware")])[:, 0].cpu().numpy())
                    truth = np.stack(truth)
                    for k2 in reads:
                        fr = np.stack(rd[k2])
                        Rr[P][k2].append([corr(fr[:, j], truth[:, j]) for j in range(3)])
                del clips, E
                if (gi + 1) % 12 == 0:
                    print(f"  {name} eval {gi + 1}/{len(Gs)} groups", flush=True)
            hr = np.stack(hrd).mean(0)
            spread = {k2: float(np.mean([np.stack(v[k2]).std(0).mean() for v in zs.values() if len(v[k2]) > 1]))
                      for k2 in ("true", "free", "aware")}
            res[name] = dict(R={P: {k2: np.nanmean(np.asarray(v), 0).round(3).tolist() for k2, v in Rr[P].items()}
                                for P in Rr}, heldout_err={"free": hr[0].round(4).tolist(), "aware": hr[1].round(4).tolist()},
                             spread=spread, n_groups=len(Gs), n_cmds=len(zs))
    held.clear()

    lines = [f"state_projector_diag tag={a.tag} n_branch={a.n_branch} K={K}"]
    for kind, (_, e_tr, e_va) in projs.items():
        lines.append(f"{kind}: read err std (fwd, lat, yaw, zMSE) train {e_tr} | val {e_va}")
    for name, r in res.items():
        lines.append(f"\n=== {name} heldout ({r['n_groups']} groups, {r['n_cmds']} commands); r fwd / lat / yaw")
        for P, d in r["R"].items():
            lines.append(f" P={P}")
            for k2, v in d.items():
                lines.append(f"   {k2:<10}" + " / ".join(f"{x:.2f}" for x in v))
        lines.append(f" heldout rollout read err (std, pair 0 vs FTM(z_true) read) {r['heldout_err']}")
        lines.append(f" across-state std of z per command (mean over dims and commands) {r['spread']}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    with open(os.path.join(OUT, f"result_{a.tag}.txt"), "w") as fo:
        fo.write(txt + "\n")


if __name__ == "__main__":
    main()
