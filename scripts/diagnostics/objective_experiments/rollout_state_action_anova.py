"""Is the FTM rollout's read-out driven by the STATE it starts from or the ACTION it is given?

`selection_readout_diag.py` found: rolling each candidate's action from its OWN state tracks the
candidates' true Froude (r close to a perfect FTM), from ONE SHARED state it does not (r ~ 0).
Two readings fit that: (a) the prediction is mostly the state's own motion (a V-JEPA2 frame is a
2-frame tubelet, it already carries velocity), and the action barely moves it; (b) the action does
move it, but in a direction unrelated to its true effect once state and action disagree.

At every decision step, the full grid P[s, a] = body(ITM(e_s, FTM(e_s, proj(a_a)))) over all
state x action pairs of the candidate library, per body channel:
  state share   var_s(mean_a P) / var(P)
  action share  var_a(mean_s P) / var(P)
  r(state)      corr(mean_a P,  truth_s)   -- does the state-marginal track the state's own truth
  r(action)     corr(mean_s P,  truth_a)   -- does the action-marginal track the action's truth
  r(diag)       corr(P[i, i],   truth_i)   -- roll_own
  r(row)        mean over s of corr(P[s, :], truth_a) -- roll_shared from each state in turn

    .venv/bin/python3 scripts/diagnostics/objective_experiments/rollout_state_action_anova.py \\
        --ckpt old=wm/runs/beh12_hinge_cleansplit/b1_adapt_clean/ckpt_lib_beh12_b1_ego_flat_cleantrain.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from wm.data.strided import stride_of  # noqa: E402
from wm.policy.planner import action_chunk_at, load_candidates  # noqa: E402


class Models:
    """ITM, FTM, body head, and -- only if the checkpoint carries one -- the action projector.
    A pretrained checkpoint has no projector, which is fine for `--z_source true`."""

    def __init__(self, ckpt_path, embodiment, action_dim, device):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = from_checkpoint(ck["config"])
        self.itm = InverseTransitionModel(cfg).to(device).eval()
        self.itm.load_state_dict(ck["itm"])
        self.ftm = ForwardTransitionModel(cfg).to(device).eval()
        self.ftm.load_state_dict(ck["ftm"])
        self.md = MotionDecoder(cfg, {embodiment: action_dim}).to(device).eval()
        self.md.load_state_dict(ck["md"], strict=False)
        self.proj = None
        if "projector" in ck:
            self.proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
            self.proj.load_state_dict(ck["projector"])
        self.action_lag = max(1, cfg.action_lag)
        self.stride = stride_of(cfg)            # frames per world-model step (1 before stride existed)
        self.offset = offset_for(ck, embodiment)
        self.embodiment = embodiment

    def ftm_step(self, e, z):
        return self.ftm(e, z, self.embodiment)

CH = ["forward", "lateral", "yaw"]


def corr(x, y):
    return np.corrcoef(x, y)[0, 1] if x.std() > 1e-12 and y.std() > 1e-12 else np.nan


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--cache", default="results/wm/cache/selection_eval_cands.pt",
                    help="frame embeddings keyed by clip path; missing clips are encoded and added")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--z_source", choices=["proj", "true"], default="proj",
                    help="proj: z = proj(action), as at control time; true: z = ITM(e_a[t], e_a[t+1]), "
                         "the z the FTM was trained on -- separates a weak projector from an FTM that ignores z")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev, h = args.device, args.horizon
    cdir = os.path.join(ROOT, args.candidates_dir)
    spec_e = REGISTRY[args.embodiment]
    cands = load_candidates(cdir, args.embodiment, per_condition=999)
    motion = [np.asarray(load(c["path"], spec_e)["body_motion"])[:, :3] for c in cands]
    cache_path = os.path.join(ROOT, args.cache)
    emb = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    missing = [c["path"] for c in cands if c["path"] not in emb]
    if missing:
        from vjepa2_encoder import VJEPA2FrameEncoder
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        for p in missing:
            emb[p] = encode_clip(encoder, load(p, spec_e)["frames"], 2).cpu().half()
        del encoder
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(emb, cache_path)
    n = len(cands)
    T = min(min(len(m) for m in motion), min(len(emb[c["path"]]) for c in cands)) - h - 2

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        ckpt = os.path.join(ROOT, path)
        rp = Models(ckpt, args.embodiment, cands[0]["actions"].shape[1], dev)
        off = rp.offset
        if args.z_source == "proj" and rp.proj is None:
            raise SystemExit(f"{name}: no projector in this checkpoint; use --z_source true")
        lag = rp.action_lag
        cyc = []
        stats = {k: [] for k in ("state", "action", "r_state", "r_action", "r_diag", "r_row")}
        for t in range(0, T - (rp.stride - 1), h):     # unchanged at stride 1
            e = torch.stack([emb[c["path"]][t].float() for c in cands])
            if off is not None:
                e = e - off.float().reshape(e.shape[1:])
            e = e.to(dev)
            k = rp.stride
            # at stride k: a k-command chunk, a pair k frames apart, truth over the k steps z spans
            a = torch.as_tensor(np.stack([c["actions"][t + lag] if k == 1 else
                                          action_chunk_at(c["actions"], t + lag, k) for c in cands]),
                                device=dev)
            if args.z_source == "proj":
                z = rp.proj(a, args.embodiment)                                           # (n, z)
            else:
                e1 = torch.stack([emb[c["path"]][t + k].float() for c in cands])
                if off is not None:
                    e1 = e1 - off.float().reshape(e1.shape[1:])
                z = rp.itm(e, e1.to(dev))
            # cycle: does ITM(e_s, FTM(e_s, z_a)) give back z_a? cosine, and share of z's variance recovered
            zc = torch.stack([rp.itm(e[s:s + 1].expand(n, -1, -1), rp.ftm_step(e[s:s + 1].expand(n, -1, -1), z))
                              for s in range(0, n, 6)])                        # (4 states, n, z)
            zc_c = zc - zc.mean(1, keepdim=True)
            z_c = z - z.mean(0, keepdim=True)
            cyc.append((torch.nn.functional.cosine_similarity(zc_c, z_c.unsqueeze(0), dim=-1).mean().item(),
                        (zc_c.pow(2).sum() / (len(zc) * z_c.pow(2).sum())).item()))
            P = []
            for s in range(n):                                                 # one state row at a time
                es = e[s:s + 1].expand(n, -1, -1)
                P.append(rp.md.body(None, rp.itm(es, rp.ftm_step(es, z))).cpu())
            P = torch.stack(P).numpy()                                         # (state, action, ch)
            truth = np.stack([m[t:t + max(h, k if k > 1 else 0)].mean(0) for m in motion])
            row = []
            for j in range(3):
                p = P[..., j]
                tot = p.var() + 1e-12
                row.append((p.mean(1).var() / tot, p.mean(0).var() / tot,
                            corr(p.mean(1), truth[:, j]), corr(p.mean(0), truth[:, j]),
                            corr(np.diag(p), truth[:, j]),
                            np.nanmean([corr(p[s], truth[:, j]) for s in range(n)])))
            for k, key in enumerate(stats):
                stats[key].append([r[k] for r in row])
        print(f"\n=== {name} ===  ({n} states x {n} actions, {len(stats['state'])} steps)")
        print(f"{'':<10}" + "".join(f"{k:>10}" for k in stats))
        for j, c in enumerate(CH):
            print(f"{c:<10}" + "".join(f"{np.nanmean(np.asarray(v)[:, j]):>10.3f}" for v in stats.values()))
        cyc = np.asarray(cyc)
        print(f"cycle (z_source={args.z_source}): cos(ITM(e,FTM(e,z)) - mean, z - mean) across actions "
              f"{cyc[:, 0].mean():.3f};  action-variance ratio out/in {cyc[:, 1].mean():.3f}")
        del rp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
