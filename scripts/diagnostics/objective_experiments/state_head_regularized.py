"""Is Delta-state alive under a head shaped like ridge's function class, or is 0.852 unreachable
by any neural head on this data?

    .venv/bin/python3 scripts/diagnostics/objective_experiments/state_head_regularized.py

`state_head_ceiling.py` and `state_head_aug_confirm.py` both found a 256-hidden, 2-layer MLP
memorises 34 training clips (state-loss -> ~0.0008) regardless of augmentation, landing at held-out
R2 ~0.06-0.15 against the offline ridge's 0.852 on the same `[pooled delta, z]` inputs. Ridge is a
LINEAR, heavily cross-validated-regularised function class. This shrinks the head to match that
class directly, instead of assuming the MLP was merely under-regularised.

    LinearStateHead   pooled delta - offset, concatenated with z, ONE linear layer to state_dim.
                      No hidden layer, no nonlinearity -- the closest a trained network gets to
                      ridge's own structure.

Plus what ridge had and the earlier neural runs did not:

    weight decay      strong, on the head only (0.1) -- FTM keeps light decay so its pretrained
                      dynamics are not regularised away along with the head's capacity
    early stopping    a further split carves a validation slice OUT of the training clips (never
                      the test clips); the snapshot with the best validation state-loss is what
                      gets scored on held-out test, not whatever the last step happens to be --
                      the neural equivalent of ridge's alpha selected by held-out clips

**The discriminator.** Climbs toward 0.852 -> the Delta-state direction was alive, the earlier MLP
was simply the wrong function class for 34 clips, and a small/regularised head is the com7 design.
Stays near 0.06-0.15 despite matching ridge's shape -> the offline ridge result does not reproduce
under a trained, early-stopped model on the same data, and Delta-state is capped here regardless of
head design.

Cross-augmentation on throughout, matching the realistic setting `state_head_aug_confirm.py`
established as necessary to test honestly.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.augment import sample_params, apply, identity_params  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


class LinearStateHead(nn.Module):
    """The closest a trained network gets to ridge's own function class: one linear layer over
    [pooled delta - offset, z], nothing else."""

    def __init__(self, cfg, state_dim, embodiments):
        super().__init__()
        self.linear = nn.Linear(cfg.token_dim + cfg.z_dim, state_dim)
        for name in embodiments:
            self.register_buffer(f"offset_{name}", torch.zeros(cfg.token_dim))

    def set_offset(self, embodiment, mean):
        with torch.no_grad():
            getattr(self, f"offset_{embodiment}").copy_(torch.as_tensor(mean, dtype=torch.float32))

    def forward(self, delta, z, embodiment="default"):
        d = delta.mean(1) if delta.dim() == 3 else delta
        d = d - getattr(self, f"offset_{embodiment}")
        return self.linear(torch.cat([d, z], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--sources", nargs="+",
                    default=["hexapod=data/egocentric/beh12_c10f10t10_ego_flat",
                             "b1=data/egocentric/beh12_b1_ego_flat"])
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="carved from TRAIN clips, for early stopping -- never touches test")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_ftm", type=float, default=1e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd_head", type=float, default=0.1)
    ap.add_argument("--wd_ftm", type=float, default=0.0)
    ap.add_argument("--lambda_state", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--freeze_ftm", action="store_true",
                    help="**the moving-target test.** With FTM frozen, pooled delta is a static "
                         "function of (e_t, e_t+1) alone -- the same input the offline ridge "
                         "oracle saw. If R2 closes toward 0.852 here but not with FTM trainable, "
                         "the earlier ceiling was FTM drift, not head capacity or Delta-state "
                         "itself.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    sources = [tuple(s.split("=", 1)) for s in args.sources]
    names = [n for n, _ in sources]

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(ck["ftm"])
    if args.freeze_ftm:
        ftm.eval()
    for p in ftm.parameters():
        p.requires_grad_(not args.freeze_ftm)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    clips_by_e = {}
    rng = np.random.default_rng(args.seed)
    fit_idx, val_idx, test_idx = {}, {}, {}
    for eid, (name, path) in enumerate(sources):
        paths = sorted(glob.glob(os.path.join(ROOT, path, "*.npz")))
        clips = []
        for p in paths:
            d = load(p, REGISTRY[name])
            clips.append({"frames": d["frames"], "bm": np.asarray(d["body_motion"])[:, channels]})
        clips_by_e[eid] = clips
        order = list(range(len(clips)))
        rng.shuffle(order)
        n_te = max(1, int(len(order) * args.test_frac))
        test_idx[eid] = order[:n_te]
        rest = order[n_te:]
        n_va = max(1, int(len(rest) * args.val_frac))
        val_idx[eid] = rest[:n_va]
        fit_idx[eid] = rest[n_va:]
        print(f"  {name}: {len(clips)} clips -> {len(fit_idx[eid])} fit / {len(val_idx[eid])} val "
              f"/ {len(test_idx[eid])} test", flush=True)

    def pts_of(idx_map, eid):
        out = []
        for ci in idx_map[eid]:
            n = min(len(clips_by_e[eid][ci]["frames"]) - 2, len(clips_by_e[eid][ci]["bm"]))
            out.extend((ci, t) for t in range(1, n, args.stride))
        return out

    fit_pts = {eid: pts_of(fit_idx, eid) for eid in range(len(names))}
    val_pts = {eid: pts_of(val_idx, eid) for eid in range(len(names))}
    test_pts = {eid: pts_of(test_idx, eid) for eid in range(len(names))}

    def encode_pair(eid, ci, t, augmented, arng):
        frames = clips_by_e[eid][ci]["frames"]
        f_t, f_1 = frames[t], frames[t + 1]
        par = sample_params(arng, *f_t.shape[:2]) if augmented else identity_params(*f_t.shape[:2])
        return apply(f_t, par), apply(f_1, par)

    @torch.no_grad()
    def compute_offset():
        offs = {}
        for eid, name in enumerate(names):
            total, count = None, 0
            for ci, t in fit_pts[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                d = (ftm(e[0:1], z) - e[0:1]).mean(1)[0]
                total = d if total is None else total + d
                count += 1
            offs[name] = (total / max(count, 1)).cpu()
        return offs

    print("\n  fitting per-embodiment offset (un-augmented, FIT clips only)", flush=True)
    offsets = compute_offset()

    # **The bug this run exists to fix.** Every earlier confirm trained on RAW body_motion (e.g.
    # magnitude ~0.02-0.15) but evaluated R2 by standardising the TRUTHS only, comparing them
    # against the model's raw-scale output -- a units mismatch that manufactures a large residual
    # regardless of model quality. Standardised here, matching ridge's own preprocessing, and used
    # consistently for both the training loss and the R2 evaluation below.
    all_bm = np.concatenate([clips_by_e[eid][ci]["bm"][t] .reshape(1, -1)
                             for eid in range(len(names)) for ci, t in fit_pts[eid]], axis=0)
    y_mu = all_bm.mean(0); y_sd = all_bm.std(0) + 1e-9
    print(f"  target standardisation (fit clips only): mu {y_mu}, sd {y_sd}", flush=True)

    state = LinearStateHead(cfg, cfg.body_dim, names).to(device)
    for name in names:
        state.set_offset(name, offsets[name])
    groups = [{"params": state.parameters(), "lr": args.lr_head, "weight_decay": args.wd_head}]
    if not args.freeze_ftm:
        groups.append({"params": ftm.parameters(), "lr": args.lr_ftm, "weight_decay": args.wd_ftm})
    opt = torch.optim.AdamW(groups)
    g = torch.Generator().manual_seed(args.seed)
    arng = np.random.default_rng(args.seed + 1)

    @torch.no_grad()
    def r2_on(pts_by_e):
        out = {}
        for eid, name in enumerate(names):
            preds, truths = [], []
            for ci, t in pts_by_e[eid]:
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                pred = ftm(e[0:1], z)
                sp = state(pred - e[0:1], z, name)[0].cpu().numpy()   # standardised-scale output
                preds.append(sp); truths.append(clips_by_e[eid][ci]["bm"][t])
            preds, truths = np.stack(preds), np.stack(truths)
            truths_std = (truths - y_mu) / y_sd     # SAME transform the model was trained to predict
            ss = ((preds - truths_std) ** 2).sum()
            ss_tot = (truths_std ** 2).sum()        # fit-mean-centred already, so this is the total
            out[name] = 1 - ss / max(ss_tot, 1e-9)
        return out

    @torch.no_grad()
    def val_loss():
        losses = []
        for eid, name in enumerate(names):
            for ci, t in val_pts[eid][:60]:            # cap for speed, re-encoding is the cost
                a, b = encode_pair(eid, ci, t, False, None)
                e = encoder.encode([a, b]).float()
                z = itm(e[0:1], e[1:2])
                pred = ftm(e[0:1], z)
                sp = state(pred - e[0:1], z, name)
                y = torch.tensor((clips_by_e[eid][ci]["bm"][t] - y_mu) / y_sd, dtype=torch.float32,
                                 device=device).unsqueeze(0)
                losses.append(F.mse_loss(sp, y).item())
        return float(np.mean(losses))

    print(f"\n  training regularised LINEAR head, wd_head={args.wd_head}, {args.steps} steps",
          flush=True)
    t0 = time.time()
    best_val, best_state, best_ftm, best_step = float("inf"), None, None, -1
    for step in range(args.steps):
        eid = step % len(names)
        pool = fit_pts[eid]
        pick = [pool[i] for i in torch.randint(len(pool), (args.batch,), generator=g).tolist()]
        frames_batch = []
        for ci, t in pick:
            a, b = encode_pair(eid, ci, t, True, arng)
            frames_batch.extend([a, b])
        e = encoder.encode(frames_batch).float()
        e = e.view(len(pick), 2, *e.shape[1:])
        e_t, e1 = e[:, 0], e[:, 1]
        y = torch.tensor(np.stack([(clips_by_e[eid][ci]["bm"][t] - y_mu) / y_sd for ci, t in pick]),
                         dtype=torch.float32, device=device)
        with torch.no_grad():
            z = itm(e_t, e1)
        pred = ftm(e_t, z)
        recon_loss = F.mse_loss(pred, e1)
        sp = state(pred - e_t, z, names[eid])
        state_loss = F.mse_loss(sp, y)
        loss = recon_loss + args.lambda_state * state_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(ftm.parameters()) + list(state.parameters()), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vl = val_loss()
            tag = ""
            if vl < best_val:
                best_val, best_step = vl, step
                best_state = {k: v.clone() for k, v in state.state_dict().items()}
                best_ftm = {k: v.clone() for k, v in ftm.state_dict().items()}
                tag = "  <- best so far"
            print(f"    step {step:5d}  recon {float(recon_loss.detach()):.5f}  "
                  f"state(train) {float(state_loss.detach()):.5f}  state(val) {vl:.5f}"
                  f"  ({(time.time() - t0) / max(step + 1, 1):.2f}s/step){tag}", flush=True)

    print(f"\n  best validation snapshot: step {best_step}, val state-loss {best_val:.5f}")
    state.load_state_dict(best_state)
    ftm.load_state_dict(best_ftm)

    print("\n  held-out TEST R2, best-validation snapshot (never touched by early stopping)")
    r2 = r2_on(test_pts)
    for name in names:
        print(f"    {name:>10}  R2 {r2[name]:+.3f}")
    print(f"\n  for reference: unregularised MLP (no aug) hexapod +0.057 b1 +0.095")
    print(f"                 unregularised MLP (aug)    hexapod +0.063 b1 +0.150")
    print(f"                 offline ridge oracle        0.852 (different validation split)")
    print("\n  READ: climbing toward 0.852 -> Delta-state alive, regularised design for com7.")
    print("  staying ~0.06-0.15 -> ridge's number does not reproduce under a trained, early-")
    print("  stopped model here -- Delta-state is capped on this data regardless of head shape.")


if __name__ == "__main__":
    main()
