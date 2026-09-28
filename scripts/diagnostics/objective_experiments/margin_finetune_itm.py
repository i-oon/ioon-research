"""Fine-tune the ITM with a margin loss that directly targets the within/across ceiling gap measured
by `family_z_ceiling.py` and `froude_ceiling.py`.

**Why this exists.** `family_z_ceiling.py` measured real z barely separating conditions within a
behaviour family (ratio 0.983), while `froude_ceiling.py` showed the ground-truth Froude does
separate them cleanly (ratio 0.431) -- z is discarding real signal, not reflecting a redundant task.
Raising `lambda_body` 4x (an MSE-regression reweight) did not move the ceiling at all (0.990) --
MSE optimizes average correctness, not relative separation between similar inputs, so it cannot fix
this by itself. This targets the actual measured failure directly, with a margin loss over the same
within/across pairs the diagnostic scripts sample.

**Design, deliberately conservative.** Only the ITM is updated. The FTM, Motion Decoder and
Cross-Body Head are frozen and used only to compute a reconstruction/motion anchor, so z is pushed
to separate similar conditions further apart WITHOUT being free to collapse into a trivial solution
(e.g. z -> noise, which would trivially maximise distance while breaking everything downstream that
depends on z meaning something). If the anchor terms hold roughly flat while the margin loss drops,
z has actually improved; if the anchor terms degrade sharply, the margin term found a cheap, useless
way to satisfy itself and the run should be discarded, not just this loss's own reading trusted.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/margin_finetune_itm.py \\
        --ckpt wm/runs/beh12_hinge_cleansplit/best.pt \\
        --data data/egocentric/beh12_c10f10t10_ego_flat_cleantrain \\
        --out wm/runs/beh12_margin_itm/best.pt
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

FAMILY = lambda cond: cond.rsplit("_", 1)[0] if "_" in cond else cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout_data", required=True, help="scored every --eval_every steps with a "
                     "fresh ridge probe; the checkpoint with the best held-out R2 is what gets "
                     "saved, not whatever the last step happens to produce")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--margin", type=float, default=0.3, help="d_within - d_across must fall below "
                     "-margin, i.e. across must beat within by at least this much")
    ap.add_argument("--lambda_margin", type=float, default=1.0)
    ap.add_argument("--lambda_anchor", type=float, default=1.0, help="weight on the frozen-FTM "
                     "reconstruction anchor that keeps z from collapsing into a trivial solution")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--pairs_per_step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])

    itm = InverseTransitionModel(cfg).to(device); itm.load_state_dict(ck["itm"]); itm.train()
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    for p in ftm.parameters():
        p.requires_grad_(False)

    reg = REGISTRY[args.embodiment]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    by_cond = defaultdict(list)
    train_e_flat, train_y_flat = [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as raw:
            cond = str(raw["condition"] if "condition" in raw.files else
                       raw["behavior"] if "behavior" in raw.files else "walk")
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        by_cond[cond].append(e)  # keep embeddings, not z -- z is recomputed each step with grad
        train_e_flat.append(e)
        train_y_flat.append(np.asarray(clip["body_motion"], dtype=np.float64)[: len(e) - 1])

    # held-out set, pre-encoded once; embeddings never change, only the ITM applied to them does
    ho_paths = sorted(glob.glob(os.path.join(ROOT, args.heldout_data, "*.npz")))
    ho_e, ho_y = [], []
    for p in ho_paths:
        clip = load(p, reg)
        e = encode_clip(encoder, clip["frames"], 2).float().to(device)
        ho_e.append(e)
        ho_y.append(np.asarray(clip["body_motion"], dtype=np.float64)[: len(e) - 1])
    del encoder
    torch.cuda.empty_cache()

    def held_out_r2():
        """Fresh ridge probe, z (this step's ITM) -> true Froude, fit on train clips, scored on
        held-out clips the margin loss never samples from. Cheap: no encoder call, ITM forward only."""
        itm.eval()
        with torch.no_grad():
            Z_tr = np.concatenate([itm(e[:-1], e[1:]).cpu().numpy() for e in train_e_flat])
            Y_tr = np.concatenate(train_y_flat)
            Z_te = np.concatenate([itm(e[:-1], e[1:]).cpu().numpy() for e in ho_e])
            Y_te = np.concatenate(ho_y)
        itm.train()
        mu, sd = Y_tr.mean(0), Y_tr.std(0) + 1e-9
        model = RidgeCV(alphas=np.logspace(-1, 4, 8)).fit(Z_tr, (Y_tr - mu) / sd)
        pred = model.predict(Z_te)
        return float(r2_score((Y_te - mu) / sd, pred))

    by_family = defaultdict(list)
    for cond in by_cond:
        by_family[FAMILY(cond)].append(cond)
    families = {f: cs for f, cs in by_family.items() if len(cs) >= 2}
    print(f"{len(by_cond)} conditions, {len(families)} usable families, "
          f"{sum(len(v) for v in by_cond.values())} clips")

    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(itm.parameters(), lr=args.lr)

    def sample_pair(fam_conds):
        """One (within, across) pair, mirroring family_z_ceiling.py's own sampling exactly."""
        c1 = fam_conds[rng.integers(0, len(fam_conds))]
        e1_list = by_cond[c1]
        i = rng.integers(0, len(e1_list))
        e1 = e1_list[i]
        t = int(rng.integers(0, e1.shape[0] - 1))
        anchor_e, anchor_e_next = e1[t:t + 1], e1[t + 1:t + 2]
        if len(e1_list) > 1:
            j = rng.integers(0, len(e1_list) - 1)
            j = j + 1 if j >= i else j
            e_within = e1_list[j]
        else:
            e_within = e1
        tb = int(rng.integers(0, e_within.shape[0] - 1))
        others = [c for c in fam_conds if c != c1]
        c2 = others[rng.integers(0, len(others))]
        e2_list = by_cond[c2]
        k = rng.integers(0, len(e2_list))
        e_across = e2_list[k]
        tk = int(rng.integers(0, e_across.shape[0] - 1))
        return (anchor_e, anchor_e_next, e_within[tb:tb + 1], e_within[tb + 1:tb + 2],
                e_across[tk:tk + 1], e_across[tk + 1:tk + 2])

    fam_names = list(families)
    best_r2, best_state, best_step = held_out_r2(), None, 0
    print(f"step     0 | held-out R2 (before any fine-tuning) {best_r2:.4f}  <- baseline to beat")
    best_state = {k: v.detach().clone() for k, v in itm.state_dict().items()}
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        margin_terms, anchor_terms = [], []
        for _ in range(args.pairs_per_step):
            fam = fam_names[rng.integers(0, len(fam_names))]
            a_e, a_en, w_e, w_en, x_e, x_en = sample_pair(families[fam])
            z_a = itm(a_e, a_en)
            z_w = itm(w_e, w_en)
            z_x = itm(x_e, x_en)
            d_within = ((z_a - z_w) ** 2).mean()
            d_across = ((z_a - z_x) ** 2).mean()
            margin_terms.append(F.relu(args.margin + d_within - d_across))
            # frozen-FTM reconstruction anchor: z must still let FTM predict correctly
            pred = ftm(a_e, z_a)
            anchor_terms.append(((pred - a_en) ** 2).mean())
        margin_loss = torch.stack(margin_terms).mean()
        anchor_loss = torch.stack(anchor_terms).mean()
        loss = args.lambda_margin * margin_loss + args.lambda_anchor * anchor_loss
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:5d} | margin {margin_loss.item():.4f} | anchor {anchor_loss.item():.4f} "
                  f"| total {loss.item():.4f}")
        if step % args.eval_every == 0:
            r2 = held_out_r2()
            flag = ""
            if r2 > best_r2:
                best_r2, best_step = r2, step
                best_state = {k: v.detach().clone() for k, v in itm.state_dict().items()}
                flag = "  <- new best, saved in memory"
            print(f"step {step:5d} | held-out R2 {r2:.4f} (best so far: {best_r2:.4f} @ step "
                  f"{best_step}){flag}")

    print(f"\nfinal: best held-out R2 = {best_r2:.4f} at step {best_step}")
    itm.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    ck_out = dict(ck)
    ck_out["itm"] = itm.state_dict()
    ck_out["margin_finetune_best_step"] = best_step
    ck_out["margin_finetune_best_heldout_r2"] = best_r2
    torch.save(ck_out, os.path.join(ROOT, args.out))
    print(f"saved BEST checkpoint (step {best_step}, held-out R2 {best_r2:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
