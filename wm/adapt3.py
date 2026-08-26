"""LAC-WM's stage 3: fine-tune the action projector and the forward model *together*.

**This exists because of F97, not as a refinement of stage 2.** Stage 2 fits the projector by
regressing `proj(a)` onto the ITM's latent `z`. On the hexapod that works (rollout gap 0.230).
On the B1 it does not (0.640 on the same latent space and clip count, 0.841 after adaptation,
1.301 on the few-shot budget) and the reason is a property of the robot: the B1's action is a
PPO policy's *response to state*, so the same twelve numbers occur in different states and are
followed by different transitions. **`a -> z` is one-to-many, and stage 2 is asking a network to
learn a mapping that is not a function.** No amount of data fixes a target like that.

Stage 3 changes the target rather than the optimiser:

    stage 2     loss = MSE( proj(a) , z_ITM )              one-to-many, unfittable here
    stage 3     loss = MSE( FDM(e_t, proj(a)) , e_t+1 )    a function, because e_t is an input

The projector no longer has to hit one particular latent. It has to emit *any* latent that drives
the forward model to the right next embedding, and the forward model may move to meet it -- which
is the half stage 2 cannot do with the FDM frozen. **Note that no ITM appears above.** Stage 3
does not consume `z_ITM` at all, so it cannot inherit the ITM's one-to-many problem.

**The MSE form of stage 3 does not work here, and `--lambda_nce` exists because of how it fails.**
Given 15k steps, 24 clips and lr 1e-4, the forward model cuts its training loss sixfold and its
held-out prediction to 0.805 of hold-still -- it genuinely learns B1 dynamics -- while `/mean-z`
stays at 0.993 and family selection stays at chance. **It improves by predicting better on
average and never learns to use the action at all.** The reason is in the objective rather than
the budget: the action-dependent part of `e_t+1` is a small fraction of its variance, so gradient
descent banks the large unconditional win and is never obliged to earn the small conditional one.
**MSE rewards prediction; planning needs discrimination**, and they are not the same objective.

So stage 3 here optionally adds an InfoNCE term that asks for the thing a planner actually does:
the true action must reach `e_t+1` more closely than actions drawn from other behaviours.

    --lambda_nce 0    faithful LAC-WM stage 3, MSE only
    --lambda_nce 1    plus contrastive, which optimises the metric the planner is scored on

**The failure mode this invites, and why the tables below are shaped as they are.** `e_t+1` is
close to `e_t`, so a model that learns to ignore `z` and copy its input scores well on the loss
while being useless for control -- the same collapse `finetune_ftm.py` watches with its `moves`
ratio. MSE alone cannot see it. So the metric that decides this file is **discrimination**: given
the true next embedding, does the forward model rank the action that actually caused it above
actions from other behaviours? A collapsed model scores at chance by construction, and a planner
is exactly a discriminator, so this is the closed-loop result available without rendering.

    .venv/bin/python3 -m wm.adapt3 --ckpt wm/runs/beh12_hexonly/adapted_b1.pt \\
        --projector wm/runs/beh12_hexonly/projector_b1_adapted.pt \\
        --data data/beh12_b1_flat --embodiment b1 --out wm/runs/beh12_hexonly/stage3_b1.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.action_projector import ActionProjector  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402

FAMILY = lambda cond: cond.rsplit("_", 1)[0] if "_" in cond else cond


def gather(directory, name, encoder, checkpoint, cache, chunk, lag, device):
    """One record per clip: embeddings, aligned actions, condition.

    **Embeddings stay on the CPU in half precision and keep all `T+1` rows.** Stage 2's `gather`
    drops the final frame because it only ever needs `z`; stage 3's target *is* the next
    embedding, so the row that stage 2 discards is a training label here.
    """
    clips = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
        clip = load(path, REGISTRY[name])
        if path not in cache:
            cache[path] = encode_clip(encoder, clip["frames"], chunk).cpu().half()
        e = cache[path].float()
        off = offset_for(checkpoint, name)
        if off is not None:
            e = e - off.cpu()
        n = len(e) - 1
        actions = torch.as_tensor(np.asarray(clip["actions"]), dtype=torch.float32)
        if len(actions) < n + lag:  # a padded action is a wrong label, and F45 priced wrong labels
            continue
        with np.load(path, allow_pickle=True) as raw:
            cond = str(raw["condition"])
        clips.append({"path": os.path.basename(path), "cond": cond, "n": n,
                      "e": e.half(), "a": actions[lag:lag + n]})
    return clips


def batches(clips, idx, size, generator=None):
    order = torch.randperm(len(idx), generator=generator) if generator is not None \
        else torch.arange(len(idx))
    for i in range(0, len(order), size):
        yield [idx[j] for j in order[i:i + size].tolist()]


def stack(clips, pairs, device):
    e_t = torch.stack([clips[c]["e"][t] for c, t in pairs]).float().to(device)
    e_next = torch.stack([clips[c]["e"][t + 1] for c, t in pairs]).float().to(device)
    a = torch.stack([clips[c]["a"][t] for c, t in pairs]).to(device)
    return e_t, e_next, a


@torch.no_grad()
def discriminate(clips, val, cand, proj, ftm, name, device, limit=240, seed=0):
    """Can the forward model pick the action that actually happened, out of one per condition?

    Scored by **behaviour family** rather than exact condition, matching how the closed loop is
    scored: `side_R_lvl0` chosen for a `side_R_lvl1` demonstration is the right behaviour at the
    wrong amplitude, and calling that a miss would understate a planner that is working.
    Candidates are taken at the same time index from a *different* clip of each condition, so the
    correct answer is never the held-out clip's own actions.
    """
    proj.eval(); ftm.eval()
    conds = sorted(cand)
    g = torch.Generator().manual_seed(seed)
    picks = val if len(val) <= limit else [val[i] for i in
                                           torch.randperm(len(val), generator=g)[:limit].tolist()]
    hit = fam_hit = 0
    # **Chance is not 1/N for the family score, and reading it as 1/N reports chance as success.**
    # Families here hold unequal numbers of conditions -- speed 4, turn 4, side_L 2, side_R 2 --
    # so a uniform guess lands in the right family about 27% of the time, which is exactly what an
    # unadapted model scores. The chance rate is accumulated per pick, over the candidates actually
    # offered at that time index, rather than assumed.
    fam_chance = 0.0
    for c, t in picks:
        truth = clips[c]["e"][t + 1].float().to(device).unsqueeze(0)
        e_t = clips[c]["e"][t].float().to(device).unsqueeze(0)
        acts, keep = [], []
        for k in conds:
            src = cand[k]
            if t < clips[src]["n"]:
                acts.append(clips[src]["a"][t]); keep.append(k)
        if len(keep) < 2:
            continue
        a = torch.stack(acts).to(device)
        pred = ftm(e_t.expand(len(keep), -1, -1), proj(a, name))
        err = ((pred - truth) ** 2).flatten(1).mean(1)
        best = keep[int(err.argmin())]
        truth_fam = FAMILY(clips[c]["cond"])
        hit += best == clips[c]["cond"]
        fam_hit += FAMILY(best) == truth_fam
        fam_chance += sum(FAMILY(k) == truth_fam for k in keep) / len(keep)
    n = max(len(picks), 1)
    return hit / n, fam_hit / n, len(picks), len(conds), fam_chance / n


@torch.no_grad()
def score(clips, val, proj, ftm, name, device, size=16):
    """Next-embedding error against the two baselines a collapsed model would score well on."""
    proj.eval(); ftm.eval()
    num = hold = mean_z = 0.0
    moves_p = moves_t = 0.0
    z_bar = None
    for pairs in batches(clips, val, size):
        e_t, e_next, a = stack(clips, pairs, device)
        z = proj(a, name)
        if z_bar is None:
            z_bar = z.mean(0, keepdim=True)
        pred = ftm(e_t, z)
        num += ((pred - e_next) ** 2).mean().item() * len(pairs)
        hold += ((e_t - e_next) ** 2).mean().item() * len(pairs)
        mean_z += ((ftm(e_t, z_bar.expand(len(pairs), -1)) - e_next) ** 2).mean().item() * len(pairs)
        moves_p += (pred - e_t).norm(dim=(1, 2)).sum().item()
        moves_t += (e_next - e_t).norm(dim=(1, 2)).sum().item()
    n = max(len(val), 1)
    return num / n, (num / n) / max(hold / n, 1e-9), (num / n) / max(mean_z / n, 1e-9), \
        moves_p / max(moves_t, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="stage 1 output -- the adapted checkpoint")
    ap.add_argument("--projector", default="", help="stage 2 output to start from; random if empty")
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--train_clips", nargs="*", default=[],
                    help="clip basenames to fit on. Default: whatever stage 1 recorded in the "
                         "checkpoint, so the few-shot budget is the same one and not quietly "
                         "widened by running this file.")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr_proj", type=float, default=1e-3)
    ap.add_argument("--lr_ftm", type=float, default=1e-5,
                    help="an order below the projector's: the forward model is 77M pretrained "
                         "parameters being nudged, the projector is being learned")
    ap.add_argument("--candidates", choices=("holdout", "train"), default="holdout",
                    help="where the candidate library comes from. **`train` is the deployment "
                         "situation**: you recorded one set of clips on the new robot and it "
                         "serves as adaptation data, projector data and candidate library at "
                         "once, because there is no reason to hold a recorded clip back from "
                         "being a candidate. `holdout` keeps them disjoint, which is stricter "
                         "but spends 36 clips where deployment spends 24 -- and comparing a run "
                         "under one rule against a run under the other measures the rule.")
    ap.add_argument("--lambda_nce", type=float, default=0.0,
                    help="weight on the contrastive term. 0 is faithful LAC-WM stage 3")
    ap.add_argument("--negatives", type=int, default=3,
                    help="actions per step drawn from *other conditions*, as the negatives. Drawn "
                         "at the same time index so gait phase is not the giveaway")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--cache", default="results/wm/cache/b1_all.pt")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    name = args.embodiment

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, args.data), name, encoder, checkpoint, cache,
                   args.chunk, max(1, cfg.action_lag), device)
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder, cache
    torch.cuda.empty_cache()

    by_name = {c["path"]: i for i, c in enumerate(clips)}
    train_names = args.train_clips or list(checkpoint.get("adapted", {}).get("train_paths", []))
    if not train_names:
        raise SystemExit("no training clips: pass --train_clips or use a stage 1 checkpoint")
    train_ids = [by_name[n] for n in train_names if n in by_name]

    # **Candidates come from the first clip of each condition, and those are barred from the test
    # set.** The planner picks among recorded behaviours; scoring it on a clip whose own actions
    # are in the candidate list measures matching a clip against itself.
    if args.candidates == "train":
        cand = {}
        for i in train_ids:
            cand.setdefault(clips[i]["cond"], i)
        val_ids = [i for i in range(len(clips)) if i not in train_ids]
    else:
        cand = {}
        for i, c in enumerate(clips):
            cand.setdefault(c["cond"], i)
        val_ids = [i for i, c in enumerate(clips)
                   if i not in train_ids and i not in cand.values()]

    # **Negatives come from a different condition, never merely a different clip.** Two clips of
    # the same condition differ by noise, and a term that pushes them apart teaches the model to
    # separate things a planner must treat as the same.
    others = {cond: [i for i in train_ids if clips[i]["cond"] != cond]
              for cond in {c["cond"] for c in clips}}

    train = [(c, t) for c in train_ids for t in range(clips[c]["n"])]
    val = [(c, t) for c in val_ids for t in range(clips[c]["n"])]
    print(f"{len(clips)} clips | fit on {len(train_ids)} ({len(train)} transitions) | "
          f"test on {len(val_ids)} held-out clips ({len(val)}) | "
          f"{len(cand)} candidate conditions from the {args.candidates} set")
    print(f"  fit: {sorted(clips[i]['path'] for i in train_ids)}")

    proj = ActionProjector(cfg, {name: clips[0]["a"].shape[1]}).to(device)
    a_all = torch.cat([clips[i]["a"] for i in train_ids])
    proj.set_stats(name, a_all.mean(0), a_all.std(0))
    if args.projector:
        st = torch.load(os.path.join(ROOT, args.projector), map_location="cpu",
                        weights_only=False)
        st = st.get("projector", st)
        missing = proj.load_state_dict(st, strict=False)
        print(f"  projector initialised from stage 2 ({args.projector})"
              f"{'' if not missing.missing_keys else ' -- partial: ' + str(missing.missing_keys)}")
    ftm = ForwardTransitionModel(cfg).to(device)
    ftm.load_state_dict(checkpoint["ftm"])

    opt = torch.optim.Adam([{"params": proj.parameters(), "lr": args.lr_proj},
                            {"params": ftm.parameters(), "lr": args.lr_ftm}])

    print(f"\n{'step':>6}{'train':>10}{'/hold':>8}{'/mean-z':>9}{'moves':>8}"
          f"{'cond':>7}{'family':>8}")
    top1, fam, npick, ncond, fam_chance = discriminate(clips, val, cand, proj, ftm,
                                                          name, device)
    mse, r_hold, r_mean, moves = score(clips, val, proj, ftm, name, device)
    print(f"{'stage2':>6}{mse:>10.4f}{r_hold:>8.3f}{r_mean:>9.3f}{moves:>8.2f}"
          f"{top1:>7.0%}{fam:>8.0%}   <- before stage 3")
    chance = 1.0 / max(ncond, 1)

    g = torch.Generator().manual_seed(0)
    step, run = 0, 0.0
    while step < args.steps:
        for pairs in batches(clips, train, args.batch, g):
            proj.train(); ftm.train()
            e_t, e_next, a = stack(clips, pairs, device)
            loss = torch.nn.functional.mse_loss(ftm(e_t, proj(a, name)), e_next)
            if args.lambda_nce > 0:
                # **The positive and its negatives go through the model in one batch.** Scoring
                # them in separate passes lets batch-order effects and any normalisation drift
                # between them, and the whole term is a comparison of those scores.
                alt = []
                for (c, t) in pairs:
                    pool = [j for j in others.get(clips[c]["cond"], []) if t < clips[j]["n"]]
                    pick = [pool[int(torch.randint(len(pool), (1,), generator=g))]
                            for _ in range(args.negatives)] if pool else []
                    alt.append(torch.stack([clips[j]["a"][t] for j in pick])
                               if pick else clips[c]["a"][t].unsqueeze(0).repeat(
                                   args.negatives, 1))
                neg = torch.stack(alt).to(device)                      # B x K x action_dim
                B, K, _ = neg.shape
                cand_a = torch.cat([a.unsqueeze(1), neg], 1).reshape(B * (K + 1), -1)
                pred = ftm(e_t.repeat_interleave(K + 1, 0), proj(cand_a, name))
                err = ((pred - e_next.repeat_interleave(K + 1, 0)) ** 2).flatten(1).mean(1)
                # index 0 of every group is the action that actually happened
                logits = (-err.view(B, K + 1)) / args.temp
                loss = loss + args.lambda_nce * torch.nn.functional.cross_entropy(
                    logits, torch.zeros(B, dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + list(ftm.parameters()),
                                           cfg.grad_clip)
            opt.step()
            run += loss.item(); step += 1
            if step % args.every == 0:
                top1, fam, _n, _c, fam_chance = discriminate(clips, val, cand, proj,
                                                                 ftm, name, device)
                mse, r_hold, r_mean, moves = score(clips, val, proj, ftm, name, device)
                print(f"{step:>6}{run / args.every:>10.4f}{r_hold:>8.3f}{r_mean:>9.3f}"
                      f"{moves:>8.2f}{top1:>7.0%}{fam:>8.0%}")
                run = 0.0
            if step >= args.steps:
                break

    print(f"\n`/hold` and `/mean-z` are ratios: below 1.0 beats the baseline. `moves` is predicted "
          f"over\nactual displacement -- near 0 is the collapse where the model copies its input, "
          f"which scores well\non MSE and cannot control anything.\n\n`cond` picks 1 of {ncond} "
          f"exactly (chance {chance:.0%}); `family` allows the right behaviour at the\nwrong "
          f"amplitude. **Chance for `family` is {fam_chance:.0%}, not {chance:.0%}** -- the "
          f"families hold unequal\nnumbers of conditions, and an unadapted model scores the "
          f"chance rate exactly. Selection is\nhappening only if `family` clears it by a margin: "
          f"{fam:.0%} vs {fam_chance:.0%} is "
          f"{'SELECTION' if fam > fam_chance + 0.15 else 'NOT selection'}.")

    if args.out:
        out = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"config": checkpoint["config"], "itm": checkpoint["itm"], "ftm": ftm.state_dict(),
                    "md": checkpoint["md"], "projector": proj.state_dict(),
                    "action_stats": checkpoint.get("action_stats"),
                    "body_stats": checkpoint.get("body_stats"),
                    "adapted": checkpoint.get("adapted"),
                    "stage3": {"train_paths": [clips[i]["path"] for i in train_ids],
                               "val_paths": [clips[i]["path"] for i in val_ids],
                               "steps": args.steps, "embodiment": name,
                               "candidates": args.candidates,
                               "candidate_paths": [clips[i]["path"] for i in cand.values()],
                               "top1": top1, "family": fam, "cond_chance": chance,
                               "family_chance": fam_chance}}, out)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
