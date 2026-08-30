"""Fit the shared body-motion head on a robot the pretrain never showed it.

    .venv/bin/python3 -m wm.fit_body_head \\
        --ckpt wm/runs/beh12_hexonly/stage3_b1_nce_s0.pt --data data/beh12_b1_flat

**Why this is now allowed, when `wm/adapt.py` deliberately leaves the motion decoder alone.** That
file's reason is written down and was correct: the decoder "is an auxiliary loss during pretraining
and plays no part at control time, so adapting it would cost time for a module the planner never
calls". **F128 changed that premise** -- it scores candidates with `body_head(proj(a))`, so the head
is now on the control path, and the argument for leaving it unadapted no longer holds.

**What is being tested, and the three outcomes are named in advance** (F130):

    B1 calibrates and the insect stays good   the shared coordinate is real and the head was simply
                                              never adapted -- no pretraining change needed
    B1 calibrates and the insect breaks       one head cannot serve both: `z` is swappable, not
                                              shared, which is F83's channel competition in another
                                              form
    B1 still fails                            the B1's latent carries no body-motion signal to read,
                                              and only a pretraining change can put one there

**Everything is frozen but the head**, which is `z_dim -> body_hidden -> body_dim`, about 8k
parameters. The latent comes from the checkpoint's own ITM on real consecutive frames, so this asks
what is *in* `z`, not what the projector can reach -- the projector's own limits are F97's problem
and would confound this one.

**The split is by clip and the evaluation is on held-out clips only.** Consecutive frames of one
clip are near-duplicates, so a frame-level split leaves the training data in the test set -- the
leak that made yaw look like it transferred until it was held out by condition instead (F76).
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
from wm.evaluate import encode_clip  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="**the checkpoint the planner uses**, not the "
                                                  "pretrain: stage 1 moves what `z` means and the "
                                                  "head has to be fitted against the latent it "
                                                  "will actually be shown (F129)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--also", nargs="*", default=[], metavar="EMBODIMENT=DIR",
                    help="fit on these embodiments as well, one head for all of them. **This is "
                         "the test of the shared coordinate itself**: fitting on the B1 alone "
                         "improves it there and costs the insect, so the question is whether one "
                         "head can hold both at once or whether `z` is only swappable.")
    ap.add_argument("--also_cache", default="results/wm/cache/bodycal_hexapod.pt")
    ap.add_argument("--latent", choices=("itm", "projector", "both"), default="itm",
                    help="which latent the head is fitted against. **`itm` asks what is in `z`; "
                         "`projector` asks what the head will actually be shown at control time**, "
                         "and they are not the same distribution -- `a -> z` is one-to-many (F97). "
                         "Fitting on `itm` lifts the ITM path from +0.20 to +0.79 and leaves the "
                         "projector path at +0.44 while widening its range 2.5x, which made "
                         "selection worse rather than better (F130). "
                         "**`both` is what a scored run actually needs**: the head is shown "
                         "`proj(a)` on the candidate side under mode D, the ITM's latent on the "
                         "candidate side under mode C, and the ITM's latent on the goal side under "
                         "either -- so it is fitted on the union. `--also` embodiments contribute "
                         "ITM latents only, since a stage-3 projector carries a head for the "
                         "adapted robot alone.")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--cache", default="results/wm/cache/b1.pt")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
    std = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)

    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    for p in itm.parameters():
        p.requires_grad_(False)
    projector = None
    if args.latent in ("projector", "both"):
        from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
        projector = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
        projector.load_state_dict(ck["projector"])
        for p in projector.parameters():
            p.requires_grad_(False)

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    zs, ys, groups = [], [], []
    spec = REGISTRY[args.embodiment]
    with torch.no_grad():
        for i, path in enumerate(paths):
            clip = load(path, spec)
            if path not in cache:
                cache[path] = encode_clip(encoder, clip["frames"], args.chunk).cpu().half()
            e = cache[path].float().to(device)
            motion = np.asarray(clip["body_motion"])[:, channels]
            n = min(len(e) - 1, len(motion) - 1)
            z_itm = (torch.cat([itm(e[t:t + 1], e[t + 1:t + 2]) for t in range(n)]).cpu()
                     if args.latent in ("itm", "both") else None)
            z_proj = None
            if projector is not None:
                acts = torch.tensor(np.asarray(clip["actions"])[:n], dtype=torch.float32).to(device)
                z_proj = projector(acts, args.embodiment).cpu()
            z = torch.cat([x for x in (z_itm, z_proj) if x is not None])
            zs.append(z)
            reps = sum(x is not None for x in (z_itm, z_proj))
            ys.append(torch.tensor(motion[:n], dtype=torch.float32).repeat(reps, 1))
            groups.append(torch.full((n * reps,), i))
    if len(cache) > before:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
    del encoder
    torch.cuda.empty_cache()

    extra_z, extra_y = [], []
    if args.also:
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        cache2_path = os.path.join(ROOT, args.also_cache)
        cache2 = torch.load(cache2_path, map_location="cpu") if os.path.exists(cache2_path) else {}
        n2 = len(cache2)
        with torch.no_grad():
            for spec_str in args.also:
                other, other_dir = spec_str.split("=", 1)
                for path in sorted(glob.glob(os.path.join(ROOT, other_dir, "*.npz"))):
                    clip = load(path, REGISTRY[other])
                    if path not in cache2:
                        cache2[path] = encode_clip(encoder, clip["frames"], args.chunk).cpu().half()
                    e = cache2[path].float().to(device)
                    motion = np.asarray(clip["body_motion"])[:, channels]
                    n = min(len(e) - 1, len(motion) - 1)
                    # **ITM latents only for the other robot.** A stage-3 projector carries a
                    # head for the robot it was adapted to and nothing else, and the goal side is
                    # read through the ITM at scoring time anyway.
                    extra_z.append(torch.cat([itm(e[t:t + 1], e[t + 1:t + 2])
                                              for t in range(n)]).cpu())
                    extra_y.append(torch.tensor(motion[:n], dtype=torch.float32))
        if len(cache2) > n2:
            torch.save(cache2, cache2_path)
        del encoder
        torch.cuda.empty_cache()
        print(f"also fitting on {sum(len(x) for x in extra_z)} transitions from {args.also}")

    z = torch.cat(zs).to(device)
    y = ((torch.cat(ys) - mean) / std).to(device)
    group = torch.cat(groups)
    z_extra = torch.cat(extra_z).to(device) if extra_z else None
    y_extra = ((torch.cat(extra_y) - mean) / std).to(device) if extra_y else None

    ids = torch.unique(group)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed))
    val_ids = ids[order[:max(1, int(args.val_frac * len(ids)))]]
    val = torch.isin(group, val_ids).to(device)
    val_paths = [os.path.basename(paths[i]) for i in val_ids.tolist()]
    print(f"{len(ids)} clips, {int(val.sum())} of {len(z)} transitions held out "
          f"({len(val_ids)} clips)")
    print(f"held out: {sorted(val_paths)}\n")

    # **The real action width, not a placeholder.** Building the decoder with a stand-in width and
    # then saving its state dict overwrites the checkpoint's own output head with a differently
    # shaped one, and every script that rebuilds a full model from it fails to load. Only
    # `body_head` is meant to change here.
    action_dim = int(load(paths[0], spec)["actions"].shape[1])
    md = MotionDecoder(cfg, {args.embodiment: action_dim}).to(device)
    md.load_state_dict(ck["md"], strict=False)
    if md.body_head is None:
        raise SystemExit("this checkpoint has no body head (lambda_body 0)")
    for p in md.parameters():
        p.requires_grad_(False)
    for p in md.body_head.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in md.body_head.parameters())

    def report(tag):
        md.eval()
        with torch.no_grad():
            pred = md.body(None, z)
            for name, m in (("train", ~val), ("held out", val)):
                err = torch.nn.functional.mse_loss(pred[m], y[m]).item()
                base = torch.nn.functional.mse_loss(
                    y[m].mean(0, keepdim=True).expand_as(y[m]), y[m]).item()
                print(f"  {tag:<8} {name:<9} MSE {err:.4f}   predicting the mean {base:.4f}   "
                      f"ratio {err / max(base, 1e-9):.3f}")

    print(f"fitting {n_train} parameters, everything else frozen")
    report("before")
    opt = torch.optim.Adam(md.body_head.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        md.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(md.body(None, z[~val]), y[~val])
        if z_extra is not None:
            loss = loss + torch.nn.functional.mse_loss(md.body(None, z_extra), y_extra)
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:4d}  train {loss.item():.4f}")
    report("after")
    print("\n  ratio is against predicting the target's mean: **below 1.0 on the held-out clips is")
    print("  the only line that means anything**, and a ratio near 1.0 there is a head that has")
    print("  learned the dataset mean and nothing else -- which is what F129 measured this head")
    print("  doing on the B1 before any fitting.")

    if args.out:
        out = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # **Update the body head only.** The checkpoint's decoder may carry heads for embodiments
        # this run never loaded; writing this decoder's whole state dict would delete them.
        saved = dict(ck)
        md_state = dict(ck["md"])
        for k, v in md.body_head.state_dict().items():
            md_state[f"body_head.{k}"] = v.cpu()
        saved["md"] = md_state
        saved["body_head_fit"] = {"data": args.data, "embodiment": args.embodiment,
                                  "val_paths": sorted(val_paths), "epochs": args.epochs,
                                  "lr": args.lr, "source": args.ckpt}
        torch.save(saved, out)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
