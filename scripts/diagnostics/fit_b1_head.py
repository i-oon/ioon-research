"""Does a backbone trained on stick insects alone help control a quadruped?

This is the cross-embodiment transfer question asked on the one genuinely different robot we have.
The 4-leg test body (F44, F48) is the stick insect with legs removed, and F47 measured that the
model reads it as the body it was cut from -- 0.578 from the base body's latent against a chance
level of 0.981. The B1 is not that: a different robot, 12 joints against 18, a trot against a
wave, a different silhouette.

The catch is that Stage 2 *trains* on the B1, so it cannot also test on it. This uses a Stage 1
checkpoint instead -- trained on insect bodies only, never having seen a quadruped -- and asks
whether its frozen features make a B1 action head cheaper to fit than random features do.

    backbone   stage1_m3d_cross, insects only          frozen
    head       new 12-D B1 head                        fitted on a few clips
    control    same head on a random backbone          identical budget

Protocol, fitting and metrics are imported from `fit_4leg_head` so the numbers sit on the same
scale as the 4-leg result and the two can be read side by side.

**The question is how much transfers, not whether the answer is zero.** Two measurements predict
little: F41b found the frozen encoder describes a loaded leg differently for each robot -- 0.531
and 0.547 across, below chance on the front legs -- and F45 found the two robots' behaviour
distributions barely overlap, the B1 spending 84.6% of its time in two trot patterns the insect
visits 9.8% and 5.7% of the time. Report the margin either way; a prediction is not a result.

**Read the margin only where both arms work.** At five clips both the pretrained and the random
backbone score negative R^2 -- neither produces a usable head -- so their ratio compares two
failures and says nothing. At nine both are positive and the comparison means something.

  .venv/bin/python3 scripts/diagnostics/fit_b1_head.py --ckpt wm/runs/s1_m3d_cross/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from fit_4leg_head import fit_head, metrics  # noqa: E402  identical protocol to the 4-leg test

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load as load_embodiment  # noqa: E402
from wm.evaluate import encode_clip, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

HEAD = "b1"


def load_models(ckpt, random_backbone, device):
    """Freeze a checkpoint's ITM and decoder trunk, add a fresh 12-D B1 head.

    Handles both checkpoint shapes: a Stage 1 run has one unnamed 18-D head and `action_mean`,
    a Stage 2 run keys its heads and statistics by embodiment.
    """
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    if "action_stats" in checkpoint:
        heads = {k: len(v[0]) for k, v in checkpoint["action_stats"].items()}
    else:
        heads = {"default": cfg.action_dim}
    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg, heads=heads).to(device).eval()
    if not random_backbone:
        itm.load_state_dict(checkpoint["itm"])
        md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    for p in itm.parameters():
        p.requires_grad_(False)
    for p in md.parameters():
        p.requires_grad_(False)
    md.add_head(HEAD, cfg.hidden, 12, device=device)
    for p in md.heads[HEAD].parameters():
        p.requires_grad_(True)
    return cfg, itm, md


@torch.no_grad()
def build_features(encoder, itm, md, paths, mean, std, action_lag, chunk, device,
                   z_mode="real", seed=0):
    """Frozen decoder features and standardised B1 commands, one row per transition.

    Reads through `wm.data.embodiment` rather than indexing the npz directly, because the B1
    stores its command under `action` where the insect uses `actions`.

    `z_mode` isolates where the transfer comes from, same three settings as the 4-leg ablation:
      real      the latent the ITM infers from the true transition
      zero      an all-zero latent -- if this costs nothing, the frame carries the transfer and
                the learned latent contributes nothing across embodiments
      shuffled  real latents permuted within the clip, keeping their distribution but breaking
                the alignment to the frame they belong with
    """
    spec = REGISTRY["b1"]
    xs, ys, raw = [], [], []
    mean_t, std_t = torch.tensor(mean, device=device), torch.tensor(std, device=device)
    for path in paths:
        clip = load_embodiment(path, spec)
        e = encode_clip(encoder, clip["frames"], chunk).to(device)
        actions = torch.tensor(clip["actions"], device=device)
        n = min(len(e) - 1, len(actions) - action_lag)
        z = torch.cat([itm(e[s:min(s + chunk, n)], e[s + 1:min(s + chunk, n) + 1])
                       for s in range(0, n, chunk)])
        if z_mode == "zero":
            z = torch.zeros_like(z)
        elif z_mode == "shuffled":
            g = torch.Generator(device=device).manual_seed(seed)
            z = z[torch.randperm(len(z), generator=g, device=device)]
        elif z_mode != "real":
            raise ValueError(f"unknown z_mode {z_mode!r}")
        for s in range(0, n, chunk):
            t = min(s + chunk, n)
            x = md.features(e[s:t], z[s:t]).squeeze(1)
            y_raw = actions[s + action_lag:t + action_lag]
            xs.append(x.cpu())
            ys.append(((y_raw - mean_t) / std_t).cpu())
            raw.append(y_raw.cpu())
    return torch.cat(xs), torch.cat(ys), torch.cat(raw)


def condition_groups(paths):
    """Clips grouped by behaviour condition, so a split can cover every behaviour on both sides.

    Reads the `condition` field the matched collection writes into each npz. Falls back to the
    `_vx` filename convention of `data/b1_framed`, which predates that field and carries commanded
    speed in the name -- without the fallback this reads one group and silently stops stratifying.
    """
    groups = {}
    for path in paths:
        name = os.path.basename(path)
        if "_vx" in name:
            key = name.split("_vx")[1]
        else:
            with np.load(path, allow_pickle=True) as data:
                if "condition" not in data.files:
                    raise SystemExit(
                        f"--stratify needs a behaviour label: {name} has neither a `_vx` name nor "
                        "a `condition` field. Rebuild with scripts/dataset/merge_behaviour_dirs.py.")
                key = str(data["condition"])
        groups.setdefault(key, []).append(path)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/s1_m3d_cross/best.pt")
    ap.add_argument("--data", default="data/b1_framed")
    ap.add_argument("--train_clips", type=int, default=5)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--stratify", action="store_true",
                    help="split within each behaviour condition, so both sides cover every "
                         "behaviour. `--train_clips` becomes a per-condition budget. Without this, "
                         "a random draw of 7 from 48 clips spanning 12 conditions leaves whole "
                         "behaviours unseen at test time, and 'does the backbone transfer' cannot "
                         "be separated from 'does the head extrapolate to a new behaviour'.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--z_modes", nargs="+", default=["real"],
                    choices=["real", "zero", "shuffled"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data_dir = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(paths) <= args.train_clips:
        raise SystemExit(f"{len(paths)} clips is too few to split at {args.train_clips}")
    device = torch.device(args.device)
    print(f"checkpoint : {args.ckpt}")
    print(f"b1 clips   : {len(paths)}  ({args.train_clips} fitted, rest held out, "
          f"{args.splits} splits)\n")

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    rows = {}
    for split in range(args.splits):
        if args.stratify:
            groups = condition_groups(paths)
            rng = np.random.default_rng(args.seed + split)
            # `train_clips` is now per condition, not in total: the whole point is that both sides
            # cover the same behaviours, and a global budget cannot guarantee that.
            per = max(1, args.train_clips // len(groups))
            train, test = [], []
            for _, group in sorted(groups.items()):
                order = rng.permutation(len(group))
                train.extend(group[i] for i in order[:per])
                test.extend(group[i] for i in order[per:])
            print(f"  split {split}  stratified: {len(train)} train / {len(test)} test, "
                  f"{len(groups)} conditions on both sides ({per} fitted each)")
        else:
            order = np.random.default_rng(args.seed + split).permutation(len(paths))
            train = [paths[i] for i in order[:args.train_clips]]
            test = [paths[i] for i in order[args.train_clips:]]
        actions = np.concatenate([load_embodiment(p, REGISTRY["b1"])["actions"] for p in train])
        mean = actions.mean(0).astype(np.float32)
        std = np.maximum(actions.std(0), 1e-6).astype(np.float32)

        arms = [(f"pretrained/{m}", False, m) for m in args.z_modes]
        arms.append(("random", True, "real"))
        for name, random_backbone, z_mode in arms:
            rows.setdefault(name, [])
            cfg, itm, md = load_models(args.ckpt, random_backbone, device)
            xtr, ytr, rtr = build_features(encoder, itm, md, train, mean, std,
                                           cfg.action_lag, args.chunk, device, z_mode, args.seed)
            xte, yte, rte = build_features(encoder, itm, md, test, mean, std,
                                           cfg.action_lag, args.chunk, device, z_mode, args.seed)
            fit_head(md.heads[HEAD], xtr, ytr, xte, yte, args.epochs, args.lr,
                     args.weight_decay, args.seed)
            m = metrics(md.heads[HEAD], xte, yte, rte, mean, std)
            rows[name].append((m["deg"], m["r2"]))
            print(f"  split {split}  {name:<11} {m['deg']:7.2f} deg   R2 {m['r2']:+.3f}")
    del encoder

    print(f"\n{'model':<20}{'deg':>10}{'R2':>9}")
    summary = {}
    for name in rows:
        deg = np.array([d for d, _ in rows[name]])
        r2 = np.array([r for _, r in rows[name]])
        summary[name] = deg.mean()
        print(f"{name:<20}{deg.mean():10.2f}{r2.mean():+9.3f}   "
              f"(+/- {deg.std():.2f})")
    margin = summary["random"] / summary["pretrained/real"]
    print(f"\nmargin over a random backbone: {margin:.2f}x")
    print("The 4-leg body scores 2.85x on this protocol. A margin near 1.0x here means the insect "
          "features\ncarry nothing to the quadruped, which is what F41b and F45 predict.")


if __name__ == "__main__":
    main()
