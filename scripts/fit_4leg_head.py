"""Few-shot new-head test for the 4-leg middle-loss stick insect.

Stage 2 has trained heads for:
  hexapod -> 18-D stick-insect actions
  b1      -> 12-D Unitree B1 actions

The 4-leg stick insect is also 12-D, but those coordinates are not B1 coordinates.  This script
therefore freezes the Stage 2 visual/latent/motion backbone, adds a fresh `middleloss` output
head, fits only that head on a few real 4-leg clips, and scores held-out 4-leg clips.

The comparison is:
  pretrained  Stage 2 ITM + MotionDecoder backbone + new 4-leg head
  random      same architecture, random ITM/backbone + new 4-leg head

If the pretrained row beats the random row with the same number of clips, Stage 2 has learned
features that help a new embodiment beyond simply fitting a small action head.
"""
import argparse
import glob
import os
import sys
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip, upgrade_decoder_state  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402


def parse_list(text):
    if not text:
        return []
    return [int(x) for x in text.split(",") if x]


def paths_for(data_dir, episodes):
    out = []
    for ep in episodes:
        p = os.path.join(data_dir, f"middleloss_ep{ep}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        out.append(p)
    return out


def all_paths(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "middleloss_ep*.npz")))
    if not paths:
        raise SystemExit(f"no middleloss_ep*.npz found in {data_dir}")
    return paths


def action_stats(paths, eps=1e-6):
    actions = [np.load(p)["actions"].astype(np.float32) for p in paths]
    arr = np.concatenate(actions, axis=0)
    return arr.mean(axis=0).astype(np.float32), np.maximum(arr.std(axis=0), eps).astype(np.float32)


@torch.no_grad()
def build_features(encoder, itm, md, paths, mean, std, action_lag, chunk, device,
                   z_mode="real", seed=0):
    """Return frozen decoder features for a set of clips.

    z_mode:
      real      ITM(e_t, e_{t+1})
      zero      all-zero latent, testing whether the current frame/backbone alone is enough
      shuffled  real latents randomly permuted within each clip, preserving the marginal z
                distribution but breaking frame-transition alignment
    """
    xs, ys, ys_raw = [], [], []
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for path in paths:
        clip = np.load(path)
        e = encode_clip(encoder, clip["frames"], chunk).to(device)
        actions = torch.tensor(clip["actions"].astype(np.float32), device=device)
        n = min(len(e) - 1, len(actions) - action_lag)
        z_parts = []
        for s in range(0, n, chunk):
            t = min(s + chunk, n)
            e_t, e_next = e[s:t], e[s + 1:t + 1]
            z_parts.append(itm(e_t, e_next))
        z_all = torch.cat(z_parts, dim=0)
        if z_mode == "zero":
            z_all = torch.zeros_like(z_all)
        elif z_mode == "shuffled":
            z_all = z_all[torch.randperm(len(z_all), generator=generator, device=device)]
        elif z_mode != "real":
            raise ValueError(f"unknown z_mode {z_mode!r}")
        for s in range(0, n, chunk):
            t = min(s + chunk, n)
            e_t = e[s:t]
            z = z_all[s:t]
            x = md.features(e_t, z).squeeze(1)
            y_raw = actions[s + action_lag:t + action_lag]
            xs.append(x.cpu())
            ys.append(((y_raw - mean_t) / std_t).cpu())
            ys_raw.append(y_raw.cpu())
    return torch.cat(xs), torch.cat(ys), torch.cat(ys_raw)


def fit_head(head, x_train, y_train, x_val, y_val, epochs, lr, weight_decay, seed):
    torch.manual_seed(seed)
    device = next(head.parameters()).device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, epochs + 1):
        head.train()
        pred = head(x_train.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            head.eval()
            with torch.no_grad():
                val = F.mse_loss(head(x_val.unsqueeze(1)).squeeze(1), y_val).item()
            if val < best["loss"]:
                best = {"loss": val, "state": {k: v.cpu().clone() for k, v in head.state_dict().items()},
                        "epoch": epoch}
    if best["state"] is not None:
        head.load_state_dict(best["state"])
    return best


@torch.no_grad()
def predict_raw(head, x, mean, std):
    device = next(head.parameters()).device
    x = x.to(device)
    pred = head(x.unsqueeze(1)).squeeze(1)
    return pred * torch.tensor(std, device=device) + torch.tensor(mean, device=device)


@torch.no_grad()
def metrics(head, x, y, y_raw, mean, std):
    device = next(head.parameters()).device
    y, y_raw = y.to(device), y_raw.to(device)
    pred_raw = predict_raw(head, x, mean, std)
    pred = (pred_raw - torch.tensor(mean, device=device)) / torch.tensor(std, device=device)
    mse = F.mse_loss(pred, y).item()
    own_mean = ((y - y.mean(dim=0)) ** 2).mean().item()
    r2 = 1.0 - mse / max(own_mean, 1e-12)
    deg = float(np.rad2deg(torch.sqrt(((pred_raw - y_raw) ** 2).mean()).item()))
    return {"mse": mse, "own_mean": own_mean, "r2": r2, "deg": deg}


def load_models(ckpt, random_backbone, device):
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])
    # The checkpoint config is what matters, but inference here is tiny; keep modules wherever
    # the encoder features will be consumed.
    stats = checkpoint["action_stats"]
    heads = {k: len(v[0]) for k, v in stats.items()}
    itm = InverseTransitionModel(cfg).to(device).eval()
    md = MotionDecoder(cfg, heads=heads).to(device).eval()
    if not random_backbone:
        itm.load_state_dict(checkpoint["itm"])
        md.load_state_dict(upgrade_decoder_state(checkpoint["md"]))
    for p in itm.parameters():
        p.requires_grad_(False)
    for p in md.parameters():
        p.requires_grad_(False)
    md.add_head("middleloss", cfg.hidden, 12, device=device)
    # Train only the new head.
    for p in md.heads["middleloss"].parameters():
        p.requires_grad_(True)
    return cfg, itm, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/stage2_clean/best.pt")
    ap.add_argument("--data", default="data/ik_4leg_middleloss_clean9")
    ap.add_argument("--test_data", default="",
                    help="optional directory for held-out test clips; defaults to --data")
    ap.add_argument("--train_eps", default="144,28,198,93,22")
    ap.add_argument("--test_eps", default="")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encode_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--save_pred", default="",
                    help="optional output .npz containing held-out predicted and ground-truth actions")
    ap.add_argument("--z_ablation", action="store_true",
                    help="also fit pretrained heads with zero_z and shuffled_z features")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    test_dir = args.test_data or args.data
    test_dir = test_dir if os.path.isabs(test_dir) else os.path.join(ROOT, test_dir)
    train_eps = parse_list(args.train_eps)
    train_paths = paths_for(data_dir, train_eps)
    if args.test_eps:
        test_paths = paths_for(test_dir, parse_list(args.test_eps))
    else:
        train_set = {os.path.basename(p) for p in train_paths}
        test_paths = [p for p in all_paths(test_dir) if os.path.basename(p) not in train_set]
    if not test_paths:
        raise SystemExit("no held-out test clips; pass fewer --train_eps or explicit --test_eps")

    mean, std = action_stats(train_paths)
    print(f"checkpoint : {args.ckpt}")
    print(f"train clips: {[os.path.basename(p).replace('.npz','') for p in train_paths]}")
    print(f"test clips : {[os.path.basename(p).replace('.npz','') for p in test_paths]}")
    print(f"action stats from train clips only: dim={len(mean)}")

    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = []
    saved_payload = None
    feature_jobs = [("pretrained_stage2", False, "real"), ("random_backbone", True, "real")]
    if args.z_ablation:
        feature_jobs[1:1] = [
            ("pretrained_zero_z", False, "zero"),
            ("pretrained_shuffled_z", False, "shuffled"),
        ]
    for label, random_backbone, z_mode in feature_jobs:
        cfg, itm, md = load_models(args.ckpt, random_backbone, device)
        x_train, y_train, raw_train = build_features(
            encoder, itm, md, train_paths, mean, std, cfg.action_lag, args.chunk, device,
            z_mode=z_mode, seed=args.seed)
        x_test, y_test, raw_test = build_features(
            encoder, itm, md, test_paths, mean, std, cfg.action_lag, args.chunk, device,
            z_mode=z_mode, seed=args.seed + 1)
        best = fit_head(md.heads["middleloss"], x_train, y_train, x_test, y_test,
                        args.epochs, args.lr, args.weight_decay, args.seed)
        train_m = metrics(md.heads["middleloss"], x_train, y_train, raw_train, mean, std)
        test_m = metrics(md.heads["middleloss"], x_test, y_test, raw_test, mean, std)
        rows.append((label, best, train_m, test_m))
        if label == "pretrained_stage2":
            pred_raw = predict_raw(md.heads["middleloss"], x_test, mean, std).cpu().numpy()
            saved_payload = {
                "pred_flat": pred_raw.astype(np.float32),
                "gt_flat": raw_test.cpu().numpy().astype(np.float32),
                "test_paths": np.array(test_paths),
                "train_paths": np.array(train_paths),
                "mean": mean,
                "std": std,
                "action_lag": np.array(cfg.action_lag),
            }

    print("\n=== 4-leg middle-loss few-shot new-head ===")
    print(f"{'model':<20} {'best_ep':>7} {'train_deg':>10} {'test_deg':>9} "
          f"{'test_mse':>9} {'own_mean':>9} {'R2':>7}")
    for label, best, train_m, test_m in rows:
        print(f"{label:<20} {best['epoch']:7d} {train_m['deg']:10.2f} {test_m['deg']:9.2f} "
              f"{test_m['mse']:9.4f} {test_m['own_mean']:9.4f} {test_m['r2']:+7.2f}")

    print("\nInterpretation: the pretrained row must beat the random row on held-out clips to count "
          "as evidence that Stage 2 transfers useful features to the new 4-leg embodiment.")
    if args.save_pred:
        if saved_payload is None:
            raise RuntimeError("no pretrained predictions were saved")
        lengths = []
        pred_parts, gt_parts, clip_names = [], [], []
        cursor = 0
        for path in test_paths:
            # Features contain one target per transition, with action_lag targets skipped at
            # the beginning.  Store per-clip sequences so the renderer can replay clip N.
            n = len(np.load(path)["actions"]) - cfg.action_lag
            pred_parts.append(saved_payload["pred_flat"][cursor:cursor + n])
            gt_parts.append(saved_payload["gt_flat"][cursor:cursor + n])
            lengths.append(n)
            clip_names.append(os.path.basename(path).replace(".npz", ""))
            cursor += n
        out_dir = os.path.dirname(args.save_pred)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            args.save_pred,
            pred=np.concatenate(pred_parts, axis=0).astype(np.float32),
            gt=np.concatenate(gt_parts, axis=0).astype(np.float32),
            lengths=np.array(lengths, dtype=np.int64),
            clips=np.array(clip_names),
            morph=np.array("middleloss"),
            held_out=np.array(True),
            epoch=np.array(-1),
            action_lag=np.array(cfg.action_lag),
            train_paths=saved_payload["train_paths"],
            test_paths=saved_payload["test_paths"],
            mean=saved_payload["mean"],
            std=saved_payload["std"],
        )
        print(f"\nsaved pretrained held-out predictions -> {args.save_pred}")


if __name__ == "__main__":
    main()
