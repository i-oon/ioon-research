"""Few-shot curve for the 4-leg middle-loss new-head test.

This answers a sample-efficiency question: does the Stage 2 backbone reduce how much data a new
embodiment-specific action head needs, compared with the same head fitted on a random backbone?
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from fit_4leg_head import (  # noqa: E402
    action_stats, all_paths, fit_head, load_models, metrics, parse_list,
)


def episode(path):
    return int(os.path.basename(path).split("_ep", 1)[1].split(".", 1)[0])


@torch.no_grad()
def encode_all(encoder, paths, chunk):
    from wm.evaluate import encode_clip

    cache = {}
    for p in paths:
        clip = np.load(p)
        cache[p] = {
            "e": encode_clip(encoder, clip["frames"], chunk).cpu(),
            "actions": torch.tensor(clip["actions"].astype(np.float32)),
        }
    return cache


@torch.no_grad()
def features_from_cache(cache, itm, md, paths, mean, std, action_lag, chunk, device):
    xs, ys, raw = [], [], []
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    for p in paths:
        e = cache[p]["e"].to(device)
        actions = cache[p]["actions"].to(device)
        n = min(len(e) - 1, len(actions) - action_lag)
        for s in range(0, n, chunk):
            t = min(s + chunk, n)
            z = itm(e[s:t], e[s + 1:t + 1])
            xs.append(md.features(e[s:t], z).squeeze(1).cpu())
            y_raw = actions[s + action_lag:t + action_lag]
            raw.append(y_raw.cpu())
            ys.append(((y_raw - mean_t) / std_t).cpu())
    return torch.cat(xs), torch.cat(ys), torch.cat(raw)


def choose(paths, budget, seed):
    rng = np.random.default_rng(seed)
    order = np.array(paths, dtype=object)
    rng.shuffle(order)
    train = sorted(order[:budget].tolist(), key=episode)
    test = sorted(order[budget:].tolist(), key=episode)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/stage2_clean/best.pt")
    ap.add_argument("--data", default="data/ik_4leg_middleloss_clean9")
    ap.add_argument("--budgets", default="1,3,5,7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--encode_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out_csv", default="results/wm/4leg_head/fewshot_curve.csv")
    ap.add_argument("--out_fig", default="results/wm/figures/4leg_fewshot_curve.png")
    args = ap.parse_args()

    data_dir = args.data if os.path.isabs(args.data) else os.path.join(ROOT, args.data)
    paths = sorted(all_paths(data_dir), key=episode)
    budgets = parse_list(args.budgets)
    seeds = parse_list(args.seeds)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"clips: {[episode(p) for p in paths]}")
    print(f"budgets={budgets} seeds={seeds}")
    encoder = VJEPA2FrameEncoder(device=args.encode_device, dtype=torch.float32)
    cache = encode_all(encoder, paths, args.chunk)

    rows = []
    for budget in budgets:
        if budget >= len(paths):
            raise SystemExit(f"budget {budget} leaves no held-out clips; have {len(paths)}")
        for split_seed in seeds:
            train_paths, test_paths = choose(paths, budget, split_seed)
            mean, std = action_stats(train_paths)
            for label, random_backbone in [("pretrained_stage2", False), ("random_backbone", True)]:
                cfg, itm, md = load_models(args.ckpt, random_backbone, device)
                x_train, y_train, raw_train = features_from_cache(
                    cache, itm, md, train_paths, mean, std, cfg.action_lag, args.chunk, device)
                x_test, y_test, raw_test = features_from_cache(
                    cache, itm, md, test_paths, mean, std, cfg.action_lag, args.chunk, device)
                best = fit_head(md.heads["middleloss"], x_train, y_train, x_test, y_test,
                                args.epochs, args.lr, args.weight_decay, split_seed)
                train_m = metrics(md.heads["middleloss"], x_train, y_train, raw_train, mean, std)
                test_m = metrics(md.heads["middleloss"], x_test, y_test, raw_test, mean, std)
                row = {
                    "budget": budget,
                    "seed": split_seed,
                    "model": label,
                    "best_epoch": best["epoch"],
                    "train_deg": train_m["deg"],
                    "test_deg": test_m["deg"],
                    "test_mse": test_m["mse"],
                    "own_mean": test_m["own_mean"],
                    "r2": test_m["r2"],
                    "train_eps": ",".join(map(str, map(episode, train_paths))),
                    "test_eps": ",".join(map(str, map(episode, test_paths))),
                }
                rows.append(row)
                print(f"budget {budget} seed {split_seed} {label:<18} "
                      f"test {test_m['deg']:.2f} deg R2 {test_m['r2']:+.2f}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {args.out_csv}")

    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for label, color in [("pretrained_stage2", "#2a9d8f"), ("random_backbone", "#e76f51")]:
        means, stds = [], []
        for b in budgets:
            vals = [r["test_deg"] for r in rows if r["budget"] == b and r["model"] == label]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        ax.errorbar(budgets, means, yerr=stds, marker="o", linewidth=2.2,
                    capsize=4, label=label.replace("_", " "), color=color)
    ax.set_xlabel("4-leg clips used to fit the new head")
    ax.set_ylabel("held-out error (deg / joint)")
    ax.set_title("Few-shot 4-leg head calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.savefig(args.out_fig, dpi=180)
    print(f"-> {args.out_fig}")


if __name__ == "__main__":
    main()
