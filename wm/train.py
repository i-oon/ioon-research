"""Train ITM + FTM + Motion Decoder on the IK forward-walk dataset.

The V-JEPA2 encoder stays frozen inside the loop because cross-augmentation needs fresh
random views each epoch, so embeddings cannot be cached. Training uses the morphologies in
cfg.train_morphs; the held-out body is never seen here.

Run from the repository root:
  .venv/bin/python3 -m wm.train --epochs 50 --batch_size 8 --name wm_v1
"""
import argparse
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config  # noqa: E402
from wm.data.dataset import IKWalkPairs, available_episodes, clip_paths, load_clip  # noqa: E402
from wm.losses import compute_losses  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

VIEW_KEYS = ("view1_t", "view1_next", "view2_t", "view2_next")


def encode_batch(encoder, batch):
    frames = []
    for key in VIEW_KEYS:
        frames.extend(list(batch[key].numpy()))
    embeddings = encoder.encode(frames).float()
    return dict(zip(VIEW_KEYS, embeddings.chunk(len(VIEW_KEYS))))


def forward_step(models, encoder, batch, cfg, device):
    views = encode_batch(encoder, batch)
    action = batch["action"].to(device)

    z = models["itm"](views["view1_t"], views["view1_next"])
    pred_next = models["ftm"](views["view2_t"], z)
    pred_action = models["md"](views["view1_t"], z)
    return compute_losses(pred_next, views["view2_next"], pred_action, action, cfg)


def run_epoch(models, encoder, loader, cfg, device, optimizer=None, scaler=None):
    training = optimizer is not None
    for model in models.values():
        model.train(training)

    totals, count = {}, 0
    for batch in loader:
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, parts = forward_step(models, encoder, batch, cfg, device)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for model in models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def cache_heldout(encoder, data_dir, cfg, device):
    """Embed a fixed slice of the held-out body once.

    Validation on unseen episodes of the *training* bodies cannot see cross-body
    over-specialisation: in wm_v2 it read 0.0016 while the held-out body degraded to 0.42.
    Evaluation needs no augmentation, so the embeddings are constant and can be cached;
    each epoch then costs only an ITM and decoder pass.
    """
    paths = clip_paths(data_dir, (cfg.heldout_morph,))[:cfg.heldout_clips]
    embeddings, actions = [], []
    for path in paths:
        clip = load_clip(path)
        frames = clip["frames"]
        parts = [encoder.encode(list(frames[i:i + 8])).float() for i in range(0, len(frames), 8)]
        embeddings.append(torch.cat(parts))
        actions.append(clip["actions"][:-1])
    return {
        "embeddings": [e.to(device) for e in embeddings],
        "actions": [torch.tensor(a, device=device) for a in actions],
    }


@torch.no_grad()
def evaluate_heldout(models, cache, mean, std, device):
    """Motion error on the held-out body, with the latent ablated as a reference."""
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    errors, ablated = [], []
    for embeddings, actions in zip(cache["embeddings"], cache["actions"]):
        e_t, e_next = embeddings[:-1], embeddings[1:]
        target = (actions - mean_t) / std_t
        z = models["itm"](e_t, e_next)
        errors.append(((models["md"](e_t, z) - target) ** 2).mean().item())
        ablated.append(((models["md"](e_t, torch.zeros_like(z)) - target) ** 2).mean().item())
    return float(np.mean(errors)), float(np.mean(ablated))


def build_models(cfg, device):
    return {
        "itm": InverseTransitionModel(cfg).to(device),
        "ftm": ForwardTransitionModel(cfg).to(device),
        "md": MotionDecoder(cfg).to(device),
    }


def parse_args(cfg):
    parser = argparse.ArgumentParser()
    for name, value in asdict(cfg).items():
        if isinstance(value, tuple):
            parser.add_argument(f"--{name}", nargs="+", default=list(value))
        else:
            parser.add_argument(f"--{name}", type=type(value), default=value)
    parser.add_argument("--name", type=str, default="wm")
    return parser.parse_args()


def main():
    args = parse_args(Config())
    cfg = Config(**{k: v for k, v in vars(args).items() if k != "name"})
    cfg.train_morphs = tuple(cfg.train_morphs)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    episodes = available_episodes(data_dir, cfg.train_morphs)
    if len(episodes) <= cfg.val_episodes:
        raise ValueError(f"need more than {cfg.val_episodes} episodes, found {episodes}")
    val_episodes = episodes[-cfg.val_episodes:]
    train_episodes = episodes[:-cfg.val_episodes]

    train_set = IKWalkPairs(data_dir, cfg.train_morphs, train_episodes, seed=cfg.seed)
    val_set = IKWalkPairs(
        data_dir, cfg.train_morphs, val_episodes,
        mean=train_set.mean, std=train_set.std, seed=cfg.seed,
    )
    print(f"train episodes {train_episodes} | val episodes {val_episodes}")

    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    encoder = VJEPA2FrameEncoder(device=str(device))
    models = build_models(cfg, device)
    parameters = [p for model in models.values() for p in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = os.path.join(ROOT, cfg.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "summary"))
    trainable = sum(p.numel() for p in parameters)
    heldout_cache = cache_heldout(encoder, data_dir, cfg, device)
    n_heldout = sum(len(a) for a in heldout_cache["actions"])
    print(f"train {len(train_set)} pairs | val {len(val_set)} pairs | "
          f"heldout '{cfg.heldout_morph}' {n_heldout} pairs | trainable {trainable/1e6:.1f}M")

    def checkpoint(epoch):
        return {
            "config": asdict(cfg),
            "epoch": epoch,
            "action_mean": train_set.mean,
            "action_std": train_set.std,
            **{name: model.state_dict() for name, model in models.items()},
        }

    # tracked separately: total is ~99% reconstruction, so selecting on it alone would
    # ignore the motion term that grounds the latent in real joint commands
    best = {"total": float("inf"), "motion": float("inf")}
    for epoch in range(1, cfg.epochs + 1):
        train_set.set_epoch(epoch)
        train_metrics = run_epoch(models, encoder, train_loader, cfg, device, optimizer, scaler)
        val_metrics = run_epoch(models, encoder, val_loader, cfg, device)
        heldout, heldout_ablated = evaluate_heldout(
            models, heldout_cache, train_set.mean, train_set.std, device
        )
        schedule.step()

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("heldout/motion", heldout, epoch)
        writer.add_scalar("heldout/motion_zero_z", heldout_ablated, epoch)
        writer.add_scalar("lr", schedule.get_last_lr()[0], epoch)
        print(
            f"epoch {epoch:3d} | train {train_metrics['total']:.4f} "
            f"(recon {train_metrics['recon']:.4f} motion {train_metrics['motion']:.4f}) | "
            f"val {val_metrics['total']:.4f} "
            f"(recon {val_metrics['recon']:.4f} motion {val_metrics['motion']:.4f}) | "
            f"heldout {cfg.heldout_morph} {heldout:.4f} (zero_z {heldout_ablated:.4f})"
        )

        for key, filename in (("total", "best.pt"), ("motion", "best_motion.pt")):
            if val_metrics[key] < best[key]:
                best[key] = val_metrics[key]
                torch.save(checkpoint(epoch), os.path.join(run_dir, filename))
        # periodic snapshots: selecting on the held-out body would leak it, but without
        # snapshots a good epoch cannot be recovered without retraining
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            torch.save(checkpoint(epoch), os.path.join(run_dir, f"epoch{epoch:03d}.pt"))
    writer.close()
    print(f"best val total {best['total']:.4f} | best val motion {best['motion']:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
