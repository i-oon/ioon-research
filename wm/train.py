"""Train ITM + FTM + Motion Decoder on the IK forward-walk dataset.

The V-JEPA2 encoder stays frozen inside the loop because cross-augmentation needs fresh
random views each epoch, so embeddings cannot be cached. Training uses the morphologies in
cfg.train_morphs; the held-out body is never seen here.

Run from the repository root:
  .venv/bin/python3 -m wm.train --epochs 50 --batch_size 8 --name stage1_100ep_clean
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
from wm.data.dataset import (  # noqa: E402
    EmbodimentBatchSampler,
    IKWalkPairs,
    MultiEmbodimentPairs,
    available_episodes,
    clip_paths,
    embodiment_split,
    load_clip,
)
from wm.data.embodiment import REGISTRY  # noqa: E402
from wm.losses import compute_losses  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.adversary import MorphAdversary, MorphProbe  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

VIEW_KEYS = ("view1_t", "view1_next", "view2_t", "view2_next")


def encode_batch(encoder, batch):
    frames = []
    for key in VIEW_KEYS:
        frames.extend(list(batch[key].numpy()))
    embeddings = encoder.encode(frames).float()
    return dict(zip(VIEW_KEYS, embeddings.chunk(len(VIEW_KEYS))))


def adv_scale(cfg, epoch):
    """Reversal strength for this epoch, ramped linearly over cfg.adv_warmup_epochs."""
    if cfg.adv_warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / cfg.adv_warmup_epochs)


def forward_step(models, encoder, batch, cfg, device, scale=1.0):
    views = encode_batch(encoder, batch)
    action = batch["action"].to(device)
    # batches are single-embodiment by construction, so one head serves the whole batch
    embodiment = batch["embodiment"][0] if "embodiment" in batch else "default"

    z = models["itm"](views["view1_t"], views["view1_next"])
    pred_next = models["ftm"](views["view2_t"], z)
    pred_action = models["md"](views["view1_t"], z, embodiment)

    cross_pred = cross_target = None
    if "cross_x_t" in batch and cfg.lambda_cross > 0:
        cross_views = encoder.encode(list(batch["cross_x_t"].numpy())).float()
        cross_pred = models["md"](cross_views, z, embodiment)
        cross_target = batch["cross_action"].to(device)

    adv_logits = probe_logits = morph_id = None
    if "morph_id" in batch:
        morph_id = batch["morph_id"].to(device)
        if "adv" in models:
            adv_logits = models["adv"](z, scale)
        if "probe" in models:
            probe_logits = models["probe"](z)
    return compute_losses(pred_next, views["view2_next"], pred_action, action, cfg,
                          adv_logits, morph_id, probe_logits, cross_pred, cross_target)


def run_epoch(models, encoder, loader, cfg, device, optimizer=None, scaler=None, scale=1.0):
    training = optimizer is not None
    for model in models.values():
        model.train(training)

    totals, count = {}, 0
    for batch in loader:
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, parts = forward_step(models, encoder, batch, cfg, device, scale)

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
    over-specialisation: in stage1_100ep_clipped it read 0.0016 while the held-out body degraded to 0.42.
    Evaluation needs no augmentation, so the embeddings are constant and can be cached;
    each epoch then costs only an ITM and decoder pass.
    """
    start, stop = cfg.frame_start, cfg.frame_stop
    all_paths = clip_paths(data_dir, (cfg.heldout_morph,))
    per_clip = max(1, (stop or 66) - start - 1)
    n_clips = min(len(all_paths), max(1, -(-cfg.heldout_pairs // per_clip)))
    paths = all_paths[:n_clips]
    embeddings, actions = [], []
    for path in paths:
        clip = load_clip(path)
        frames = clip["frames"][start:stop or None]
        parts = [encoder.encode(list(frames[i:i + 8])).float() for i in range(0, len(frames), 8)]
        embeddings.append(torch.cat(parts))
        actions.append(clip["actions"][start:stop or None][:-1])
    return {
        "embeddings": [e.to(device) for e in embeddings],
        "actions": [torch.tensor(a, device=device) for a in actions],
    }


@torch.no_grad()
def evaluate_heldout(models, cache, mean, std, device, chunk=8, action_lag=1):
    """Motion error on the held-out body, with each of the decoder's two inputs ablated.

    Both ablations are needed because they answer different questions. Zeroing z removes the
    gait phase along with anything else it carries, so a larger gap says the decoder needs z,
    not that z carries the body. Zeroing the frame is the one that tracks whether the decoder
    reads morphology from pixels: crossing the two inputs shows it currently takes the body
    from z and ignores the frame, and an intervention meant to reverse that should show up
    here as the frame becoming harder to do without.

    Chunked: a whole clip at once would run 512-token attention over ~65 transitions, which
    does not fit alongside the training step's own allocations.
    """
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    errors, no_z, no_x = [], [], []
    for embeddings, actions in zip(cache["embeddings"], cache["actions"]):
        target = (actions - mean_t) / std_t
        total = len(embeddings) - max(1, action_lag)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            e_t, e_next = embeddings[start:stop], embeddings[start + 1:stop + 1]
            expected = target[start + action_lag:stop + action_lag]
            z = models["itm"](e_t, e_next)
            errors.append(((models["md"](e_t, z) - expected) ** 2).mean().item())
            no_z.append(((models["md"](e_t, torch.zeros_like(z)) - expected) ** 2).mean().item())
            no_x.append(((models["md"](torch.zeros_like(e_t), z) - expected) ** 2).mean().item())
    return float(np.mean(errors)), float(np.mean(no_z)), float(np.mean(no_x))


def build_models(cfg, device, heads=None, n_bodies=0):
    models = {
        "itm": InverseTransitionModel(cfg).to(device),
        "ftm": ForwardTransitionModel(cfg).to(device),
        "md": MotionDecoder(cfg, heads=heads).to(device),
    }
    if n_bodies >= 2:
        # always on: a pure read-out of how decodable the body is from z, costing one small MLP
        models["probe"] = MorphProbe(cfg.z_dim, n_bodies, cfg.adv_hidden).to(device)
        if cfg.lambda_adv > 0:
            models["adv"] = MorphAdversary(cfg.z_dim, n_bodies, cfg.adv_hidden).to(device)
    elif cfg.lambda_adv > 0:
        raise ValueError("lambda_adv needs at least two training bodies to discriminate")
    return models


def build_cross_embodiment(cfg, root):
    """Datasets, sampler and decoder heads for training across embodiments."""
    specs = [tuple(s.split("=", 1)) for s in cfg.sources]
    train_sources, val_sources = embodiment_split(specs, cfg.val_fraction, root)
    train_set = MultiEmbodimentPairs(train_sources, seed=cfg.seed,
                                     cross_augment=cfg.cross_augment, action_lag=cfg.action_lag)
    val_set = MultiEmbodimentPairs(val_sources, stats=train_set.stats, seed=cfg.seed,
                                   cross_augment=cfg.cross_augment, action_lag=cfg.action_lag)
    heads = {name: REGISTRY[name].action_dim for name, _ in specs}
    return train_set, val_set, heads


def _bool(text):
    """argparse with type=bool turns any non-empty string into True, so "--flag false" would
    silently switch the flag on. Every boolean field needs this instead."""
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def parse_args(cfg):
    parser = argparse.ArgumentParser()
    for name, value in asdict(cfg).items():
        if isinstance(value, tuple):
            parser.add_argument(f"--{name}", nargs="+", default=list(value))
        elif isinstance(value, bool):
            parser.add_argument(f"--{name}", type=_bool, default=value)
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
    cross_embodiment = bool(cfg.sources)

    if cross_embodiment:
        train_set, val_set, heads = build_cross_embodiment(cfg, ROOT)
        train_sampler = EmbodimentBatchSampler(train_set, cfg.batch_size, True, cfg.seed)
        val_sampler = EmbodimentBatchSampler(val_set, cfg.batch_size, False, cfg.seed)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=cfg.num_workers)
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=cfg.num_workers)
        counts = {k: len(v) for k, v in train_set.embodiment_indices().items()}
        print(f"cross-embodiment: {counts} | heads {heads}")
    else:
        heads, train_sampler = None, None
        episodes = available_episodes(data_dir, cfg.train_morphs)
        if len(episodes) <= cfg.val_episodes:
            raise ValueError(f"need more than {cfg.val_episodes} episodes, found {episodes}")
        val_episodes = episodes[-cfg.val_episodes:]
        train_episodes = episodes[:-cfg.val_episodes]

        frame_range = (cfg.frame_start, cfg.frame_stop)
        train_set = IKWalkPairs(data_dir, cfg.train_morphs, train_episodes, seed=cfg.seed,
                                frame_range=frame_range, within_body_std=cfg.within_body_std,
                                cross_augment=cfg.cross_augment, action_lag=cfg.action_lag)
        val_set = IKWalkPairs(
            data_dir, cfg.train_morphs, val_episodes,
            mean=train_set.mean, std=train_set.std, seed=cfg.seed, frame_range=frame_range,
            cross_augment=cfg.cross_augment, action_lag=cfg.action_lag,
        )
        print(f"train episodes {train_episodes} | val episodes {val_episodes}")
        loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=False)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    encoder = VJEPA2FrameEncoder(device=str(device))
    n_bodies = len(getattr(train_set, "morphs", []) or [])
    models = build_models(cfg, device, heads=heads, n_bodies=n_bodies)
    if "adv" in models:
        print(f"adversary on z: {n_bodies} bodies {train_set.morphs}, lambda_adv {cfg.lambda_adv}")
    parameters = [p for model in models.values() for p in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = os.path.join(ROOT, cfg.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "summary"))
    trainable = sum(p.numel() for p in parameters)
    # the held-out-body monitor is specific to the single-morphology setting; cross-embodiment
    # transfer is measured separately, per embodiment, in wm.evaluate
    heldout_cache = None if cross_embodiment else cache_heldout(encoder, data_dir, cfg, device)
    heldout_note = (
        "heldout n/a (cross-embodiment)" if cross_embodiment
        else f"heldout '{cfg.heldout_morph}' {sum(len(a) for a in heldout_cache['actions'])} pairs"
    )
    print(f"train {len(train_set)} pairs | val {len(val_set)} pairs | "
          f"{heldout_note} | trainable {trainable/1e6:.1f}M")

    def checkpoint(epoch):
        state = {
            "config": asdict(cfg),
            "epoch": epoch,
            **{name: model.state_dict() for name, model in models.items()},
        }
        if cross_embodiment:
            state["action_stats"] = train_set.stats
        else:
            state["action_mean"] = train_set.mean
            state["action_std"] = train_set.std
        return state

    # tracked separately: total is ~99% reconstruction, so selecting on it alone would
    # ignore the motion term that grounds the latent in real joint commands
    best = {"total": float("inf"), "motion": float("inf")}
    for epoch in range(1, cfg.epochs + 1):
        train_set.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        scale = adv_scale(cfg, epoch)
        train_metrics = run_epoch(models, encoder, train_loader, cfg, device, optimizer, scaler, scale)
        val_metrics = run_epoch(models, encoder, val_loader, cfg, device, scale=scale)
        heldout = heldout_no_z = heldout_no_x = float("nan")
        if heldout_cache is not None:
            heldout, heldout_no_z, heldout_no_x = evaluate_heldout(
                models, heldout_cache, train_set.mean, train_set.std, device,
                action_lag=cfg.action_lag,
            )
        schedule.step()

        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        if heldout_cache is not None:
            writer.add_scalar("heldout/motion", heldout, epoch)
            writer.add_scalar("heldout/motion_zero_z", heldout_no_z, epoch)
            writer.add_scalar("heldout/motion_zero_x", heldout_no_x, epoch)
        writer.add_scalar("lr", schedule.get_last_lr()[0], epoch)
        suffix = (
            "" if heldout_cache is None
            else (f" | heldout {cfg.heldout_morph} {heldout:.4f} "
                  f"(zero_z {heldout_no_z:.4f} zero_x {heldout_no_x:.4f})")
        )
        print(
            f"epoch {epoch:3d} | train {train_metrics['total']:.4f} "
            f"(recon {train_metrics['recon']:.4f} motion {train_metrics['motion']:.4f}) | "
            f"val {val_metrics['total']:.4f} "
            f"(recon {val_metrics['recon']:.4f} motion {val_metrics['motion']:.4f})"
            + (f" | cross {train_metrics['cross']:.4f}"
               if "cross" in train_metrics else "")
            + (f" | adv {train_metrics['adv_accuracy']:.3f} (x{scale:.2f})"
               if "adv_accuracy" in train_metrics else "")
            + (f" probe {train_metrics['probe_accuracy']:.3f}"
               if "probe_accuracy" in train_metrics else "") + suffix
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
