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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import Config, chunk_of  # noqa: E402
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


def encode_batch(encoder, batch, offset=None):
    frames = []
    for key in VIEW_KEYS:
        frames.extend(list(batch[key].numpy()))
    embeddings = encoder.encode(frames).float()
    if offset is not None:
        # broadcast over batch and patch tokens: one 1408-vector removed from every token, so the
        # appearance offset goes and the spatial arrangement of the robot stays
        embeddings = embeddings - offset
    return dict(zip(VIEW_KEYS, embeddings.chunk(len(VIEW_KEYS))))


@torch.no_grad()
def embedding_offsets(encoder, dataset, cfg, device):
    """Per-embodiment mean of the encoder's output, one vector of `token_dim` per embodiment.

    Estimated from un-augmented frames, since the offset being removed is a property of how a
    robot renders rather than of any particular crop. Averaged over frames and over patch
    positions, so what is subtracted is a global appearance shift and not a template of where the
    robot usually sits in the frame -- subtracting per position would delete silhouette.
    """
    offsets = {}
    for name, indices in dataset.embodiment_indices().items():
        step = max(1, len(indices) // cfg.center_frames)
        picked = indices[::step][:cfg.center_frames]
        total, count = None, 0
        for start in range(0, len(picked), 8):
            frames = [dataset.frame_at(i) for i in picked[start:start + 8]]
            e = encoder.encode(frames).float()
            batch_sum = e.sum(dim=(0, 1))
            total = batch_sum if total is None else total + batch_sum
            count += e.shape[0] * e.shape[1]
        offsets[name] = (total / count).to(device)
        print(f"  {name}: mean over {len(picked)} frames, "
              f"norm {offsets[name].norm().item():.3f}")
    pair = list(offsets.values())
    if len(pair) == 2:
        print(f"  offset between the two embodiments: {(pair[0] - pair[1]).norm().item():.3f}")
    return offsets


def write_run_config(run_dir, cfg, name, train_set, val_set, cross_embodiment):
    """Write `config.yaml` beside the checkpoints, at startup.

    The config used to live only inside the checkpoint, so deleting a 370 MB file to free disk
    also destroyed the only record of what the run was: nine run directories are now empty while
    their numbers are still quoted in FINDINGS.md, unreproducible and undescribable. A few KB of
    YAML written up front costs nothing and survives.

    Written before training rather than after, so a run that crashes or is cancelled still leaves
    a record of what it was attempting.
    """
    record = {"name": name, "config": asdict(cfg)}
    if cross_embodiment:
        counts = {k: len(v) for k, v in train_set.embodiment_indices().items()}
        record["data"] = {
            "train_pairs": counts,
            "val_pairs": {k: len(v) for k, v in val_set.embodiment_indices().items()},
            "train_clips_per_body": _clip_counts(train_set),
            "val_clips_per_body": _clip_counts(val_set),
            "balance_ratio": (f"{max(counts.values()) / max(min(counts.values()), 1):.2f}:1"
                              if len(counts) > 1 else "n/a"),
        }
    else:
        record["data"] = {"train_pairs": len(train_set), "val_pairs": len(val_set),
                          "bodies": sorted(getattr(train_set, "morphs", []) or [])}
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(record, fh, sort_keys=False, default_flow_style=False)
    print(f"config -> {path}")


def _clip_counts(dataset):
    counts = {}
    for clip in getattr(dataset, "clips", []):
        key = f"{clip.get('embodiment', 'default')}/{clip.get('body', '?')}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def adv_scale(cfg, epoch):
    """Reversal strength for this epoch, ramped linearly over cfg.adv_warmup_epochs."""
    if cfg.adv_warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / cfg.adv_warmup_epochs)


def forward_step(models, encoder, batch, cfg, device, scale=1.0, offsets=None):
    # batches are single-embodiment by construction, so one head and one offset serve the batch
    embodiment = batch["embodiment"][0] if "embodiment" in batch else "default"
    offset = offsets.get(embodiment) if offsets else None
    views = encode_batch(encoder, batch, offset)
    action = batch["action"].to(device)

    z = models["itm"](views["view1_t"], views["view1_next"])
    pred_next = models["ftm"](views["view2_t"], z, embodiment)
    # `.chunk`, not `.__call__`: the decoder's public output is the first command only, so
    # every diagnostic keeps its 2-D contract while the loss sees the whole window. The reshape
    # is the guard -- it raises on a mismatch instead of broadcasting.
    pred_action = models["md"].chunk(views["view1_t"], z, embodiment).reshape(action.shape)

    cross_pred = cross_target = None
    if "cross_x_t" in batch and cfg.lambda_cross > 0:
        cross_views = encoder.encode(list(batch["cross_x_t"].numpy())).float()
        if offset is not None:
            # the partner body is a different morphology of the same embodiment, so it shares
            # this batch's offset
            cross_views = cross_views - offset
        cross_target = batch["cross_action"].to(device)
        cross_pred = models["md"].chunk(cross_views, z, embodiment).reshape(cross_target.shape)

    body_pred = body_target = None
    if models["md"].body_head is not None and "body_motion" in batch:
        # the same view and the same trunk the joint head just used
        body_pred = models["md"].body(views["view1_t"], z)
        body_target = batch["body_motion"].to(device)

    # --- ActSWM terms (F146, F151, F152). Off unless `lambda_hinge` or `lambda_readout` is set,
    # so every earlier run reproduces byte for byte.
    hinge = readout_loss = None
    if cfg.lambda_hinge > 0 or cfg.lambda_readout > 0:
        # **The null is `ITM(e_t, e_t)` -- the latent of "nothing happened".** F148's standing
        # stance is an *action*, and pretraining has no action projector to map it into `z`; a
        # hinge built on `proj(stance)` puts exactly zero gradient into `z`, measured on both
        # bodies (F151). The stance null remains the right one for anything measured through the
        # projector, which is a different stage. **Never compare the two stages' `/mean-z`.**
        z_null = models["itm"](views["view1_t"], views["view1_t"])
        real, null, seps = views["view2_t"], views["view2_t"], []
        for _ in range(max(1, cfg.hinge_K)):
            real = models["ftm"](real, z, embodiment)
            null = models["ftm"](null, z_null, embodiment)
            seps.append(1 - F.cosine_similarity(real.flatten(1), null.flatten(1), dim=1).mean())
        if cfg.lambda_hinge > 0:
            # margin 0.1, not ActSWM's 0.3: at 0.3 the term overshoots, switches itself off and
            # collapses -- 0.019, 0.137, 0.496, 0.008 with its gradient dying to 0.00006 (F151).
            # At 0.1 separation rises and holds on both bodies (F152).
            hinge = F.relu(cfg.hinge_margin - torch.stack(seps)).mean()
        if cfg.lambda_readout > 0 and "readout" in models:
            # the readout is frozen and randomly initialised: it cannot relocate the boundary it
            # scores, so the only way to lower this is to make the transitions separable (F150)
            readout_loss = F.mse_loss(models["readout"][embodiment](views["view2_t"], real),
                                      action.reshape(len(action), -1)[:, :models["readout"][
                                          embodiment].out_dim])

    adv_logits = probe_logits = morph_id = None
    if "morph_id" in batch:
        morph_id = batch["morph_id"].to(device)
        if "adv" in models:
            adv_logits = models["adv"](z, scale)
        if "probe" in models:
            probe_logits = models["probe"](z)
    loss, parts = compute_losses(pred_next, views["view2_next"], pred_action, action, cfg,
                                 adv_logits, morph_id, probe_logits, cross_pred, cross_target,
                                 body_pred, body_target)
    if hinge is not None:
        loss = loss + cfg.lambda_hinge * hinge
        parts["hinge"] = float(hinge)
        parts["separation"] = float(seps[-1])
    if readout_loss is not None:
        loss = loss + cfg.lambda_readout * readout_loss
        parts["readout"] = float(readout_loss)
    return loss, parts


def run_epoch(models, encoder, loader, cfg, device, optimizer=None, scaler=None, scale=1.0,
              offsets=None):
    training = optimizer is not None
    for model in models.values():
        model.train(training)

    totals, count = {}, 0
    for batch in loader:
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, parts = forward_step(models, encoder, batch, cfg, device, scale, offsets)

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
        # keep every command: with action_lag 1 the target for the last usable transition is
        # the final command, and dropping it here left the last chunk empty, which broadcast
        # silently and turned the whole held-out metric into nan
        actions.append(clip["actions"][start:stop or None])
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
        total = min(len(embeddings) - 1, len(target) - action_lag)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            e_t, e_next = embeddings[start:stop], embeddings[start + 1:stop + 1]
            expected = target[start + action_lag:stop + action_lag]
            assert len(expected) == stop - start, "held-out target and frames are misaligned"
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
    if cfg.lambda_readout > 0:
        # **A new module, never the ITM.** The ITM produces the `z` the action projector is later
        # fitted to imitate; freezing it at random weights would make that `z` arbitrary and break
        # every control-time path. This readout feeds nothing downstream -- its only job is to
        # route gradient into the forward model so transitions are genuinely separated.
        from scripts.diagnostics.check_actswm_wiring import FrozenActionReadout
        # a ModuleDict, not a plain dict: `run_epoch` calls `.train()` and `.parameters()` on
        # every entry of `models`, and a bare dict has neither
        import torch.nn as nn
        heads_spec = dict(heads or {})
        models["readout"] = nn.ModuleDict(
            {name: FrozenActionReadout(cfg.token_dim, int(dim), hidden=cfg.readout_hidden)
             for name, dim in heads_spec.items()}).to(device)
        for head in models["readout"].values():
            head.out_dim = head.net[-1].out_features
    return models


def build_cross_embodiment(cfg, root):
    """Datasets, sampler and decoder heads for training across embodiments."""
    specs = [tuple(s.split("=", 1)) for s in cfg.sources]
    train_sources, val_sources = embodiment_split(specs, cfg.val_fraction, root,
                                                  heldout_bodies=tuple(cfg.heldout_bodies),
                                                  clips_per_body=tuple(cfg.clips_per_body))
    train_set = MultiEmbodimentPairs(train_sources, seed=cfg.seed,
                                     cross_augment=cfg.cross_augment, action_lag=cfg.action_lag,
                                     body_channels=_channels(cfg), frame_stride=cfg.frame_stride,
                                     action_chunk=chunk_of(cfg))
    # body_stats too, not only the action stats: a validation split that centres body motion on
    # its own mean is scoring against a different target than the one being trained.
    val_set = MultiEmbodimentPairs(val_sources, stats=train_set.stats, seed=cfg.seed,
                                   cross_augment=cfg.cross_augment, action_lag=cfg.action_lag,
                                   body_stats=train_set.body_stats,
                                   body_channels=_channels(cfg), frame_stride=cfg.frame_stride,
                                   action_chunk=chunk_of(cfg))
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


def _channels(cfg):
    """`cfg.body_channels` as integers.

    `parse_args` builds tuple options with `nargs="+"` and no `type=`, so `--body_channels 0 2`
    arrives as the strings `["0", "2"]` and indexes nothing. A YAML config gives real integers.
    Both have to work.
    """
    return tuple(int(c) for c in cfg.body_channels)


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
    parser.add_argument("--resume", type=str, default="", metavar="auto|PATH",
                        help="continue from a run's last.pt. 'auto' looks in the run directory. "
                             "The epoch count must match the interrupted run, since the cosine "
                             "schedule anneals over the total.")
    return parser.parse_args()


def main():
    args = parse_args(Config())
    # --name and --resume are how the run is invoked, not part of what it is
    cfg = Config(**{k: v for k, v in vars(args).items() if k not in ("name", "resume")})
    cfg.train_morphs = tuple(cfg.train_morphs)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(ROOT, cfg.data_dir)
    cross_embodiment = bool(cfg.sources)

    if cross_embodiment:
        train_set, val_set, heads = build_cross_embodiment(cfg, ROOT)
        train_sampler = EmbodimentBatchSampler(train_set, cfg.batch_size, True, cfg.seed,
                                       balance=cfg.balance_embodiments)
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
                                cross_augment=cfg.cross_augment, action_lag=cfg.action_lag,
                                frame_stride=cfg.frame_stride, action_chunk=chunk_of(cfg))
        val_set = IKWalkPairs(
            data_dir, cfg.train_morphs, val_episodes,
            mean=train_set.mean, std=train_set.std, seed=cfg.seed, frame_range=frame_range,
            cross_augment=cfg.cross_augment, action_lag=cfg.action_lag,
            frame_stride=cfg.frame_stride, action_chunk=chunk_of(cfg),
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
    write_run_config(run_dir, cfg, args.name, train_set, val_set, cross_embodiment)
    writer = SummaryWriter(os.path.join(run_dir, "summary"))
    trainable = sum(p.numel() for p in parameters)
    # the held-out-body monitor is specific to the single-morphology setting; cross-embodiment
    # transfer is measured separately, per embodiment, in wm.evaluate
    heldout_cache = None if cross_embodiment else cache_heldout(encoder, data_dir, cfg, device)
    heldout_note = (
        "heldout n/a (cross-embodiment)" if cross_embodiment
        else f"heldout '{cfg.heldout_morph}' "
             f"{sum(len(a) - max(1, cfg.action_lag) for a in heldout_cache['actions'])} pairs"
    )
    print(f"train {len(train_set)} pairs | val {len(val_set)} pairs | "
          f"{heldout_note} | trainable {trainable/1e6:.1f}M")

    offsets = None
    if cfg.center_embeddings:
        if not cross_embodiment:
            raise SystemExit("center_embeddings needs --sources: there is one embodiment here, "
                             "and centring it removes a constant that changes nothing")
        print("per-embodiment appearance offset, subtracted before anything is trained on it:")
        offsets = embedding_offsets(encoder, train_set, cfg, device)

    def checkpoint(epoch):
        state = {
            "config": asdict(cfg),
            "epoch": epoch,
            **{name: model.state_dict() for name, model in models.items()},
        }
        if cross_embodiment:
            state["action_stats"] = train_set.stats
            if getattr(train_set, "body_stats", None) is not None:
                # every downstream script that decodes body motion has to undo the same
                # standardisation, and these statistics are pooled across embodiments rather than
                # per embodiment, so they cannot be recomputed from one robot's clips
                state["body_stats"] = train_set.body_stats
        else:
            state["action_mean"] = train_set.mean
            state["action_std"] = train_set.std
        if offsets is not None:
            # stored because every downstream script has to subtract the same vector: feeding a
            # centred model raw embeddings is a silent distribution shift, not an error
            state["embedding_offsets"] = {k: v.cpu() for k, v in offsets.items()}
        return state

    def training_state(epoch, best):
        """Everything needed to continue: weights plus the optimiser's momenta, the schedule's
        position, the AMP scaler, and which epoch we reached.

        Kept in its own `last.pt` and overwritten each epoch rather than added to every snapshot:
        AdamW carries two moments per parameter, so including it would take each of the periodic
        checkpoints from ~370 MB to over a gigabyte, and only the newest one is ever resumed from.
        """
        return {**checkpoint(epoch), "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(), "scaler": scaler.state_dict(), "best": best}

    # tracked separately: total is ~99% reconstruction, so selecting on it alone would
    # ignore the motion term that grounds the latent in real joint commands
    best = {"selection": float("inf"), "total": float("inf"), "motion": float("inf")}
    start_epoch = 1
    resume_path = os.path.join(run_dir, "last.pt") if args.resume == "auto" else args.resume
    if resume_path and os.path.exists(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["config"]["epochs"] != cfg.epochs:
            raise SystemExit(
                f"cannot resume: this run was {saved['config']['epochs']} epochs and you asked "
                f"for {cfg.epochs}. The cosine schedule anneals over the total, so continuing "
                f"with a different one restarts the learning rate partway through and the result "
                f"is neither run. Start fresh under a new --name.")
        for name, model in models.items():
            model.load_state_dict(saved[name])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        scaler.load_state_dict(saved["scaler"])
        best = saved["best"]
        # runs checkpointed before `selection` existed carry only total/motion
        best.setdefault("selection", float("inf"))
        start_epoch = saved["epoch"] + 1
        print(f"resuming {resume_path} at epoch {start_epoch} of {cfg.epochs} "
              f"(best so far total {best['total']:.4f}, motion {best['motion']:.4f})")
    elif args.resume and args.resume != "auto":
        raise SystemExit(f"--resume {args.resume}: not found")

    for epoch in range(start_epoch, cfg.epochs + 1):
        train_set.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        scale = adv_scale(cfg, epoch)
        train_metrics = run_epoch(models, encoder, train_loader, cfg, device, optimizer, scaler,
                                  scale, offsets)
        val_metrics = run_epoch(models, encoder, val_loader, cfg, device, scale=scale,
                                offsets=offsets)
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
            # `motion` above averages over the whole command window at action_chunk > 1, so it
            # is not comparable to a chunk-1 run. This is the column that is. Printed rather than
            # left in tensorboard because it is the number that says whether chunking worked.
            + (f" | motion@1 {train_metrics['motion_first']:.4f}/"
               f"{val_metrics['motion_first']:.4f}"
               if "motion_first" in train_metrics else "")
            + (f" | cross {train_metrics['cross']:.4f}"
               if "cross" in train_metrics else "")
            # Printed because it was not. `lambda_body 0.5` ran for 60 epochs contributing 0.002
            # of a 6.0 total, and the line above showed only recon and motion, so the term looked
            # absent rather than negligible. Any term that enters the loss has to enter this line.
            + (f" | body {train_metrics['body']:.4f}"
               if "body" in train_metrics else "")
            + (f" | adv {train_metrics['adv_accuracy']:.3f} (x{scale:.2f})"
               if "adv_accuracy" in train_metrics else "")
            + (f" probe {train_metrics['probe_accuracy']:.3f}"
               if "probe_accuracy" in train_metrics else "") + suffix
        )

        # `selection`, not `total` -- see wm/losses.py. `total` includes whichever experimental
        # term this run enables, so selecting on it checkpoints the arms of a matched pair at
        # different epochs and every comparison downstream inherits the mismatch.
        for key, filename in (("selection", "best.pt"), ("motion", "best_motion.pt")):
            if val_metrics[key] < best[key]:
                best[key] = val_metrics[key]
                torch.save(checkpoint(epoch), os.path.join(run_dir, filename))
        # periodic snapshots: selecting on the held-out body would leak it, but without
        # snapshots a good epoch cannot be recovered without retraining
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            torch.save(checkpoint(epoch), os.path.join(run_dir, f"epoch{epoch:03d}.pt"))
        # written every epoch, overwritten in place, so an interrupted run loses at most one
        torch.save(training_state(epoch, best), os.path.join(run_dir, "last.pt"))
    writer.close()
    # `total` is no longer tracked -- selection replaced it as the checkpoint criterion, and
    # printing an untouched inf reads as a failure
    print(f"best val selection {best['selection']:.4f} | motion {best['motion']:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
