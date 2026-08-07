"""Hyperparameters for the latent-action world model.

Architecture values follow LAC-WM (Table 4 / Section 3.1). Optimisation values are
scaled down for a single 2080 Ti; the paper used 64 H200s with batch size 512.
"""
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data/ik_walk_all6"
    train_morphs: tuple = ("long", "short")
    heldout_morph: str = "medium"
    val_episodes: int = 1
    # Sized in transitions, not clips: a clip yields (frames_used - 1) transitions, so a fixed
    # clip count silently shrinks the probe when frame_start/frame_stop narrow the range. At 80
    # transitions the held-out MSE swung 0.14-0.34 between epochs from sampling noise alone.
    heldout_pairs: int = 400
    checkpoint_every: int = 2
    # Frames before ~45 have the robot partly outside the right edge: the fixed camera is
    # anchored to the start pose and the robot walks into view. 0 keeps every frame.
    frame_start: int = 0
    frame_stop: int = 0

    # Scale the motion target by how much each joint moves within a body rather than by the
    # spread of the training bodies pooled together. Pooling counts the posture gap between
    # bodies as amplitude and drops the coxa-femur and femur-tibia weight to ~0.12 of
    # thorax-coxa; see action_stats in wm/data/dataset.py. False reproduces runs recorded
    # before 2026-08-08.
    within_body_std: bool = True

    # Cross-embodiment mode. Empty keeps the single-morphology path above; otherwise give
    # "name=dir" per embodiment, e.g. hexapod=data/ik_walk_100 b1=data/b1. Batches are drawn
    # from one embodiment at a time and routed to that embodiment's decoder head.
    sources: tuple = ()
    val_fraction: float = 0.1

    token_dim: int = 1408
    grid: int = 16
    action_dim: int = 18

    z_dim: int = 64
    hidden: int = 512
    heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    itm_self_blocks: int = 2
    itm_cross_blocks: int = 2
    ftm_blocks: int = 8
    z_tokens: int = 1
    md_pool: int = 2

    lambda_recon: float = 1.0
    lambda_motion: float = 1.0

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    grad_clip: float = 1.0
    num_workers: int = 4

    device: str = "cuda"
    seed: int = 0
    out_dir: str = "wm/runs"
