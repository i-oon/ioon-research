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

    # How much capacity sits between the latent and the joint command: "mlp" is the original
    # decoder, "linear" keeps the cross-attention backbone with a single output projection,
    # "probe" removes the backbone entirely and linearly maps mean-pooled tokens plus z.
    # Capacity helps on the training bodies and hurts on a held-out one (FINDINGS.md F4b).
    md_head: str = "mlp"
    md_pool: int = 2

    lambda_recon: float = 1.0
    lambda_motion: float = 1.0

    # Decode one body's latent against a different body's frame, supervised by that body's
    # command. Every body walks the same expert episodes, so at a given timestep they share the
    # intent and differ only in geometry. Reading the body out of z gives the wrong answer here,
    # which is the point: the loss finally asks for the appearance-to-morphology mapping that
    # transfer needs, rather than merely permitting it. Off by default. See OPEN_QUESTION.md Q7.
    lambda_cross: float = 0.0

    # Two independently augmented views of each pair, which is what stops the ITM smuggling
    # x_{t+1} into z. Measured cost: the FTM's target becomes 4.39x more augmentation noise than
    # signal, so L_recon takes 99% of the gradient while chasing something unpredictable, and z
    # is never pressured to become an action. False falls back to the dimensional bottleneck,
    # z at 64 numbers against e_{t+1}'s 359,000. See FINDINGS.md F25.
    cross_augment: bool = True

    # Gradient-reversal head that removes body identity from z. Off by default, and superseded
    # by lambda_cross: stripping the code moved the decoder onto the frame and made transfer
    # worse, because it has no reason to read morphology there. See wm/models/adversary.py.
    lambda_adv: float = 0.0
    adv_hidden: int = 128
    # Ramp the reversal strength from 0 to 1 over this many epochs. Pushing adversarially while
    # reconstruction is still falling steeply drives the classifier below chance, which means the
    # latent is shuffling the body code faster than the classifier tracks it rather than dropping
    # it: a 5-epoch smoke run reached 0.009 against a 0.200 chance level. 0 disables the ramp.
    adv_warmup_epochs: int = 5

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    grad_clip: float = 1.0
    num_workers: int = 4

    device: str = "cuda"
    seed: int = 0
    out_dir: str = "wm/runs"
