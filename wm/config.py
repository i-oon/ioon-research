"""Hyperparameters for the latent-action world model.

Architecture values follow LAC-WM (Table 4 / Section 3.1). Optimisation values are
scaled down for a single 2080 Ti; the paper used 64 H200s with batch size 512.
"""
from dataclasses import dataclass, fields


def from_checkpoint(saved):
    """Rebuild a Config from a checkpoint, keeping the behaviour that checkpoint was trained with.

    A field added after a run was recorded must fall back to what that run actually did, not to
    the current default, or the checkpoint gets scored against a target it never saw.
    """
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in saved.items() if k in known})
    for name, before in LEGACY_DEFAULTS.items():
        if name not in saved:
            setattr(cfg, name, before)
    cfg.train_morphs = tuple(cfg.train_morphs)
    return cfg


# What each field did before it existed, for runs recorded without it.
LEGACY_DEFAULTS = {
    "action_lag": 0,
    "cross_augment": True,
    "within_body_std": False,
    "lambda_cross": 0.0,
    "lambda_adv": 0.0,
    "ftm_embodiment_channel": False,
    "center_embeddings": False,
    "heldout_bodies": (),
    "clips_per_body": (),
}


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

    # Bodies withheld from cross-embodiment training, so Stage 2 has a generalisation test at all.
    # Two embodiments cannot hold each other out -- removing one leaves one -- and `sources` takes
    # a directory and globs it, so without this every body is trained on and the only things
    # measurable are the latent's composition and a validation split of 67 B1 transitions.
    # Names must match the clip prefix, e.g. c08f09t09. Score them afterwards with
    # scripts/diagnostics/score_body.py, which needs no retraining.
    heldout_bodies: tuple = ()

    # Cap clips per body, as "embodiment=N". Without it the hexapod brings 6 bodies x 27 clips
    # = 10,530 transitions against the B1's 1,003, and `balance_embodiments` papers over the
    # 10:1 gap by repeating the B1 data ten times an epoch -- about 126 passes over the same
    # 1,003 pairs across 12 epochs, on a validation split too small to detect the memorisation
    # that invites. Balancing the *sampler* is not balancing the *data*.
    #
    # F13 says the hexapod side loses little: sixteen times more episodes of the same bodies
    # changed nothing, because what matters is how many bodies there are, not how many episodes
    # each walks. So the cap costs episodes, which are flat, and keeps bodies, which are not.
    #
    # hexapod=4 gives 6 bodies x 3 training clips x 65 = 1,170 transitions against the B1's
    # 1,003, a ratio of 1.17:1.
    clips_per_body: tuple = ()

    # Give every embodiment the same number of batches per epoch by repeating the smaller ones.
    # Proportional sampling hands the insect-plus-B1 pairing 6.3% of its gradient steps to the
    # quadruped, which would confound any claim about a shared latent space. False restores
    # proportional draws. See EmbodimentBatchSampler.
    balance_embodiments: bool = True

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

    # Hand the Forward Transition Model the embodiment identity directly, as an extra latent
    # token, so z is not the only route by which it can know which robot it is looking at.
    #
    # Measured motivation: 33.0% of z's variance is the embodiment, and deleting the eight
    # directions that carry it costs 1.69x against a random-direction control's 1.16x, so the
    # identity in z is load-bearing rather than passive leakage (FINDINGS.md F39). Removing it
    # with an adversary alone therefore breaks something the model depends on. The channel
    # relieves the need first; only then does removing the ability cost nothing.
    #
    # Names come from cfg.sources, so any script rebuilding a model from a checkpoint's config
    # gets the right embedding size without being told. A no-op in single-morphology mode.
    ftm_embodiment_channel: bool = False

    # Subtract a per-embodiment mean from the encoder's output before anything is trained on it,
    # so the constant appearance difference between two robots never reaches the ITM.
    #
    # The two datasets differ in ways that are not behaviour: the insect renders orange and fills
    # about a quarter of the frame, the B1 renders grey and fills about three quarters. Measured,
    # that shows up as the two embodiments' pooled embeddings sitting 3.94x apart relative to
    # their own spread, and it is enough on its own to make a stance-fraction readout fitted on
    # one embodiment fail on the other at 4.72x -- an offset absorbed into the readout's
    # intercept, not a difference in how contact is represented (FINDINGS.md F41).
    #
    # Centre only, never scale. The diagnosis was an offset; dividing each of the 1,408
    # dimensions by its own spread would additionally reweight L_recon across dimensions, which
    # changes what the forward model optimises and makes its loss incomparable to every run so
    # far. One 1,408-vector per embodiment, subtracted from every patch token, so spatial
    # structure is untouched.
    #
    # This attacks the same 33% as ftm_embodiment_channel from the other end: the channel removes
    # the *need* for z to carry identity, this removes the *supply*. Run them one at a time or
    # neither result is interpretable.
    center_embeddings: bool = False
    # Frames per embodiment used to estimate the mean. The offset is a global property of how a
    # robot renders, not something that needs the whole dataset to pin down.
    center_frames: int = 300

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

    # Which joint command the Motion Decoder is asked for, counted from frame t.
    #
    # The collector applies cmds[t], steps the simulator, and only then captures frames[t], so
    # frames[t] is the *result* of actions[t] and the transition frames[t] -> frames[t+1] is
    # caused by actions[t+1]. With action_lag 0 the target is therefore already visible in the
    # decoder's own input, e_t, and z has nothing left to supply: measured, giving the ITM two
    # copies of e_t instead of a real transition costs only 1.11-1.19x (FINDINGS.md F29).
    #
    # action_lag 1 asks for the action that caused the transition, which is what z is defined to
    # represent. The decoder never sees e_{t+1}, so that answer can only arrive through z.
    # 0 reproduces every run recorded before 2026-08-09.
    action_lag: int = 1

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
