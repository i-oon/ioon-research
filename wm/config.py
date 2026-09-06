"""Hyperparameters for the latent-action world model.

Architecture values follow LAC-WM (Table 4 / Section 3.1). Optimisation values are
scaled down for a single 2080 Ti; the paper used 64 H200s with batch size 512.
"""
from dataclasses import dataclass, fields


def chunk_of(cfg):
    """How many commands the Motion Decoder predicts per pair.

    0 follows `frame_stride`, which is what pairs the action target with the interval the latent
    actually spans; see `Config.action_chunk`. Read through `getattr` so a Config rebuilt from a
    checkpoint recorded before either field existed still answers 1.
    """
    return max(1, int(getattr(cfg, "action_chunk", 0) or getattr(cfg, "frame_stride", 1) or 1))


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
    "frame_stride": 1,
    "action_chunk": 1,
    "cross_augment": True,
    "within_body_std": False,
    "lambda_cross": 0.0,
    "lambda_adv": 0.0,
    "lambda_body": 0.0,
    "lambda_hinge": 0.0,
    "lambda_readout": 0.0,
    "lambda_ldad": 0.0,
    "lambda_state": 0.0,
    "state_hidden": 256,
    "state_use_delta": True,
    "lambda_rollout": 0.0,
    "hinge_margin": 0.1,
    "hinge_K": 3,
    "readout_hidden": 512,
    "body_channels": (0,),
    "body_dim": 1,
    "body_sees_frame": False,
    "body_hidden": 128,
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
    # scripts/diagnostics/decoder/score_body.py, which needs no retraining.
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
    # How far apart the ITM's two frames sit. **1 is the default and is measured to be the wrong
    # unit for this data**: at 20 Hz `t -> t+1` is 50 ms, one nineteenth of a 0.95 s stride and 19%
    # of the pose change half a stride carries (F54). At that spacing the next frame is largely
    # guessable from the current one, so nothing forces the forward model to read `z` -- F87
    # measured the frame outweighing the latent 28x, and a latent from a different behaviour
    # costing 0.25%. LAC-WM down-samples the observation frequency by chunking to five steps for
    # the same reason. **F54 also found the long baseline losing at every horizon across robots**,
    # so raising this must be checked against transfer, not only against z-usage.
    frame_stride: int = 1
    action_lag: int = 1

    # How many consecutive joint commands the Motion Decoder is asked for, starting at
    # `t + action_lag`. **0 means "follow frame_stride", which is the only setting that keeps the
    # two halves of the objective describing the same interval.**
    #
    # Widening `frame_stride` without this is measurably wrong, not merely suboptimal. The pair
    # e_t -> e_{t+k} is caused by k commands, so `z` is asked to summarise k steps while
    # `L_motion` still scores it against one. Measured (F88): stride 10 raised the forward model's
    # use of the latent 1.6x (sweep z 4.257 -> 6.764) and simultaneously took validation motion
    # from 0.218 to 0.928 -- about the level of predicting the training mean, i.e. the decoder
    # stopped working. LAC-WM does not hit this because it chunks both: "we chunk the actions into
    # 5-step sequences".
    #
    # At chunk 1 the target keeps its old shape (action_dim,) exactly, so every run recorded
    # before this existed is reproduced bit-for-bit.
    action_chunk: int = 0

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

    # Decode body-level motion from z through a head **shared by every embodiment**, so that one
    # latent is required to mean the same thing on both robots. `lambda_motion` cannot do this: it
    # supervises through per-embodiment heads onto joint commands that have no correspondence
    # across 18 and 12 dimensions, which is what leaves the trunk free to partition by robot
    # (F55). 0.0 reproduces every run recorded before 2026-08-17.
    lambda_body: float = 0.0
    # --- ActSWM (F146). Zero by default: every run before 2026-08-31 reproduces unchanged.
    lambda_ldad: float = 0.0      # Delta-JEPA's LDAD: decode the action from the PREDICTED
    # state difference `FTM(e_t, z) - e_t`. Their sweep puts the useful range at 10-50 and their
    # best at 50, which is far above every other term here -- the term is meant to dominate, and a
    # small value reproduces the collapse it exists to prevent.
    ldad_layers: int = 3          # their decoder depth
    lambda_hinge: float = 0.0     # separation between the real-action and null-action rollouts
    lambda_readout: float = 0.0   # the frozen readout must recover the action from the prediction
    # **Auto-regressive 2-step consistency**, Demo-JEPA's `sloss` shape: FTM(FTM(e_t,z1),z2) against
    # the true e_t+2, z2 = ITM(e_t+1, e_t+2) from real frames (never from the FTM's own prediction,
    # so a bad step 1 cannot relabel what z2 means). `multistep_derisk.py` tested a version of this
    # on cached embeddings, ITM frozen, no cross-augmentation, and found no effect distinguishable
    # from fine-tuning noise -- this is the same question asked on the real loader instead, which
    # needs a third view per sample (`rollout_k=2` on `MultiEmbodimentPairs`) and is the reason this
    # is a separate flag rather than folded into an existing one. 0.0 reproduces every run before
    # 2026-09-05.
    lambda_rollout: float = 0.0
    hinge_margin: float = 0.1     # **0.1, not ActSWM's 0.3** -- at 0.3 the term collapses (F151)
    hinge_K: int = 3              # **3, not their 12** -- our rollout is reliable to about here
    readout_hidden: int = 512
    # Decode body motion from the FTM's predicted CHANGE (`FTM(e_t,z) - e_t`), not from z alone.
    # `lambda_body` supervises z directly and drowns under lambda_recon's 360,448-dim target
    # (z-alone ridge R2 0.005 on the embedding against 0.359 on body motion, same z, same session);
    # this reads the state off the world model's own rollout instead of bypassing it. Zero by
    # default: every run before this existed reproduces unchanged.
    lambda_state: float = 0.0
    state_hidden: int = 256
    # **F192, control this flag decides.** Offline probes found z alone predicts forward speed
    # better than z combined with the FTM's pooled predicted change (R2 0.781 vs 0.625 oracle,
    # 0.791 vs 0.537 on proj(action), the z ranking deploys) -- combining dilutes z rather than
    # adding to it. False drops `pool`/the offset from StateHead entirely, reading z_proj(z) alone.
    # True (default) reproduces every run before this flag existed, unchanged.
    state_use_delta: bool = True
    body_dim: int = 1            # must equal len(body_channels)
    # Which columns of `body_motion` the shared head supervises. Column 0 is forward speed, 1 is
    # lateral, 2 is yaw. Default (0,) is forward only -- lateral is an embodiment label in disguise
    # (F58: AUC 0.788 from that column alone) and yaw is the candidate `data/allocentric/beh12_*` was built to
    # test (F77). Kept in the config rather than as a module constant so a control arm and a
    # widened arm are the same code path with different settings.
    body_channels: tuple = (0,)
    # True reproduces F64's negative result: conditioning the shared head on the frame lets it
    # identify the robot and decode per-robot, and transfer collapses to -10.5 / -57.2.
    body_sees_frame: bool = False
    body_hidden: int = 128
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
