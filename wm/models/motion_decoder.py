"""Motion Decoder: (x_t, z_t) -> predicted joint command.

The latent action is the query in a cross-attention over the current frame's visual tokens,
which are first downsampled by a strided 2D convolution over the patch grid to cut compute.
Conditioning on x_t is what lets one shared latent decode to different joint values for
different bodies.

`cfg.md_head` selects how much capacity sits between the latent and the joint command. On a
body held out of training, capacity costs accuracy: with identical inputs, a linear readout
reached 4.88 deg per joint on the held-out body where the full decoder reached 10.94, while
the full decoder was 5.7x better on the bodies it trained on. Two training bodies do not
constrain what happens between them, so a model free enough to fit them as two separate cases
does exactly that. See FINDINGS.md F4b.

  mlp     cross-attention backbone, two-layer head. The original design.
  linear  cross-attention backbone, single linear projection as the head.
  probe   no backbone at all: mean-pooled visual tokens concatenated with z, one linear layer.
  pooled  mlp, plus the mean over patch tokens concatenated onto the attention output.

`pooled` exists because of a specific measurement. Ridge regression on the mean over V-JEPA2's
256 patch tokens recovers a held-out body's three segment scales to 0.050, 0.039 and 0.002,
while the trained decoder's answer for that body implies (0.98, 0.98, 0.97) against an actual
(0.80, 0.90, 0.90) -- worse than the linear map, with 5.2M parameters against 4,227
(FINDINGS.md F20). The information is in the frame, linearly available, and the decoder does not
use it.

The difference is how each one reaches the tokens. The probe averages all of them. The decoder
sees them only through cross-attention with `z` as the query, so it retrieves what `z` asks for,
and `z` is 64 percent gait phase (F19); morphology sits in the tokens unqueried. `pooled` adds
the averaged view alongside the attention path, so the same signal the probe uses is reachable
without `z` having to ask.
"""
import torch
import torch.nn as nn

from ..config import chunk_of
from .blocks import CrossAttentionBlock

HEAD_MODES = ("mlp", "linear", "probe", "pooled")


def _head(mode, hidden, action_dim):
    if mode in ("mlp", "pooled"):
        return nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
    if mode in ("linear", "probe"):
        return nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, action_dim))
    raise ValueError(f"md_head must be one of {HEAD_MODES}, got {mode!r}")


class MotionDecoder(nn.Module):
    """Shared backbone, one output head per embodiment.

    The backbone is what has to generalise: it reads the behaviour from z against the visual
    context. Only the final projection is embodiment-specific, because action spaces of
    different dimensionality have no shared coordinates.
    """

    def __init__(self, cfg, heads=None):
        super().__init__()
        self.grid = cfg.grid
        self.mode = getattr(cfg, "md_head", "mlp")
        if self.mode not in HEAD_MODES:
            raise ValueError(f"md_head must be one of {HEAD_MODES}, got {self.mode!r}")

        if self.mode == "probe":
            # no attention, no depth: the head sees the frame only as an average over patch
            # tokens, which is the same input the ridge diagnostic in scripts/diagnostics/morphology_axis.py
            # was fitted on
            self.width = cfg.token_dim + cfg.z_dim
        else:
            self.width = cfg.hidden
            self.proj = nn.Linear(cfg.token_dim, cfg.hidden)
            self.downsample = nn.Conv2d(cfg.hidden, cfg.hidden, cfg.md_pool, stride=cfg.md_pool)
            self.query_proj = nn.Linear(cfg.z_dim, cfg.hidden)
            self.cross = CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            if self.mode == "pooled":
                # Projected from the raw token dimension, so this route does not depend on
                # anything z conditions.
                #
                # It adds a residual term to the action rather than being concatenated into the
                # head's input. Concatenating was tried first and the model learned to suppress
                # it: over four epochs the frame-ablation gap fell from 12.5x to 7.1x, meaning
                # the decoder used the frame *less* once given a second route to it. A fusion
                # layer can down-weight whichever input it likes, and it down-weighted the new
                # one. A residual cannot be routed around -- it lands directly on the output --
                # so the only way to keep it from hurting is to make it carry something useful.
                self.pooled_proj = nn.Sequential(
                    nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, cfg.hidden), nn.GELU())

        # **The head predicts a window of commands, not one.** With `frame_stride` k the pair
        # e_t -> e_{t+k} is caused by k commands, so scoring `z` against a single one asks the
        # latent to summarise an interval and then grades it on an instant. Measured (F88):
        # widening the frames alone took validation motion from 0.218 to 0.928, roughly the level
        # of predicting the training mean. LAC-WM chunks both sides for this reason.
        #
        # `chunk_of` returns 1 whenever `frame_stride` is 1, so every checkpoint recorded before
        # this existed rebuilds with identical parameter shapes and loads unchanged.
        self.action_chunk = chunk_of(cfg)
        spec = heads or {"default": cfg.action_dim}
        self.action_dims = dict(spec)
        self.heads = nn.ModuleDict({name: _head(self.mode, self.width, dim * self.action_chunk)
                                    for name, dim in spec.items()})

        # **One head, no embodiment key, and blind to the frame.** Both properties are load-bearing
        # and the second was measured the hard way.
        #
        # The heads above are per-embodiment because 18-D and 12-D joint commands are different
        # spaces, so nothing in `L_motion` requires one `z` to mean the same thing on both robots.
        # LAC-WM does not face that fork: its motion decoder targets end-effector and camera pose,
        # the same numbers for a human hand, a humanoid and a Franka arm, so one output layer
        # enforces shared meaning. Its EAC-WM baseline -- separate encoders per embodiment, what we
        # had -- gives embeddings "clearly separated by dataset" in their Figure 2 (F55, F58).
        #
        # **Why it does not take `x_t`, unlike every other head here.** LAC-WM's MD is conditioned
        # on the observation, so matching the method means passing the frame. That was tried
        # (`stage2_speed7_bodyframe`, F64) and it destroys the effect: cross-embodiment transfer
        # goes from +0.544 / +0.435 to **-10.5 / -57.2**, worse than no term at all.
        #
        # The reason is not that the head ignores `z` -- ablating `z` still costs 2.32x. It is that
        # **the frame reveals which robot this is**, so the head learns one mapping per robot and
        # `z` is free to use a robot-specific code again. That is the per-embodiment head problem
        # re-entering through the image. Body speed is readable from a still frame (the frozen
        # encoder scores R^2 0.676 on it), which is what makes the shortcut available here and not
        # in LAC-WM's setting.
        #
        # So: any input that identifies the embodiment lets this head decode conditionally, and
        # conditional decoding is exactly what it exists to prevent. `z` is built from two frames,
        # so the head still reads vision -- through the bottleneck being shaped, which is the point.
        build_body = getattr(cfg, "lambda_body", 0.0) > 0 and getattr(cfg, "body_dim", 0)
        self.body_sees_frame = getattr(cfg, "body_sees_frame", False)
        if not build_body:
            self.body_head = None
        elif self.body_sees_frame:
            # kept so F64's negative result is reproducible, not because it should be used
            self.body_head = _head(self.mode, self.width, cfg.body_dim)
        else:
            self.body_head = nn.Sequential(
                nn.LayerNorm(cfg.z_dim),
                nn.Linear(cfg.z_dim, cfg.body_hidden),
                nn.GELU(),
                nn.Linear(cfg.body_hidden, cfg.body_dim),
            )
        # per-embodiment because action spaces differ in dimension, but one linear layer each,
        # so the share of the decoder that transfers to a new embodiment barely moves
        self.offsets = nn.ModuleDict(
            {name: nn.Linear(cfg.hidden, dim * self.action_chunk) for name, dim in spec.items()}
        ) if self.mode == "pooled" else None
        if self.offsets is not None:
            for layer in self.offsets.values():
                nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def features(self, x_t, z):
        if self.mode == "probe":
            return torch.cat([x_t.mean(dim=1), z.squeeze(1) if z.dim() == 3 else z], dim=-1).unsqueeze(1)
        batch = x_t.shape[0]
        tokens = self.proj(x_t)
        tokens = tokens.transpose(1, 2).reshape(batch, -1, self.grid, self.grid)
        tokens = self.downsample(tokens).flatten(2).transpose(1, 2)
        return self.cross(self.query_proj(z).unsqueeze(1), tokens)

    def chunk(self, x_t, z, embodiment="default"):
        """The whole predicted command window, `(batch, action_chunk, action_dim)`.

        This is what `L_motion` trains on. `forward` returns only the first step, so that the
        dozen diagnostics and refit scripts written against a 2-D output keep working -- a 3-D
        prediction would broadcast against their 2-D targets rather than raise, and report a
        plausible wrong number.
        """
        action = self.heads[embodiment](self.features(x_t, z)).squeeze(1)
        if self.offsets is not None:
            # starts at exactly zero, so training begins identical to the mlp decoder and any
            # departure from it is the pooled route being used
            action = action + self.offsets[embodiment](self.pooled_proj(x_t.mean(dim=1)))
        return action.reshape(action.shape[0], self.action_chunk, -1)

    def forward(self, x_t, z, embodiment="default"):
        """The command at `t + action_lag`: the first step of the window, shape `(batch, dim)`.

        Identical to the pre-chunking output whenever `action_chunk` is 1, which is every run
        with `frame_stride` 1.
        """
        return self.chunk(x_t, z, embodiment)[:, 0]

    def body(self, x_t, z):
        """Body motion from `z` alone. `x_t` is accepted and ignored unless body_sees_frame."""
        if self.body_head is None:
            raise RuntimeError("lambda_body is 0; no shared body head was built")
        if self.body_sees_frame:
            return self.body_head(self.features(x_t, z)).squeeze(1)
        return self.body_head(z)

    def add_head(self, name, hidden, action_dim, device=None):
        """A body with a new action space needs its own head; the backbone stays frozen."""
        head = _head(self.mode, self.width, action_dim * self.action_chunk)
        self.action_dims[name] = action_dim
        self.heads[name] = head.to(device) if device else head
        if self.offsets is not None:
            offset = nn.Linear(hidden, action_dim * self.action_chunk)
            nn.init.zeros_(offset.weight); nn.init.zeros_(offset.bias)
            self.offsets[name] = offset.to(device) if device else offset
        return self.heads[name]
