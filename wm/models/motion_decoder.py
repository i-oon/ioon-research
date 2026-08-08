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
"""
import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock

HEAD_MODES = ("mlp", "linear", "probe")


def _head(mode, hidden, action_dim):
    if mode == "mlp":
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
            # tokens, which is the same input the ridge diagnostic in scripts/morphology_axis.py
            # was fitted on
            self.width = cfg.token_dim + cfg.z_dim
        else:
            self.width = cfg.hidden
            self.proj = nn.Linear(cfg.token_dim, cfg.hidden)
            self.downsample = nn.Conv2d(cfg.hidden, cfg.hidden, cfg.md_pool, stride=cfg.md_pool)
            self.query_proj = nn.Linear(cfg.z_dim, cfg.hidden)
            self.cross = CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)

        self.heads = nn.ModuleDict(
            {name: _head(self.mode, self.width, dim)
             for name, dim in (heads or {"default": cfg.action_dim}).items()}
        )

    def features(self, x_t, z):
        if self.mode == "probe":
            return torch.cat([x_t.mean(dim=1), z.squeeze(1) if z.dim() == 3 else z], dim=-1).unsqueeze(1)
        batch = x_t.shape[0]
        tokens = self.proj(x_t)
        tokens = tokens.transpose(1, 2).reshape(batch, -1, self.grid, self.grid)
        tokens = self.downsample(tokens).flatten(2).transpose(1, 2)
        return self.cross(self.query_proj(z).unsqueeze(1), tokens)

    def forward(self, x_t, z, embodiment="default"):
        return self.heads[embodiment](self.features(x_t, z)).squeeze(1)

    def add_head(self, name, hidden, action_dim, device=None):
        """A body with a new action space needs its own head; the backbone stays frozen."""
        head = _head(self.mode, self.width, action_dim)
        self.heads[name] = head.to(device) if device else head
        return self.heads[name]
