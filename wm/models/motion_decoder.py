"""Motion Decoder: (x_t, z_t) -> predicted joint command.

The latent action is the query in a cross-attention over the current frame's visual tokens,
which are first downsampled by a strided 2D convolution over the patch grid to cut compute.
Conditioning on x_t is what lets one shared latent decode to different joint values for
different bodies.
"""
import torch.nn as nn

from .blocks import CrossAttentionBlock


def _head(hidden, action_dim):
    return nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, action_dim),
    )


class MotionDecoder(nn.Module):
    """Shared backbone, one output head per embodiment.

    The backbone is what has to generalise: it reads the behaviour from z against the visual
    context. Only the final projection is embodiment-specific, because action spaces of
    different dimensionality have no shared coordinates.
    """

    def __init__(self, cfg, heads=None):
        super().__init__()
        self.grid = cfg.grid
        self.proj = nn.Linear(cfg.token_dim, cfg.hidden)
        self.downsample = nn.Conv2d(cfg.hidden, cfg.hidden, cfg.md_pool, stride=cfg.md_pool)
        self.query_proj = nn.Linear(cfg.z_dim, cfg.hidden)
        self.cross = CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
        self.heads = nn.ModuleDict(
            {name: _head(cfg.hidden, dim) for name, dim in (heads or {"default": cfg.action_dim}).items()}
        )

    def features(self, x_t, z):
        batch = x_t.shape[0]
        tokens = self.proj(x_t)
        tokens = tokens.transpose(1, 2).reshape(batch, -1, self.grid, self.grid)
        tokens = self.downsample(tokens).flatten(2).transpose(1, 2)
        return self.cross(self.query_proj(z).unsqueeze(1), tokens)

    def forward(self, x_t, z, embodiment="default"):
        return self.heads[embodiment](self.features(x_t, z)).squeeze(1)

    def add_head(self, name, hidden, action_dim, device=None):
        """A body with a new action space needs its own head; the backbone stays frozen."""
        head = _head(hidden, action_dim)
        self.heads[name] = head.to(device) if device else head
        return self.heads[name]
