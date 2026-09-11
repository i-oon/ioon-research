"""The spatial-preserving recurrent cell validated in
`scripts/diagnostics/objective_experiments/spatial_recurrent_killgate.py` (F180's ConvGRU
kill-gate, gap +0.069, best of four architecture variants on a 2000-iteration probe).

**Extracted here, unchanged, so the full-budget retrain
(`scripts/diagnostics/objective_experiments/convgru_full_retrain.py`) trains the EXACT
architecture the probe measured** -- not a re-implementation that could silently drift. The
original kill-gate script is left untouched and still reproduces its own recorded result exactly;
this module is the new, single source of truth for anything built on top of it going forward.

Per-embodiment differences (action dimension: hexapod 18, B1 12) are handled by a per-embodiment
`action_to_map` head, the same convention `ActionProjector` uses elsewhere in this codebase for a
shared body with embodiment-specific input dimensions. Everything else (the ConvGRU cell, the
spatial-to-channel projection, the decoder) is shared across embodiments, since the hypothesis
under test is a shared, morphology-agnostic recurrent latent -- not a per-body model.
"""
import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """Every gate is a 3x3 conv over the spatial map, not a linear layer over a pooled vector.
    `h` stays `[B, hidden_channels, grid, grid]` for the model's entire lifetime."""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv_zr = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels,
                                 kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(in_channels + hidden_channels, hidden_channels,
                                kernel_size=3, padding=1)

    def forward(self, x, h):
        zr = self.conv_zr(torch.cat([x, h], dim=1))
        z, r = zr.chunk(2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * h_tilde


class SpatialRecurrentModel(nn.Module):
    """Predicts delta-Froude (body-motion change) from a V-JEPA2 token grid + action, carrying
    genuine spatial recurrent state across a sequence. `token_dim`/`grid` match V-JEPA2's cached
    config (256 tokens = 16x16, `token_dim=1408`); `action_dims` is `{embodiment: action_dim}` for
    every body this instance will ever be trained or evaluated on -- fixed at construction, not
    extendable, so a checkpoint's embodiment coverage is legible from its own state dict shape."""

    def __init__(self, action_dims, token_dim=1408, grid=16, in_channels=32, hidden_channels=64,
                body_dim=3):
        super().__init__()
        self.grid = grid
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.proj_in = nn.Linear(token_dim, in_channels)
        self.action_to_map = nn.ModuleDict({
            name: nn.Linear(dim, in_channels) for name, dim in action_dims.items()})
        self.cell = ConvGRUCell(in_channels, hidden_channels)
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Linear(hidden_channels, 64),
                                     nn.GELU(), nn.Linear(64, body_dim))

    def step(self, h, tokens, action, embodiment):
        """tokens: [B,256,token_dim]  action: [B,action_dim]  h: [B,hidden,grid,grid]
        -> new h, predicted delta-Froude [B,body_dim]"""
        b = tokens.shape[0]
        x = self.proj_in(tokens).transpose(1, 2).reshape(b, self.in_channels, self.grid, self.grid)
        x = x + self.action_to_map[embodiment](action).view(b, self.in_channels, 1, 1)
        h = self.cell(x, h)
        pooled = h.mean(dim=(2, 3))          # pooling happens ONLY here, at final read-out
        return h, self.decoder(pooled)

    def init_hidden(self, b, device):
        return torch.zeros(b, self.hidden_channels, self.grid, self.grid, device=device)
