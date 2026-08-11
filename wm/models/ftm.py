"""Forward Transition Model: (x_t, z_t) -> predicted x_t+1.

Each block applies self-attention over the visual tokens, self-attention over the latent
action tokens, then cross-attention from the visual tokens to the latent action. With the
default single latent token the latter self-attention is a no-op; it is kept because the
block structure follows LAC-WM, where latent actions are sequences of chunked steps.

`cfg.ftm_embodiment_channel` appends a learned per-embodiment token to the latent sequence.
The point is to give the model a route to the identity that is not z: with 33.0% of z's
variance being "which robot this is", and that identity load-bearing rather than incidental
(FINDINGS.md F39), nothing can strip it from z without breaking what depends on it. The
channel supplies it separately so z is free to stop carrying it.

A token rather than something added into z, so that with the channel off the model is
bit-for-bit the one trained before, and so the identity stays separable from the latent
instead of being mixed back into the quantity being measured. The latent self-attention that
is a no-op at one token becomes real once this second token exists.
"""
import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock, SelfAttentionBlock


def embodiment_names(cfg):
    """Embodiment names in a fixed order, taken from cfg.sources ("name=dir" per entry).

    Read off the config rather than passed in, so a script that rebuilds a model from a
    checkpoint gets the right embedding size without having to know what it was trained on.
    """
    return tuple(spec.split("=")[0] for spec in cfg.sources)


class FTMBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.visual = SelfAttentionBlock(dim, heads, mlp_ratio, dropout)
        self.latent = SelfAttentionBlock(dim, heads, mlp_ratio, dropout)
        self.cross = CrossAttentionBlock(dim, heads, mlp_ratio, dropout)

    def forward(self, visual, latent):
        visual = self.visual(visual)
        latent = self.latent(latent)
        return self.cross(visual, latent), latent


class ForwardTransitionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.z_tokens = cfg.z_tokens
        self.hidden = cfg.hidden

        self.visual_proj = nn.Linear(cfg.token_dim, cfg.hidden)
        self.latent_proj = nn.Linear(cfg.z_dim, cfg.hidden * cfg.z_tokens)

        names = embodiment_names(cfg) if getattr(cfg, "ftm_embodiment_channel", False) else ()
        self.embodiments = {name: i for i, name in enumerate(names)}
        # zero-initialised, so every embodiment starts indistinguishable and training begins
        # equivalent to having no channel; any separation that appears is the model choosing to
        # use it rather than an initialisation the run inherited
        self.embodiment_token = nn.Embedding(len(names), cfg.hidden) if names else None
        if self.embodiment_token is not None:
            nn.init.zeros_(self.embodiment_token.weight)

        self.blocks = nn.ModuleList(
            FTMBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.ftm_blocks)
        )
        self.norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.token_dim)

    def forward(self, x_t, z, embodiment=None):
        visual = self.visual_proj(x_t)
        latent = self.latent_proj(z).view(-1, self.z_tokens, self.hidden)
        if self.embodiment_token is not None:
            if embodiment not in self.embodiments:
                raise KeyError(f"embodiment {embodiment!r} not in {sorted(self.embodiments)}; "
                               "the channel is on, so every call has to name one")
            index = torch.tensor(self.embodiments[embodiment], device=latent.device)
            token = self.embodiment_token(index).expand(latent.shape[0], 1, -1)
            latent = torch.cat([latent, token], dim=1)
        for block in self.blocks:
            visual, latent = block(visual, latent)
        return self.head(self.norm(visual))
