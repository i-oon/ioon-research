"""Forward Transition Model: (x_t, z_t) -> predicted x_t+1.

Each block applies self-attention over the visual tokens, self-attention over the latent
action tokens, then cross-attention from the visual tokens to the latent action. With the
default single latent token the latter self-attention is a no-op; it is kept because the
block structure follows LAC-WM, where latent actions are sequences of chunked steps.
"""
import torch.nn as nn

from .blocks import CrossAttentionBlock, SelfAttentionBlock


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
        self.blocks = nn.ModuleList(
            FTMBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.ftm_blocks)
        )
        self.norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.token_dim)

    def forward(self, x_t, z):
        visual = self.visual_proj(x_t)
        latent = self.latent_proj(z).view(-1, self.z_tokens, self.hidden)
        for block in self.blocks:
            visual, latent = block(visual, latent)
        return self.head(self.norm(visual))
