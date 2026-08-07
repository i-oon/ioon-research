"""Inverse Transition Model: (x_t, x_t+1) -> latent action z_t.

Causal self-attention contextualises the two frames' tokens — tokens of x_t may not attend
to x_t+1, so the current frame's representation stays independent of the future. A learned
query token then cross-attends to that context and is projected to z_t.
"""
import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock, SelfAttentionBlock


class InverseTransitionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.token_dim, cfg.hidden)
        self.frame_embed = nn.Parameter(torch.zeros(2, 1, cfg.hidden))
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.hidden))

        self.self_blocks = nn.ModuleList(
            SelfAttentionBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.itm_self_blocks)
        )
        self.cross_blocks = nn.ModuleList(
            CrossAttentionBlock(cfg.hidden, cfg.heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.itm_cross_blocks)
        )
        self.norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.z_dim)

        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    @staticmethod
    def _causal_mask(n_tokens, device):
        total = 2 * n_tokens
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:n_tokens, n_tokens:] = True
        return mask

    def forward(self, x_t, x_next):
        n_tokens = x_t.shape[1]
        tokens = torch.cat(
            [self.proj(x_t) + self.frame_embed[0], self.proj(x_next) + self.frame_embed[1]],
            dim=1,
        )
        mask = self._causal_mask(n_tokens, tokens.device)
        for block in self.self_blocks:
            tokens = block(tokens, attn_mask=mask)

        query = self.query.expand(tokens.shape[0], -1, -1)
        for block in self.cross_blocks:
            query = block(query, tokens)
        return self.head(self.norm(query)).squeeze(1)
