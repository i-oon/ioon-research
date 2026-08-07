"""Pre-norm transformer blocks shared by the ITM, FTM and motion decoder."""
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0, dropout=0.0):
        super().__init__()
        inner = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, dropout)

    def forward(self, query, context):
        kv = self.norm_kv(context)
        attended, _ = self.attn(self.norm_q(query), kv, kv, need_weights=False)
        query = query + attended
        return query + self.mlp(self.norm2(query))
