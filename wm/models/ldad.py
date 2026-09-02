"""Delta-JEPA's Latent Difference Action Decoder: read the action out of `e_t+1 - e_t`.

**The point is the input, not the architecture.** Delta-JEPA (2606.31232) trains the world model so
that the *difference* of consecutive state latents carries the action, with a large weight, and
reports that this is what prevents action-insensitive collapse. Its own explanation for why the
endpoints are not enough is exactly F168's result here: given `[z_t, z_t+1]` a decoder can read the
action off action-correlated cues in `z_t+1` without modelling the transition at all. **A difference
cannot be read that way**, which is the whole argument for the term.

`e_t+1` here is the FORWARD MODEL'S PREDICTION, not the true next embedding. Decoding the action
from a difference of two true frames would train nothing -- the gradient has to reach the forward
model, which is the module that collapses.

Three transformer layers over the patch tokens of the difference, matching their setup. The pooled
token is decoded per embodiment, because 18-D and 12-D joint commands share no coordinates.
"""
import torch
import torch.nn as nn


class LatentDifferenceDecoder(nn.Module):
    def __init__(self, cfg, heads, layers=3, width=None, n_heads=8):
        super().__init__()
        width = width or cfg.hidden
        self.proj = nn.Linear(cfg.token_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.cls, std=0.02)
        block = nn.TransformerEncoderLayer(width, n_heads, width * 4, batch_first=True,
                                           norm_first=True, dropout=cfg.dropout)
        self.body = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleDict({n: nn.Linear(width, int(d)) for n, d in heads.items()})

    def forward(self, delta, embodiment="default"):
        x = self.proj(delta)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        return self.heads[embodiment](self.norm(self.body(x)[:, 0]))
