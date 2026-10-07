"""Shared-z alignment: soft contrastive (InfoNCE) loss on a small projection head, across bodies.

For an anchor transition i of body A with standardised Froude F_i, and candidates j from OTHER bodies:

    u          = normalize(g(z)),  g = LayerNorm -> Linear(z, 64) -> GELU -> Linear(64, 64)
    logits_ij  = cos(u_i, u_j) / tau
    w_ij       = softmax_j(-|F_i - F_j|^2 / (2 sigma^2))          (soft targets, sum to 1 per anchor)
    L          = mean_i CE(softmax(logits_i), w_i) = -mean_i sum_j w_ij log softmax_j(logits_ij)

Positives are Froude-near pairs ACROSS bodies, negatives Froude-far ones; within-body pairs are never
compared (the candidates are always another body's). Unlike `lambda_sim` (regress cos(z_i, z_j) onto a
Froude similarity, F278-F280), this only asks for the ranking of other-body neighbours to follow
Froude, and only in the head's space, so z keeps whatever else it needs for the forward model.

Used in pretraining (`Config.lambda_align`, wm/train.py: per-body FIFO queues of detached (u, F),
MoCo style, because a batch is one body) and in adaptation (`wm.adapt --anchor_align`: a fixed bank of
the pretrained bodies' (z, F) from the unadapted ITM). Off by default in both.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignHead(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, z):
        return F.normalize(self.net(z.float().flatten(1)), dim=1)


def soft_targets(f, fc, sigma):
    """(n_anchor, n_cand) soft targets from standardised Froude; each row sums to 1."""
    d2 = torch.cdist(f.float().flatten(1), fc.float().flatten(1)) ** 2
    return torch.softmax(-d2 / (2 * sigma ** 2), dim=1)


def soft_infonce(u, f, uc, fc, tau=0.1, sigma=0.25):
    """u (n, d) unit anchors with Froude f; uc (m, d) unit candidates (other bodies) with Froude fc."""
    logits = (u.float() @ uc.float().T) / tau
    w = soft_targets(f, fc, sigma)
    return -(w * torch.log_softmax(logits, dim=1)).sum(1).mean()


class AlignQueue:
    """Per-body FIFO of detached (u, F) for pretraining (one body per batch)."""

    def __init__(self, size=4096):
        self.size = size
        self.u, self.f = {}, {}

    def others(self, body):
        keys = [b for b in sorted(self.u) if b != body]
        if not keys:
            return None, None
        return torch.cat([self.u[b] for b in keys]), torch.cat([self.f[b] for b in keys])

    def push(self, body, u, f):
        u, f = u.detach().float().cpu(), f.detach().float().flatten(1).cpu()
        if body in self.u:
            u, f = torch.cat([self.u[body], u]), torch.cat([self.f[body], f])
        self.u[body], self.f[body] = u[-self.size:], f[-self.size:]
