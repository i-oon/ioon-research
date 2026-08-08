"""Gradient-reversal head that pushes body identity out of the latent action.

The reason for this is measured, not assumed. Decomposing the variance of `z` across five
training bodies at matched gait phases gives 64.1 percent to where in the gait cycle the frame
is, 11.1 percent to which body it is, and the rest to interaction. So `z` is doing the job it
was designed for. But a linear probe still recovers the body from `z` at 72 percent against a
20 percent chance level, and crossing the decoder's inputs shows it takes the body from `z` and
not from the frame: feeding body A's frame with body B's latent produces body B's joint
commands to within 3.5 degrees.

That is the failure. The decoder has the whole frame available, which carries leg lengths in
full, and instead keys off a short well-separated code inside `z`, because a lookup over five
codes is an easier way to reduce the training loss than reading geometry off pixels. A lookup
does not extend to a body that has no code.

Removing that code leaves the decoder only one route to the body, the frame. The 11 percent
this targets is small, so the gait information `z` mainly carries should survive; whether it
does is what `adv/accuracy` in the training log reports.

Gradient reversal trains the classifier to be as good an adversary as it can while the reversed
gradient pushes the ITM to defeat it, so no separate optimiser or alternating schedule is needed.
"""
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forwards, negated and scaled backwards."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class MorphAdversary(nn.Module):
    """Predicts which training body `z` came from, against a reversed gradient.

    Kept deliberately small and non-linear: a classifier with too much capacity wins outright
    and the reversed gradient it sends back is dominated by its own overfitting rather than by
    information genuinely present in `z`.
    """

    def __init__(self, z_dim, n_bodies, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bodies),
        )

    def forward(self, z, scale=1.0):
        if z.dim() == 3:
            z = z.squeeze(1)
        return self.net(GradientReversal.apply(z, scale))


class MorphProbe(nn.Module):
    """Measures how decodable the body still is from z, without touching training.

    Needed because the adversary's own accuracy is ambiguous: near chance can mean the
    reversal removed the body, or that the classifier fighting a reversed gradient never got
    good enough to find it. This one reads a detached z, so it trains freely and sends nothing
    back. Its accuracy is a read-out, and it is logged whether or not lambda_adv is set, which
    gives the control run its baseline for free.
    """

    def __init__(self, z_dim, n_bodies, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bodies),
        )

    def forward(self, z):
        if z.dim() == 3:
            z = z.squeeze(1)
        return self.net(z.detach())
