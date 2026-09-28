"""LoRA (Hu et al. 2021) for ITM/FTM adaptation, following LAC-WM's own recipe (rank 2, applied to
IDM+FDM during their Stage 1 embodiment adaptation, `doc/ref/notes_lac_wm.md` section 5.3).

**Scope, stated plainly -- narrower than first attempted, and why.** `nn.MultiheadAttention` is not
just hard to reach into, it is actively incompatible with a generic wrapper: its fused forward
implementation reads `self.out_proj.weight`/`.bias` as raw tensors directly (not via
`self.out_proj(x)`), so `out_proj` must remain a literal `nn.Linear` with those attributes -- a
`LoRALinear` swapped in for it breaks with `AttributeError: 'LoRALinear' object has no attribute
'weight'` the moment MHA's own forward runs. Its fused QKV `in_proj_weight` is a single raw
Parameter, not a submodule, for the same reason unreachable this way. **This wraps only each
block's `Mlp` (two plain `nn.Linear`s, called via ordinary `self.net(x)`, genuinely swappable)**;
`out_proj` and `in_proj_weight` stay frozen, untouched, real `nn.Linear`/`Parameter` objects. That
is real capacity -- every `SelfAttentionBlock`/`CrossAttentionBlock` in ITM and FTM has an `Mlp` --
but it is narrower than LAC-WM's own LoRA (which the paper does not specify down to which weight
matrices), and this file does not claim otherwise.

**Why this exists.** Full-parameter adaptation (`wm/adapt.py`'s default) was measured, on this
project's own checkpoint (`beh24_hinge_cleansplit` adapted to B1), to leave hexapod's own rollout
gap ratio at 0.918 -- barely better than predicting the mean, down from a healthy pretrained value
-- while gaining a good B1 ratio (0.307). That is catastrophic forgetting, not a clean adaptation.
LoRA freezes every pretrained weight and learns only a small, low-rank delta on top, which bounds
how much the update can move the model away from what it already knew. This is not why LoRA was
originally introduced (parameter/memory efficiency for large-model fine-tuning was the original
motivation) -- reducing forgetting is a side effect of the same mechanism (frozen base + a
tiny additive update), which is why LAC-WM (and this project) uses it here.
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank delta: y = base(x) + scale * B(A(x)).

    `A` is initialised the standard way (small random, `kaiming_uniform_`) and `B` is initialised
    to exactly zero, so the wrapped layer is bit-for-bit identical to the original at the moment
    of wrapping -- training starts from the pretrained function, not a perturbed one.
    """

    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scale = self.alpha / self.rank
        in_features, out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scale * delta


def apply_lora(model: nn.Module, rank: int = 2, alpha: float = None):
    """Walks `model`, replaces every `nn.Linear` inside an `Mlp` block (see module docstring for
    why attention's `out_proj`/`in_proj_weight` are excluded) with a `LoRALinear` wrapping the
    SAME (now-frozen) weights. In place; also returns the count replaced, so a caller can assert
    it matched expectations rather than silently wrapping zero layers on an architecture change.

    Freezes every OTHER parameter in `model` too (LayerNorms, attention, embeddings, anything not
    an Mlp Linear), since the point is that only the LoRA deltas should be trainable after this
    call -- a caller that wants some other part trainable (e.g. a task head) must re-enable it
    explicitly afterward.
    """
    from .blocks import Mlp

    for p in model.parameters():
        p.requires_grad_(False)

    replaced = 0
    for module in model.modules():
        if isinstance(module, Mlp):
            for child_name, child in list(module.net.named_children()):
                if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                    setattr(module.net, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                    replaced += 1
    return replaced


def lora_parameters(model: nn.Module):
    """Every trainable (LoRA) parameter in `model` -- what an optimiser should actually see."""
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def merge_and_unwrap_lora(model: nn.Module):
    """Folds every `LoRALinear`'s trained delta into a plain `nn.Linear` and puts that back in the
    model, in place. **Required before saving a checkpoint** -- every other script in this
    codebase (`rollout`, `fit_projector`, the planners) loads a plain `InverseTransitionModel`/
    `ForwardTransitionModel` state dict; a checkpoint saved straight from the wrapped model would
    have `...base.weight`/`...lora_A`/`...lora_B` keys instead of `...weight`, and every one of
    them would fail to load. After this call the model (and its `state_dict()`) is bit-for-bit
    structurally identical to one that was never wrapped -- the adaptation is baked into the
    weights, not carried as a separate module.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged = nn.Linear(child.base.in_features, child.base.out_features,
                                   bias=child.base.bias is not None)
                delta_weight = child.scale * (child.lora_B @ child.lora_A)
                merged.weight.copy_(child.base.weight + delta_weight)
                if child.base.bias is not None:
                    merged.bias.copy_(child.base.bias)
                setattr(module, child_name, merged)
                replaced += 1
    return replaced
