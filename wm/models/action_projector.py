"""Maps an explicit action onto the latent action, so the world model can be driven at control time.

**The inverse model cannot run in the loop, and this is the module that replaces it.**
`z_t = ITM(e_t, e_{t+1})` needs the next frame, which at control time is the thing being decided.
Every latent this project has measured was read off two ground-truth frames -- reconstruction, not
control (F72). LAC-WM states the same constraint and the same answer: "since future observations,
required by the IDM, are unavailable at inference time, we train an action projector that maps
explicit actions into the latent action space".

With it, planning samples in **action space** rather than latent space:

    a^1..a^n  ->  projector  ->  z^1..z^n  ->  FDM rollout  ->  score  ->  execute the winner

which matters for two reasons. Every candidate is executable by construction, where a sampled `z`
need not correspond to any behaviour the robot can perform. And nothing has to decode `z -> a` at
run time, so the Motion Decoder -- whose ability to generalise across bodies has never been
measured -- leaves the runtime path entirely.

**One projector per embodiment**, because 18-D hexapod and 12-D quadruped commands share no
correspondence. That is the same reason `MotionDecoder` keeps per-embodiment output heads, and it is
not a weakening of the shared latent: the *target* `z` is one space, and each projector's job is to
find its own body's route into it.

**Fitted on the target robot's own actions**, which is the honest cost of a new body. "A new body
needs only video" overstates it and overstates LAC-WM, whose abstract adapts "through finetuning".
Video is what lets the world model span incomparable bodies; this module still needs actions. F45
measured how cheap that is -- one B1 clip clears break-even.
"""
import torch
import torch.nn as nn

from ..config import chunk_of


def action_dims_from(saved):
    """`{embodiment: action_dim}` for a saved projector, whether or not the file records it.

    **`wm/fit_projector` writes `action_dims`; `wm/adapt3` did not until 2026-08-30**, so every
    stage-3 checkpoint produced before then -- including the six the objective comparison rests on
    -- carries a projector that cannot be rebuilt from the key alone. The per-embodiment `mean_*`
    buffer is the action's width by construction, so the dimensions are always recoverable.

    Defined here rather than in each caller because three scripts and the planner each did it
    separately, and the planner's copy was fixed while the other three kept failing with a
    `KeyError` that reads like a corrupt checkpoint.
    """
    state = saved.get("projector", saved) if isinstance(saved, dict) else saved
    dims = saved.get("action_dims") if isinstance(saved, dict) else None
    dims = dims or {k[len("mean_"):]: v.numel() for k, v in state.items()
                    if k.startswith("mean_")}
    if not dims:
        raise KeyError("projector carries neither action_dims nor mean_* buffers")
    return dims


class ActionProjector(nn.Module):
    """`{embodiment: action_dim}` -> a shared `z_dim` latent."""

    def __init__(self, cfg, action_dims, hidden=None, depth=2):
        super().__init__()
        width = hidden or cfg.hidden
        self.z_dim = cfg.z_dim
        # **A stride-k latent summarises k commands, so the projector reads all k** (`action_chunk`,
        # `wm/data/strided.py`): input (..., k, dim), flattened after standardising each joint. The
        # statistics stay per joint -- one command's worth -- so a chunk is standardised the same
        # way its commands are one at a time. `chunk_of` is 1 for every checkpoint recorded before
        # stride existed, which keeps their shapes and outputs exactly as they were.
        self.chunk = chunk_of(cfg)

        def stack(dim):
            layers, d = [], dim
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.GELU()]
                d = width
            return nn.Sequential(*layers, nn.Linear(width, cfg.z_dim))

        self.nets = nn.ModuleDict({name: stack(dim * self.chunk) for name, dim in action_dims.items()})

        # **Standardise the action per embodiment, and keep the statistics inside the module.**
        # Joint commands are radians in body-specific ranges; the latent is whatever the ITM made.
        # Fitting across that gap unnormalised puts the two embodiments on different loss scales, so
        # whichever has the larger command range dominates the gradient. Registered as buffers so a
        # checkpoint carries them -- an earlier module in this project stored statistics outside the
        # state dict and a reloaded model silently used the wrong ones.
        for name, dim in action_dims.items():
            self.register_buffer(f"mean_{name}", torch.zeros(dim))
            self.register_buffer(f"std_{name}", torch.ones(dim))

    def set_stats(self, embodiment, mean, std):
        with torch.no_grad():
            getattr(self, f"mean_{embodiment}").copy_(torch.as_tensor(mean, dtype=torch.float32))
            getattr(self, f"std_{embodiment}").copy_(
                torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, action, embodiment):
        """`(..., dim)` at chunk 1; `(..., chunk, dim)` above it -- a wrong shape raises."""
        a = (action - getattr(self, f"mean_{embodiment}")) / getattr(self, f"std_{embodiment}")
        if self.chunk > 1:
            if a.shape[-2] != self.chunk:
                raise ValueError(f"projector expects {self.chunk} commands per latent, got shape "
                                 f"{tuple(action.shape)}")
            a = a.flatten(-2)
        return self.nets[embodiment](a)
