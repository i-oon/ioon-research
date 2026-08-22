"""Maps an explicit action onto the latent action, so the world model can be driven at control time.

**The inverse model cannot run in the loop, and this is the module that replaces it.**
`z_t = ITM(e_t, e_{t+1})` needs the next frame, which at control time is the thing being decided.
Every latent this project has measured was read off two ground-truth frames -- reconstruction, not
control (F81). LAC-WM states the same constraint and the same answer: "since future observations,
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
Video is what lets the world model span incomparable bodies; this module still needs actions. F52
measured how cheap that is -- one B1 clip clears break-even.
"""
import torch
import torch.nn as nn


class ActionProjector(nn.Module):
    """`{embodiment: action_dim}` -> a shared `z_dim` latent."""

    def __init__(self, cfg, action_dims, hidden=None, depth=2):
        super().__init__()
        width = hidden or cfg.hidden
        self.z_dim = cfg.z_dim

        def stack(dim):
            layers, d = [], dim
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.GELU()]
                d = width
            return nn.Sequential(*layers, nn.Linear(width, cfg.z_dim))

        self.nets = nn.ModuleDict({name: stack(dim) for name, dim in action_dims.items()})

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
        a = (action - getattr(self, f"mean_{embodiment}")) / getattr(self, f"std_{embodiment}")
        return self.nets[embodiment](a)
