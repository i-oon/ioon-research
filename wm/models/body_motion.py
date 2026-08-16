"""Decode body-level motion from the latent, with one head shared by every embodiment.

This is the piece the pipeline was missing, and its absence is what F55 traces the whole
switch-behaviour failure back to.

`L_motion` supervises `z` through a **per-embodiment** head onto joint commands: 18 numbers for the
hexapod, 12 for the B1, with no correspondence between them. Nothing in that objective ever asks
one latent to mean the same thing on both robots, so the shared trunk is free to partition by robot
-- and F43/F46 measured that it does, reading embodiment out of `z` at 0.994 while a per-leg
readout transfers at 0.373 across robots against 0.531 from the frozen encoder alone.

LAC-WM does have such a term and ours did not. Their motion decoder targets a hand-unified label --
a 9-D end-effector pose plus a 9-D camera pose, the same convention for human hands, a humanoid and
a Franka arm -- so `z` is pushed toward a representation that decodes the same way for every
embodiment. Their own EAC-WM ablation, identical weight sharing without that pressure, produces
visibly disjoint per-dataset clusters.

**The locomotion equivalent of an end-effector pose is body motion.** Every legged robot has a
body, a forward speed and a height, whatever its leg count, and F56 measured that this is the level
where the two robots actually overlap: a hexapod at 0.13 m hip height and a B1 at 0.56 m walk at
Froude 0.155 and 0.159. Leg-level quantities cannot be shared -- one leg's phase fixes all four of
the B1's legs and almost none of the insect's five others -- but body-level ones can.

**Deliberately one head for all embodiments.** A per-embodiment head here would reintroduce exactly
the freedom this term exists to remove.
"""
import torch.nn as nn


class BodyMotionDecoder(nn.Module):
    """z -> body motion, dimensionless.

    Small on purpose. The point is to constrain `z`, not to build an accurate velocity estimator;
    capacity here would let the head absorb an embodiment-specific correction and leave `z`
    unconstrained, which is the failure mode being fixed.
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.z_dim),
            nn.Linear(cfg.z_dim, cfg.body_hidden),
            nn.GELU(),
            nn.Linear(cfg.body_hidden, cfg.body_dim),
        )

    def forward(self, z):
        return self.net(z)
