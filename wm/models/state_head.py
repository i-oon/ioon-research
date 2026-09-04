"""Reads the FTM's predicted change, not the frame, so it cannot decode robot identity instead of
motion -- the trap `body_sees_frame=True` fell into once already (F64: cross-embodiment transfer
+0.544/+0.435 -> -10.5/-57.2, because a frame-conditioned head learns one mapping per robot and
stops needing the latent to carry the behaviour).

    delta = FTM(e_t, z) - e_t     (or a later rolled step's own delta)
    pooled = delta.mean(1) - offset_<embodiment>
    StateHead(pooled, z) -> body motion

**The delta alone was not enough.** Measured (`frame_vs_delta_classify.py`): mean-pooling over patch
tokens *concentrates* an identity leak rather than diluting it -- pooled delta reads embodiment at
0.977, nearly as leaky as the raw frame. Removing each embodiment's own mean pooled-delta collapsed
that to chance (0.464, held-out clips, offset fit on training clips only) while Delta-state ridge R2
was unchanged (0.852 both ways) -- an additive per-embodiment offset, F41's mechanism, not something
distributed that needs an adversary.

**The offset is a frozen buffer, per embodiment, exactly like `ActionProjector`'s `mean_*`/`std_*`.**
Fit once from training data via `set_offset`, never recomputed from an evaluation batch -- doing that
would let an eval body centre on its own statistics, the same oracle leak `center_embeddings` avoids
for the encoder. Registered as a buffer so a checkpoint carries it, the same discipline
`ActionProjector` documents: an earlier module in this project kept statistics outside the state
dict and a reloaded model silently used the wrong ones.

**Known limitation, not yet resolved.** The offset is a property of the CURRENT (ITM, FTM), which
keep training under `L_recon`; a long run risks the fixed offset drifting stale as the pair it was
computed from changes. Fine for a short confirm fine-tune (`multistep_derisk.py`-scale); a full
retrain may need periodic re-estimation from training clips only, not yet implemented here.
"""
import torch
import torch.nn as nn


class StateHead(nn.Module):
    def __init__(self, cfg, state_dim, embodiments, hidden=None):
        super().__init__()
        hidden = hidden or cfg.state_hidden
        self.pool = nn.Sequential(nn.LayerNorm(cfg.token_dim), nn.Linear(cfg.token_dim, hidden),
                                  nn.GELU())
        self.z_proj = nn.Linear(cfg.z_dim, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, state_dim))
        for name in embodiments:
            # zero until `set_offset` fits it; a state head built before the offset is computed
            # (e.g. for a shape check) is then the un-corrected version, not a broken one
            self.register_buffer(f"offset_{name}", torch.zeros(cfg.token_dim))

    def set_offset(self, embodiment, mean):
        """Fix this embodiment's offset. Call once, from training clips, before training -- never
        from an evaluation batch."""
        with torch.no_grad():
            getattr(self, f"offset_{embodiment}").copy_(
                torch.as_tensor(mean, dtype=torch.float32))

    def forward(self, delta, z, embodiment="default"):
        d = delta.mean(1) if delta.dim() == 3 else delta      # mean over patch tokens
        d = d - getattr(self, f"offset_{embodiment}")         # AttributeError on an unknown name,
        return self.head(self.pool(d) + self.z_proj(z))       # not a silent skip of the correction
