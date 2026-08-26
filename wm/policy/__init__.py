"""Control-time modules: what runs after pretraining, using only what exists at inference.

`z_t = ITM(e_t, e_{t+1})` needs the next frame, which at control time is the thing being decided,
so **nothing here may import the inverse model**. The latent reaches the forward model through the
action projector instead -- the constraint LAC-WM states and answers the same way.
"""
