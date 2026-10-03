"""The one place that says what a transition is when the world model steps `k` frames at a time.

**Why a module of its own.** `frame_stride`/`action_chunk` existed in the pretraining loader only;
Stage 1 (`wm.adapt`), Stage 2 (`wm.fit_projector`), Stage 4 (`wm.fit_body_head`) and the planners
all hard-coded one-step pairs (`e_t -> e_{t+1}`, one action, `body_motion[t]`). A stride-k checkpoint
put through them would have been adapted, projected and read at a spacing it was never trained on --
and would have loaded and run without complaint. F259 measured why stride matters (a real
counterfactual future carries the action at r 0.15-0.20 after one step, 0.60-0.77 after eleven), so
every stage now builds its transitions here, from the checkpoint's own config.

A transition starting at `t`, stride `k`, action lag `lag`:

    frames      e_t -> e_{t+k}                         what the ITM reads, the FTM predicts
    action      actions[t + lag : t + lag + k]         the k commands that caused it (`action_window`)
    body        mean(body_motion[t : t + k])           the Froude over the interval z spans

At `k = 1` every function returns exactly what the one-step code did, so every checkpoint trained
before this module existed is read the same way it always was.

**`start` = the clip's `first_pair`** (`first_pair_of(clip)`). Counterfactual-branch clips carry the
source clip's prefix up to the branch index; only transitions starting at or after it are
training pairs (the training Datasets in `wm/data/dataset.py` already index from there). Every
helper here takes `start` and counts/returns transitions t = start .. start + n - 1, so stages 1-4
see the same pairs as pretraining. `start = 0` (every clip without the field) is bit-identical to
the code before the argument existed.
"""
import numpy as np
import torch


def stride_of(cfg):
    """Frames per world-model step, from a checkpoint's config (1 for anything recorded before).
    Raises if the action chunk differs from the stride: every stage builds k-command chunks for a
    k-frame latent, and a projector of another width would be fitted silently wrong."""
    from ..config import chunk_of
    k = max(1, int(getattr(cfg, "frame_stride", 1) or 1))
    if chunk_of(cfg) != k:
        raise ValueError(f"action_chunk {chunk_of(cfg)} != frame_stride {k}: unsupported by the "
                         "adaptation stages and planners (wm/data/strided.py)")
    return k


def first_pair_of(clip):
    """The first usable transition start of a loaded clip (0 unless it is a CF branch)."""
    return int(clip.get("first_pair", 0) or 0) if isinstance(clip, dict) else 0


def n_transitions(n_frames, k, lag=0, n_actions=None, n_body=None, start=0):
    """How many transitions a clip yields: every start `t >= start` whose frame pair, action window
    and body window all exist."""
    n = n_frames - k
    if n_actions is not None:
        n = min(n, n_actions - lag - k + 1)
    if n_body is not None:
        n = min(n, n_body - k + 1)
    return max(0, n - int(start))


@torch.no_grad()
def pair_latents(itm, e, k, n=None, batch=8, start=0):
    """`z_t = ITM(e_t, e_{t+k})` for start <= t < start + n (default n: every start with a partner)."""
    n = len(e) - k - start if n is None else n
    end = start + n
    return torch.cat([itm(e[t:min(t + batch, end)], e[t + k:min(t + batch, end) + k])
                      for t in range(start, end, batch)])


def action_chunks(actions, lag, k, n, start=0):
    """`actions[t + lag : t + lag + k]` for start <= t < start + n: (n, dim) at k == 1, (n, k, dim) above."""
    a = np.asarray(actions)
    if k == 1:
        return a[start + lag:start + lag + n]
    return np.stack([a[t + lag:t + lag + k] for t in range(start, start + n)])


def body_targets(motion, k, n, start=0):
    """Froude over each transition's interval (start <= t < start + n): `motion[t]` at k == 1,
    mean of `motion[t:t+k]` above."""
    m = np.asarray(motion)
    if k == 1:
        return m[start:start + n]
    c = np.concatenate([np.zeros_like(m[:1]), np.cumsum(m, 0)])
    t = start + np.arange(n)
    return (c[t + k] - c[t]) / k
