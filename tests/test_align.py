"""Shared-z alignment (wm/align.py, Config.lambda_align, wm.adapt --anchor_align). CPU only, no data.

    .venv/bin/python3 tests/test_align.py          (or: .venv/bin/python3 -m pytest tests/test_align.py)

  soft targets       rows sum to 1; nearest Froude gets the largest weight
  collapse check     random u scores worse than u where Froude-matched pairs share a direction;
                     collapsed u (all equal) scores worse than matched too
  off = absent       lambda_align 0 builds no head, logs no `align`, leaves the queue untouched
  smoke              tiny ITM/FTM, two synthetic bodies, lambda_align 0.1: the align term falls
                     (first vs last 50 steps once the queues are full)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from wm.align import AlignQueue, soft_infonce, soft_targets  # noqa: E402


def test_soft_targets():
    g = torch.Generator().manual_seed(0)
    f, fc = torch.randn(16, 3, generator=g), torch.randn(40, 3, generator=g)
    w = soft_targets(f, fc, 0.25)
    assert torch.allclose(w.sum(1), torch.ones(16), atol=1e-6)
    assert (w.argmax(1) == torch.cdist(f, fc).argmin(1)).all()


def test_collapse_check():
    g = torch.Generator().manual_seed(0)
    n, d = 64, 64
    f = torch.randn(n, 3, generator=g)
    fc = f + 0.05 * torch.randn(n, 3, generator=g)       # other body: same motions, slightly off
    # matched: u is a fixed (random, injective) function of Froude, so Froude-near pairs share a direction
    W = torch.randn(3, d, generator=g)
    match = lambda x: torch.nn.functional.normalize(torch.tanh(x @ W) * 3, dim=1)
    rnd = lambda m: torch.nn.functional.normalize(torch.randn(m, d, generator=g), dim=1)
    l_match = soft_infonce(match(f), f, match(fc), fc).item()
    l_rand = soft_infonce(rnd(n), f, rnd(n), fc).item()
    one = torch.nn.functional.normalize(torch.ones(1, d), dim=1)
    l_coll = soft_infonce(one.expand(n, d), f, one.expand(n, d), fc).item()
    assert l_rand > l_match, (l_rand, l_match)
    assert l_coll > l_match, (l_coll, l_match)
    assert abs(l_coll - np.log(n)) < 1e-4                  # collapse = uniform prediction = log m


def test_queue_other_bodies_only():
    q = AlignQueue(5)
    q.push("a", torch.ones(3, 2), torch.zeros(3, 1))
    q.push("a", torch.ones(4, 2), torch.zeros(4, 1))
    q.push("b", 2 * torch.ones(2, 2), torch.zeros(2, 1))
    assert len(q.u["a"]) == 5                             # FIFO cap per body
    u, _ = q.others("a")
    assert len(u) == 2 and (u == 2).all()                 # never its own body
    assert q.others("c")[0].shape[0] == 7


# --------------------------------------------------------------------------- tiny model on CPU
def _tiny(**kw):
    from wm.config import Config
    c = Config()
    base = dict(token_dim=16, hidden=32, heads=2, z_dim=8, itm_self_blocks=1, itm_cross_blocks=1, ftm_blocks=1,
                action_dim=4, lambda_body=0.5, detach_body_z=False, lambda_motion=0.0, lambda_hinge=0.0,
                body_dim=3, body_channels=(0, 1, 2), sources=("hexapod=x", "b1=y"))
    for k, v in {**base, **kw}.items():
        setattr(c, k, v)
    return c


class _Enc:
    """Stand-in for V-JEPA2: the batch already holds (N, 256, 16) float 'embeddings'."""
    def encode(self, frames):
        return torch.as_tensor(np.stack(frames)).float()


def _batch(rng, body, n=8):
    # Froude f drives the transition; each body has its own appearance (offset + mixing), so z must
    # read motion through a body-specific map -- the setting the cross-body alignment targets
    f = rng.normal(size=(n, 3)).astype(np.float32)
    mix = np.random.default_rng(1 if body == "hexapod" else 2).normal(size=(3, 16)).astype(np.float32)
    off = 2.0 if body == "hexapod" else -2.0
    x = rng.normal(size=(n, 256, 16)).astype(np.float32) * 0.3 + off
    nxt = x + (f @ mix)[:, None, :] * 0.5
    b = {"view1_t": x, "view1_next": nxt, "view2_t": x, "view2_next": nxt}
    b = {k: torch.as_tensor(v) for k, v in b.items()}
    b["action"] = torch.as_tensor(rng.normal(size=(n, 4)).astype(np.float32))
    b["body_motion"] = torch.as_tensor(f)
    b["embodiment"] = [body] * n
    return b


def test_off_is_absent():
    import wm.train as T
    T.ALIGN_QUEUE = None
    cfg = _tiny()
    models = T.build_models(cfg, "cpu", heads={"hexapod": 4, "b1": 4})
    assert "align" not in models
    _, parts = T.forward_step(models, _Enc(), _batch(np.random.default_rng(0), "hexapod"), cfg, "cpu")
    assert "align" not in parts and T.ALIGN_QUEUE is None


def test_smoke_align_decreases(steps=600, verbose=False):
    import wm.train as T
    T.ALIGN_QUEUE = None
    torch.manual_seed(0)
    cfg = _tiny(lambda_align=0.1, align_queue=512, align_min_queue=256)
    models = T.build_models(cfg, "cpu", heads={"hexapod": 4, "b1": 4})
    assert "align" in models
    params = [p for m in models.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3)
    rng = np.random.default_rng(0)
    hist = []
    for step in range(steps):
        body = "hexapod" if step % 2 == 0 else "b1"
        loss, parts = T.forward_step(models, _Enc(), _batch(rng, body), cfg, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if "align" in parts:
            hist.append(parts["align"])
        if verbose and step % 50 == 0:
            print(f"step {step:4d} {body:7s} recon {parts['recon']:.4f} body {parts['body']:.4f} "
                  f"align {parts.get('align', float('nan')):.4f} cands {parts.get('align_cands', 0):.0f}")
    assert len(hist) > 100, "align never switched on"
    first, last = np.mean(hist[:50]), np.mean(hist[-50:])
    if verbose:
        print(f"align first-50 {first:.4f} -> last-50 {last:.4f} ({len(hist)} active steps)")
    assert last < first, (first, last)
    T.ALIGN_QUEUE = None


if __name__ == "__main__":
    verbose = "-v" in sys.argv
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(verbose=True) if (verbose and name == "test_smoke_align_decreases") else fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
