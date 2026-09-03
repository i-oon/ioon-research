"""Three wiring checks before the ActSWM rebuild burns five hours of pretraining.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/check_actswm_wiring.py

**Nothing here trains anything.** Each check answers a question that is cheap now and expensive
after a full pretrain has been run on a mis-wired objective.

  1  the null-action contrast   roll the forward model twice from the same `e_t`, once on the real
                                action's latent and once on the **null** action's -- the standing
                                stance of that body (F148), never a zero vector, which collapses
                                both robots. Confirm the two rollouts differ.

  2  the frozen readout         instantiate the new module -- randomly initialised, never trained,
                                reading `[e_t, e_t+1] -> action` -- and confirm one backward pass
                                leaves **its** parameters with no gradient while the forward model
                                receives one. **The ITM is not touched**: it produces the `z` the
                                projector imitates, and freezing it at random weights would make
                                that `z` arbitrary and break every control-time path.

  3  the starting sensitivity   `/mean-z` against the same null, on the current three-channel
                                checkpoint, so the rebuild's improvement is measured from the right
                                number rather than from the one-channel run's.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


class FrozenActionReadout(nn.Module):
    """`[e_t, e_t+1] -> action`, randomly initialised and never trained.

    **Randomly initialised is the mechanism, not a detail.** A readout that learns can move the
    boundary it scores to wherever the loss is looking -- which is what our contrastive repair did,
    producing sensitivity that lived only in the projector's region and only on the body it was
    adapted to (F139, F143). A fixed random map cannot relocate anything, so the only way to lower
    the loss is to make the transitions themselves separable.

    Nothing downstream consumes its output. It exists to route gradient into the forward model.
    """

    def __init__(self, token_dim, action_dim, hidden=256, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(2 * token_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, action_dim))
        for p in self.net.parameters():
            with torch.no_grad():
                p.copy_(torch.empty_like(p).normal_(0.0, 0.02, generator=g)
                        if p.dim() > 1 else torch.zeros_like(p))
            p.requires_grad_(False)

    def forward(self, e_t, e_next):
        return self.net(torch.cat([e_t.mean(-2), e_next.mean(-2)], dim=-1))


def stance_action(directory, embodiment):
    """The standing stance of this body: the pose its clips start in (F148)."""
    p = sorted(glob.glob(os.path.join(ROOT, directory, "*.npz")))[0]
    return np.asarray(load(p, REGISTRY[embodiment])["actions"])[0].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="wm/runs/beh12_hex-b1_body3/best.pt")
    ap.add_argument("--projector", default="wm/runs/beh12_hex-b1_body3/projector_b1_adapted.pt")
    ap.add_argument("--bodies", nargs="+",
                    default=["hexapod=data/allocentric/beh12_c10f10t10_flat:results/wm/cache/hex_c10.pt",
                             "b1=data/allocentric/beh12_b1_flat:results/wm/cache/b1_body3.pt"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    saved = torch.load(os.path.join(ROOT, args.projector), map_location="cpu", weights_only=False)
    proj = ActionProjector(cfg, action_dims_from(saved)).to(device).eval()
    proj.load_state_dict(saved["projector"])

    print("CHECK 1 — the null-action contrast, real z against the standing stance's z\n")
    print(f"  {'body':<9}{'horizon':>8}{'||real - null||':>18}{'||real - e_t||':>17}"
          f"{'null / real':>13}")
    sens = {}
    for spec in args.bodies:
        name, rest = spec.split("=", 1)
        directory, cache_path = rest.split(":", 1)
        stance = stance_action(directory, name)
        cache = torch.load(os.path.join(ROOT, cache_path), map_location="cpu")
        from vjepa2_encoder import VJEPA2FrameEncoder
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        clips = gather(os.path.join(ROOT, directory), name, enc, ck, cache, 2,
                       max(1, cfg.action_lag), device)[:args.clips]
        del enc
        torch.cuda.empty_cache()
        z_null = proj(torch.tensor(stance, device=device).unsqueeze(0), name)
        print(f"  {name}: stance action |a| = {float(np.abs(stance).max()):.3f}, "
              f"zero vector would be 0.000 — using the stance")
        rows = {h: [0.0, 0.0, 0] for h in args.horizons}
        mz = {h: [0.0, 0.0, 0.0, 0] for h in args.horizons}
        with torch.no_grad():
            for c in clips:
                e = c["e"].float().to(device)
                for t in range(2, min(c["n"], len(e) - max(args.horizons) - 1), args.stride):
                    e0 = e[t:t + 1]
                    real, null = e0, e0
                    for i in range(max(args.horizons)):
                        zr = proj(c["a"][t + i].unsqueeze(0).to(device), name)
                        real = ftm(real, zr)
                        null = ftm(null, z_null)
                        h = i + 1
                        if h in rows:
                            rows[h][0] += float((real - null).pow(2).mean())
                            rows[h][1] += float((real - e0).pow(2).mean())
                            rows[h][2] += 1
                            truth = e[t + h]
                            mz[h][0] += float((real[0] - truth).pow(2).mean())
                            mz[h][1] += float((null[0] - truth).pow(2).mean())
                            mz[h][3] += 1
        for h in args.horizons:
            d, m, n = rows[h]
            print(f"  {name:<9}{h:>8}{d / n:>18.4f}{m / n:>17.4f}{d / max(m, 1e-9):>13.3f}")
        sens[name] = {h: (mz[h][0] / mz[h][3], mz[h][1] / mz[h][3]) for h in args.horizons}

    print("\n  `||real - null||` is what the separation term has to grow. If it were ~0 the hinge")
    print("  would have nothing to push apart; if the null were the zero vector it would be large")
    print("  and meaningless, because the null would be a fall (F148).\n")

    print("CHECK 2 — the frozen readout: no gradient to itself, gradient through it to the FDM\n")
    ftm_train = ForwardTransitionModel(cfg).to(device)
    ftm_train.load_state_dict(ck["ftm"])
    readout = FrozenActionReadout(cfg.token_dim, 18).to(device)
    n_frozen = sum(p.numel() for p in readout.parameters())
    e0 = torch.randn(4, cfg.grid ** 2, cfg.token_dim, device=device) * 0.1
    z = torch.randn(4, cfg.z_dim, device=device)
    pred = ftm_train(e0, z)
    loss = readout(e0, pred).pow(2).mean()
    loss.backward()
    g_readout = [p.grad for p in readout.parameters() if p.grad is not None]
    g_ftm = sum(float(p.grad.norm()) for p in ftm_train.parameters() if p.grad is not None)
    n_ftm = sum(1 for p in ftm_train.parameters() if p.grad is not None)
    print(f"  readout: {n_frozen} parameters, requires_grad = "
          f"{ {p.requires_grad for p in readout.parameters()} }")
    print(f"  gradient reaching the readout's own parameters : {len(g_readout)} tensors  "
          f"-> {'NONE, correct' if not g_readout else 'PRESENT, WRONG'}")
    print(f"  gradient reaching the forward model            : {n_ftm} tensors, "
          f"total norm {g_ftm:.4f}  -> {'flows, correct' if g_ftm > 0 else 'ZERO, WRONG'}")
    print(f"  the ITM was not instantiated for training and is untouched")

    print("\nCHECK 3 — action-sensitivity against the same null, on this checkpoint\n")
    print(f"  {'body':<9}{'horizon':>8}{'error on real z':>18}{'error on null z':>18}"
          f"{'/mean-z':>10}")
    for name, d in sens.items():
        for h, (r, nl) in d.items():
            print(f"  {name:<9}{h:>8}{r:>18.4f}{nl:>18.4f}{r / max(nl, 1e-9):>10.3f}")
    print("\n  `/mean-z` here is the rollout on the real action over the rollout on the null action.")
    print("  **1.0 means the action changed nothing.** This is the number the rebuild has to move.")


if __name__ == "__main__":
    main()
