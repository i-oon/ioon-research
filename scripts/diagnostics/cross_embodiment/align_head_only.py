"""Can an alignment head alone align frozen z across bodies? (F318 follow-up, 2026-10-07)

z of the six-legged hexapod (c10) and B1 from a frozen checkpoint, held-out random-room clips; Froude = the training label
(mean 1 s CoM Froude over [t, t+k)), standardised over both bodies. A fresh AlignHead is trained ONLY on the soft InfoNCE
(wm/align.soft_infonce, both directions c10 <-> B1) on half of the clips and evaluated on the other half. Compared with the
best possible loss (entropy of the soft targets) and chance (log of the number of candidates).
  near the floor -> z carries matchable motion; the pretraining term failed for training reasons
  near chance    -> z of the two bodies is not matchable by a small head

    .venv/bin/python3 scripts/diagnostics/cross_embodiment/align_head_only.py wm/runs/round2_align_s0_rr/best.pt
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/figures", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
import shared_latent_figure as S  # noqa: E402
from wm.align import AlignHead, soft_infonce, soft_targets  # noqa: E402

CW = "data/counterfactual_walks"
S.BODIES[:] = [("c10", "hexapod", f"{CW}/rr_c10_clips_heldout", "results/wm/cache/rr_c10_clips_heldout.pt"),
               ("c08", "hexapod", f"{CW}/rr_c08_clips_heldout", "results/wm/cache/rr_c08_clips_heldout.pt"),
               ("B1", "b1", f"{CW}/rr_b1_clips_heldout", "results/wm/cache/test_v4_b1_heldout.pt")]


def floor(fa, fb, sigma):
    w = soft_targets(fa, fb, sigma)
    return float(-(w * w.clamp_min(1e-12).log()).sum(1).mean())


def main():
    ckpt = sys.argv[1]
    sigma, tau = float(os.environ.get("SIGMA", 0.25)), float(os.environ.get("TAU", 0.1))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Z, B, F, G, C = S.latents(ckpt, dev)
    Z = torch.as_tensor(Z.reshape(len(Z), -1), dtype=torch.float32)
    F = torch.as_tensor((F - F.mean(0)) / F.std(0), dtype=torch.float32)
    clips = np.unique(G)
    tr_clips = set(np.random.default_rng(0).permutation(clips)[: len(clips) // 2])
    tr = torch.tensor([g in tr_clips for g in G]); hx, b1 = torch.tensor(B == 0), torch.tensor(B == 2)
    sets = {k: (Z[m & hx], F[m & hx], Z[m & b1], F[m & b1]) for k, m in (("train", tr), ("test", ~tr))}
    torch.manual_seed(0)
    head = AlignHead(Z.shape[1]); opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    def loss(k):
        za, fa, zb, fb = sets[k]
        ua, ub = head(za), head(zb)
        return 0.5 * (soft_infonce(ua, fa, ub, fb, tau, sigma) + soft_infonce(ub, fb, ua, fa, tau, sigma))
    for it in range(2001):
        opt.zero_grad(); l = loss("train"); l.backward(); opt.step()
        if it % 500 == 0:
            with torch.no_grad():
                print(f"step {it:4d}  train {l.item():.3f}  test {loss('test').item():.3f}", flush=True)
    za, fa, zb, fb = sets["test"]
    print(f"test: c10 {len(za)} / B1 {len(zb)} transitions; sigma {sigma} tau {tau}")
    print(f"best possible (target entropy) {0.5 * (floor(fa, fb, sigma) + floor(fb, fa, sigma)):.3f}; "
          f"chance {0.5 * (np.log(len(zb)) + np.log(len(za))):.3f}")


if __name__ == "__main__":
    main()
