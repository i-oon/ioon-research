"""Is z shared in the ALIGNED space g(z)? (arm J follow-up, 2026-10-07)

The contrastive alignment of arm J acts on a projection head u = g(z), so that z itself can keep body-specific detail; the
shared-latent tests measure z. This runs the same tests (body-ID probe, cross-body R2, k-NN mixing, retrieval;
scripts/figures/shared_latent_figure.numbers) on z and on u = g(z) of the same held-out random-room clips.

    .venv/bin/python3 scripts/diagnostics/cross_embodiment/shared_in_align_space.py wm/runs/round2_align_s0_rr/best.pt
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/figures", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
import shared_latent_figure as S  # noqa: E402
from wm.align import AlignHead  # noqa: E402

CW = "data/counterfactual_walks"
S.BODIES[:] = [("c10 (pretrain)", "hexapod", f"{CW}/rr_c10_clips_heldout", "results/wm/cache/rr_c10_clips_heldout.pt"),
               ("c08 (held-out)", "hexapod", f"{CW}/rr_c08_clips_heldout", "results/wm/cache/rr_c08_clips_heldout.pt"),
               ("B1", "b1", f"{CW}/rr_b1_clips_heldout", "results/wm/cache/test_v4_b1_heldout.pt")]


def fmt(name, res):
    acc, xfer, mix, retr = res
    f = lambda v: " / ".join(f"{x:+.2f}" for x in v)  # noqa: E731
    return (f"{name:<8} body-ID {acc:.2f} | R2 c10->c08 {f(xfer[1])} | R2 c10->B1 {f(xfer[2])} | "
            f"kNN mix {mix:.2f} | retrieval c08 {retr[1]:.2f} B1 {retr[2]:.2f}")


def main():
    ckpt = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Z, B, F, G, C = S.latents(ckpt, dev)
    ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
    if "align" not in ck:
        raise SystemExit("checkpoint has no alignment head")
    sd = ck["align"]
    head = AlignHead(Z.reshape(len(Z), -1).shape[1], dim=sd["net.1.weight"].shape[0])
    head.load_state_dict(sd); head.eval()
    with torch.no_grad():
        U = head(torch.as_tensor(Z).reshape(len(Z), -1)).numpy()
    rng = np.random.default_rng(0)
    lines = [f"{ckpt}  (held-out random-room clips; same tests as the shared-latent page)",
             fmt("z", S.numbers(Z.reshape(len(Z), -1), B, F, G, np.random.default_rng(0))),
             fmt("g(z)", S.numbers(U, B, F, G, rng))]
    txt = "\n".join(lines); print(txt)
    out = os.path.join(ROOT, "results/eval/round2_align_s0_rr/shared_in_align_space.txt")
    open(out, "w").write(txt + "\n"); print("->", out)


if __name__ == "__main__":
    main()
