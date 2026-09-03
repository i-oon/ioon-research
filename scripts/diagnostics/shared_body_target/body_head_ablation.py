"""Does the body-motion term constrain the latent, or does the head read the frame instead?

`L_body` exists to force `z` to mean the same thing on both robots. It can only do that if `z` is
the head's *only* route to the answer. Give the head the frame as well and it may take the shortcut:
body speed is readable from a single still image -- the frozen encoder scores R^2 0.676 on it -- so
the head can satisfy the loss while leaving `z` untouched, and the term becomes decorative.

That is not a design question to settle by argument. Zero one input at a time and see which one the
loss actually depends on.

    real z      the head as trained
    zero z      all-zero latent, frame intact. If the loss barely moves, `z` was carrying nothing
                and the term never constrained the latent.
    zero frame  all-zero visual tokens, latent intact. If the loss barely moves, the frame was
                carrying nothing and the head is effectively latent-only.

Measured on held-out clips, in the standardised units the loss itself uses, so 1.0 is what
predicting the training mean would score.

  .venv/bin/python3 scripts/diagnostics/shared_body_target/body_head_ablation.py --ckpt wm/runs/<run>/last.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import BODY_CHANNELS, REGISTRY, body_motion  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

DT = {"hexapod": 0.05, "b1": 0.02}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--insect_dir", default="data/allocentric/fwd_hex7speed")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    stats = ck.get("body_stats")
    if stats is None:
        raise SystemExit("checkpoint has no body_stats; it was not trained with lambda_body")
    mean, std = float(np.asarray(stats[0]).ravel()[0]), float(np.asarray(stats[1]).ravel()[0])

    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    heads = {"hexapod": 18, "b1": 12}
    md = MotionDecoder(cfg, heads=heads).to(device).eval()
    md.load_state_dict(ck["md"])
    if md.body_head is None:
        raise SystemExit("this checkpoint's decoder has no shared body head")

    encoder = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    rows = {"real z": [], "zero z": [], "zero frame": []}
    for name, pattern, spec in (("hexapod", f"{args.insect_dir}/*.npz", "hexapod"),
                                ("b1", "data/allocentric/fwd_b1_50hz/*.npz", "b1")):
        paths = sorted(p for p in glob.glob(os.path.join(ROOT, pattern))
                       if "manifest" not in os.path.basename(p))[-args.clips:]
        for path in paths:
            clip = np.load(path, allow_pickle=True)
            pos = clip["head"] if "head" in clip.files else clip["base_pos"]
            target = body_motion(pos.astype(np.float64), DT[name])[:, BODY_CHANNELS]
            e = encode_clip(encoder, clip["frames"], 4).to(device)
            off = offset_for(ck, name)
            if off is not None:
                e = e - off.to(device)
            n = len(e) - 1
            with torch.no_grad():
                z = torch.cat([itm(e[t:min(t + 8, n)], e[t + 1:min(t + 8, n) + 1])
                               for t in range(0, n, 8)])
                y = torch.tensor((target[:n] - mean) / std, device=device, dtype=torch.float32)
                for label, xt, zz in (("real z", e[:n], z),
                                      ("zero z", e[:n], torch.zeros_like(z)),
                                      ("zero frame", torch.zeros_like(e[:n]), z)):
                    pred = md.body(xt, zz)
                    rows[label].append(float(((pred - y) ** 2).mean()))

    base = float(np.mean(rows["real z"]))
    print(f"\n  {args.ckpt}")
    print(f"\n{'input':<14}{'body loss':>12}{'vs real z':>12}")
    for label in ("real z", "zero z", "zero frame"):
        v = float(np.mean(rows[label]))
        print(f"  {label:<12}{v:>12.4f}{v / base:>11.2f}x")
    print("\n  1.0 in the loss column is what predicting the training mean scores.")
    print("  A `zero z` row close to `real z` means the head reads the frame and the term")
    print("  never constrained the latent -- which is what it exists to do.")


if __name__ == "__main__":
    main()
