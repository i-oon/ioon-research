"""Adapt a pretrained ITM and forward model to a new robot, and save a checkpoint to plan with.

**This is LAC-WM's stage 1, and until now it existed only as a diagnostic.**
`scripts/diagnostics/finetune_ftm.py` adapts, scores and throws the weights away, which answers
"how much does N clips buy" and leaves nothing to run a controller on. F96 needed exactly that: the
cross-embodiment planner defaults instead of selecting, and the measurement located the cause in
the forward model's ignorance of the target robot rather than in the action projector -- which is
untestable without an adapted forward model saved to disk.

    stage 1     this file: fine-tune the ITM and FDM on N clips of the new robot
    stage 2     `wm.fit_projector` against the *adapted* ITM, since stage 1 moved what `z` means
    stage 3     joint fine-tuning of projector and FDM -- still not built

**Only the ITM and forward model move.** The encoder stays frozen as everywhere else in this
project, and the motion decoder is not adapted at all: it is an auxiliary loss during pretraining
and plays no part at control time, so adapting it would cost time for a module the planner never
calls. The saved checkpoint carries the original decoder weights unchanged so that scripts
rebuilding a full model from it still work.

  .venv/bin/python3 -m wm.adapt --ckpt wm/runs/beh12_hexonly/best.pt \\
      --data data/beh12_b1_v2 --embodiment b1 --clips 9 --out wm/runs/beh12_hexonly/adapted_b1.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from diagnostics.finetune_ftm import adapt, embeddings_for, rollout  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="clips of the robot to adapt to")
    ap.add_argument("--embodiment", default="b1")
    ap.add_argument("--clips", type=int, default=9,
                    help="how many clips of the new robot the adaptation may use. **This is the "
                         "few-shot budget and it is the claim**: slide 15 measures three clips "
                         "clearing break-even where starting cold never does.")
    ap.add_argument("--test_clips", type=int, default=10,
                    help="held out of adaptation, used only to report the rollout ratio")
    ap.add_argument("--steps", type=int, default=1000, help="optimiser updates, not epochs")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = from_checkpoint(checkpoint["config"])

    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    if args.clips + args.test_clips > len(paths):
        raise SystemExit(f"{args.clips} adapt + {args.test_clips} test exceeds {len(paths)} clips; "
                         "the two sets would overlap and the rollout ratio would be measured "
                         "partly on clips the model was fitted on")
    order = np.random.default_rng(args.seed).permutation(len(paths))
    train = [paths[i] for i in order[:args.clips]]
    test = [paths[i] for i in order[-args.test_clips:]]

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    train_e = embeddings_for(encoder, train, args.chunk)
    test_e = embeddings_for(encoder, test, args.chunk)
    del encoder
    torch.cuda.empty_cache()

    itm = InverseTransitionModel(cfg).to(device)
    ftm = ForwardTransitionModel(cfg).to(device)
    itm.load_state_dict(checkpoint["itm"])
    ftm.load_state_dict(checkpoint["ftm"])

    before, _ = rollout(itm, ftm, test_e, args.horizons, device)
    loss = adapt(itm, ftm, train_e, args.steps, args.lr, args.seed, device)
    after, moved = rollout(itm, ftm, test_e, args.horizons, device)

    print(f"\nadapted on {len(train)} clips of {args.embodiment}, {args.steps} updates, "
          f"final loss {loss:.4f}")
    print(f"{'horizon':>9}{'before':>9}{'after':>9}{'moves':>9}")
    for h in args.horizons:
        print(f"{h:>9}{before[h]:>9.2f}{after[h]:>9.2f}{moved[h]:>9.2f}")
    print("\nRatios are against holding the frame still; 1.0 means no better than predicting no")
    print("motion. `moves` is predicted displacement over actual -- near 0 would be a model that")
    print("learned to sit still, which scores 1.0 by construction.")

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # The decoder and the statistics travel unchanged: nothing here adapted them, and a checkpoint
    # missing them would break every script that rebuilds a full model from one.
    saved = {"config": checkpoint["config"], "epoch": -1,
             "itm": itm.state_dict(), "ftm": ftm.state_dict(), "md": checkpoint["md"],
             "adapted": {"embodiment": args.embodiment, "clips": len(train),
                         "steps": args.steps, "source": args.ckpt,
                         "train_paths": [os.path.basename(p) for p in train]}}
    for key in ("action_stats", "body_stats", "action_mean", "action_std", "offsets"):
        if key in checkpoint:
            saved[key] = checkpoint[key]
    torch.save(saved, out)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
