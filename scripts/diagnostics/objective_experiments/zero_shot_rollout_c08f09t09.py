"""Zero-shot FTM rollout on the held-out leg-length body (c08f09t09), no adaptation at all.

F43/F44 already measured this pattern for a much bigger morphology gap (insect-trained FTM rolled
on B1 video): 0.57-0.71x, worse than predicting no motion. c08f09t09 is a far smaller gap (same
species, different leg-segment ratios, not a different robot family) -- this checks whether that
smaller gap is small enough for zero-shot rollout to actually work, and whether beh24's confirmed
probe-R2 gain (0.492->0.530 on beh12's own held-out conditions) also shows up here, on a body
neither checkpoint has ever seen.

Uses `rollout()` directly from `finetune_ftm.py` -- no adaptation step, frozen ITM+FTM only.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/zero_shot_rollout_c08f09t09.py
"""
import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "cross_embodiment"))

from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from finetune_ftm import build, rollout  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data/egocentric/beh12_c08f09t09_ego_flat")
HORIZONS = [1, 3, 5, 10]
CHECKPOINTS = {
    "beh12 (12 conditions)": os.path.join(ROOT, "wm/runs/beh12_hinge_cleansplit/best.pt"),
    "beh24 (24 conditions)": os.path.join(ROOT, "wm/runs/beh24_hinge_cleansplit/best.pt"),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    print(f"{len(paths)} c08f09t09 clips (held-out leg-length body, zero-shot, no adaptation)\n")

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = [encode_clip(encoder, __import__("numpy").load(p, allow_pickle=True)["frames"], 4).to(device)
             for p in paths]
    del encoder
    torch.cuda.empty_cache()

    print(f"{'checkpoint':<24}" + "".join(f"{'h=' + str(h):>9}" for h in HORIZONS))
    for name, ckpt in CHECKPOINTS.items():
        _, itm, ftm = build(ckpt, pretrained=True, device=device)
        itm.eval(); ftm.eval()
        clips_cpu = [c.cpu() for c in clips]
        ratio, moved = rollout(itm, ftm, clips_cpu, HORIZONS, device)
        print(f"{name:<24}" + "".join(f"{ratio[h]:>9.2f}x" for h in HORIZONS))
        print(f"{'  moves':<24}" + "".join(f"{moved[h]:>9.2f}" for h in HORIZONS))
        del itm, ftm
        torch.cuda.empty_cache()

    print("\nAbove 1.00x: FTM's closed-loop rollout beats holding the frame still on this")
    print("never-seen body. Below 1.00x: worse than predicting no motion at all.")
    print("`moves`: predicted displacement as a fraction of the real one -- near 0 with ratio")
    print("near 1.0 means the model learned to sit still, not the dynamics.")


if __name__ == "__main__":
    main()
