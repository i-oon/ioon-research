"""Encode the Step 0 pilot frames with frozen V-JEPA2 -> e_t.

Each frame is encoded independently via the duplicated-frame trick
(VJEPA2FrameEncoder), giving (256, 1408) patch tokens, then mean-pooled to a
single 1408-d whole-frame vector. Mean-pooling is deliberate: the per-patch
route was tested on three backgrounds and abandoned (see direction_plan.md
"Step 0 Check 3 ABANDONED") -- pooling over 256 patches averages out the
per-patch noise floor that killed it.

Usage:
  python scripts/step0_encode.py --data data/step0 --out data/step0/embeddings.npz
"""
import argparse
import glob
import os

import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder

GAIT_LOOP_LEN = 64   # length of the replayed CSV segment. Recorded for reference only; it is a
                     # trim length chosen by hand, not a natural gait period, and is no longer
                     # used to label behaviour. See PROGRESS.md 10.12.
CONTACT_THRESH = 0.5  # N; foot force above this = planted (stance). Lowered from 3.0 on 2026-07-21 so the
                      # planted-leg count (~3) is physically sane: at 3.0 N only ~2.3 legs register because
                      # front legs load lightly (~0.2-0.4 N median). This ONLY sets how many legs count as
                      # planted; it does NOT and cannot change the gait's timing. The gait is a wave /
                      # phase-staggered pattern, not a tripod, on both our replay and the mature expert
                      # (66k) -- that is the real Medauroidea gait and is not something to "fix". Contact
                      # decodability is flat (~0.85 macro-F1) across 0.5-3.0 N. See report/NUMBERS.md 3.1.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/step0")
    ap.add_argument("--out", type=str, default="data/step0/embeddings.npz")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*_ep*.npz")))
    print(f"found {len(files)} episodes")

    encoder = VJEPA2FrameEncoder()

    E, morph, episode, step_idx, forces = [], [], [], [], []
    for f in files:
        tag = os.path.basename(f).replace(".npz", "")
        m, ep = tag.rsplit("_ep", 1)
        d = np.load(f)
        frames = d["frames"]

        pooled = []
        for i in range(0, len(frames), args.batch):
            e = encoder.encode(list(frames[i:i + args.batch]))   # (B, 256, 1408)
            pooled.append(e.mean(dim=1).float().cpu().numpy())   # -> (B, 1408)
        pooled = np.concatenate(pooled, axis=0)

        E.append(pooled)
        morph += [m] * len(pooled)
        episode += [int(ep)] * len(pooled)
        step_idx.append(d["step_idx"])
        forces.append(d["forces"] if "forces" in d else np.zeros((len(pooled), 6), np.float32))
        print(f"  {tag:12s} -> {pooled.shape}")

    E = np.concatenate(E, axis=0)
    step_idx = np.concatenate(step_idx)
    F = np.concatenate(forces, axis=0)            # (N, 6) raw foot forces

    # --- 6-bit foot contact: the behaviour label (threshold at 3 N) ---
    # The old alternative, phase = (step % 64) // 8, is not produced any more. 64 is the length
    # of the segment trimmed out of the animal recording rather than a natural gait period, the
    # loop seam jumps 14.75 degrees, and identical commands do not put different bodies in the
    # same pose. It measured the clock, not the body. See PROGRESS.md 10.12 and 10.13.
    contact = (F > CONTACT_THRESH).astype(int)    # (N, 6): which feet are planted
    contact_code = contact.dot(1 << np.arange(6)) # 0..63, one integer per pattern
    n_support = contact.sum(axis=1)               # 0..6, how many feet planted

    np.savez_compressed(
        args.out,
        e=E, morph=np.array(morph), episode=np.array(episode),
        step_idx=step_idx,
        forces=F, contact=contact, contact_code=contact_code, n_support=n_support,
    )
    print(f"\ne_t: {E.shape}  (dim={E.shape[1]})")
    print(f"morphologies: {sorted(set(morph))}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
