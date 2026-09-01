"""Does the shared body head read a robot's motion, per channel, on each robot separately?

**The check F129 exists because nobody ran it.** The head was used to score candidates on the B1
(F128) and read as compressed and too narrow. Measured on the body it trained on it is exact --
correlation +0.99, compression 1.0x -- and on the B1 it returns the dataset mean for every
behaviour, because `beh12_hexonly` is a hexapod-only pretrain and neither `wm/adapt.py` nor
`wm/adapt3.py` adapts the motion decoder. **Run this before scoring anything with a body head.**

    predicted   body_head(ITM(e_t, e_t+1)), un-standardised with the checkpoint's own body_stats
    measured    the clip's own body motion, from its recorded trajectory

    compression   spread of measured / spread of predicted, over the 5th-95th percentiles.
                  1.0 is a head that reproduces the range; large means it answers near-constantly.

**Report both robots and every channel.** A head can calibrate on forward and return a constant on
yaw -- that is F83's channel competition, and it is a different result from "the head does not
work". The pass bar fixed in F129 is **compression under about 1.5x on every channel of both
robots**.

    .venv/bin/python3 scripts/diagnostics/body_head_calibration.py \\
        --ckpt wm/runs/beh12_hexonly/best.pt \\
        --data hexapod=data/allocentric/beh12_c10f10t10_flat b1=data/allocentric/beh12_b1_flat
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.adapt3 import FAMILY, gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402

NAMES = ("forward", "lateral", "yaw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True, metavar="EMBODIMENT=DIR")
    ap.add_argument("--cache_dir", default="results/wm/cache")
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--stride", type=int, default=10, help="frames between samples within a clip")
    ap.add_argument("--only", nargs="*", default=[],
                    help="clip basenames to score, ignoring the rest. **Use the held-out list when "
                         "the head has been fitted** (`wm/fit_body_head`), or the number is read "
                         "off clips the head was trained on and means nothing.")
    ap.add_argument("--held_out", action="store_true",
                    help="score only the clips a `wm/fit_body_head` run held out, read from the "
                         "checkpoint's own record so the split cannot drift")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean = np.asarray(ck["body_stats"][0]).ravel()
    std = np.asarray(ck["body_stats"][1]).ravel()

    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    print(f"{args.ckpt}: body_dim {cfg.body_dim}, channels {channels} "
          f"({', '.join(NAMES[c] for c in channels)})\n")
    for spec in args.data:
        name, directory = spec.split("=", 1)
        cache_path = os.path.join(ROOT, args.cache_dir, f"bodycal_{name}.pt")
        cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
        before = len(cache)
        clips = gather(os.path.join(ROOT, directory), name, encoder, ck, cache,
                       args.chunk, max(1, cfg.action_lag), device)
        keep = set(args.only)
        if args.held_out:
            fit = ck.get("body_head_fit") or {}
            if fit.get("embodiment") == name:
                keep |= set(fit.get("val_paths", []))
                print(f"  {name}: scoring the {len(fit.get('val_paths', []))} clips the head fit "
                      f"held out")
            else:
                print(f"  {name}: not the fitted embodiment, scoring every clip")
        if keep:
            clips = [c for c in clips if c["path"] in keep] or clips
        if len(cache) > before:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(cache, cache_path)

        # a motion decoder needs an action width per embodiment; only the body head is used here
        md = MotionDecoder(cfg, {name: clips[0]["a"].shape[1]}).to(device).eval()
        md.load_state_dict(ck["md"], strict=False)
        if md.body_head is None:
            raise SystemExit("this checkpoint has no body head (lambda_body 0)")

        truth, pred, fams = [], [], []
        with torch.no_grad():
            for c in clips:
                motion = load(os.path.join(ROOT, directory, c["path"]),
                              REGISTRY[name])["body_motion"]
                e = c["e"].float().to(device)
                for t in range(5, min(c["n"], len(motion) - 1), args.stride):
                    z = itm(e[t:t + 1], e[t + 1:t + 2])
                    p = md.body(None, z).squeeze(0).cpu().numpy() * std + mean
                    truth.append(np.asarray(motion[t])[channels])
                    pred.append(np.atleast_1d(p))
                    fams.append(FAMILY(c["cond"]))
        truth, pred = np.array(truth), np.array(pred)

        print(f"  {name}  ({len(clips)} clips, {len(truth)} samples)")
        print(f"    {'channel':<10}{'corr':>8}{'measured':>11}{'predicted':>11}{'compression':>13}")
        for j, ch in enumerate(channels):
            t_lo, t_hi = np.percentile(truth[:, j], [5, 95])
            p_lo, p_hi = np.percentile(pred[:, j], [5, 95])
            comp = (t_hi - t_lo) / max(p_hi - p_lo, 1e-9)
            flag = "" if comp < 1.5 else "   <- fails the 1.5x bar"
            print(f"    {NAMES[ch]:<10}{np.corrcoef(truth[:, j], pred[:, j])[0, 1]:>+8.2f}"
                  f"{t_hi - t_lo:>11.3f}{p_hi - p_lo:>11.3f}{comp:>12.1f}x{flag}")
        by = collections.defaultdict(lambda: [[], []])
        for t_, p_, f in zip(truth, pred, fams):
            by[f][0].append(t_); by[f][1].append(p_)
        print(f"    {'family':<10}" + "".join(f"{NAMES[ch]+' t/p':>18}" for ch in channels))
        for f, v in sorted(by.items()):
            row = np.mean(v[0], axis=0), np.mean(v[1], axis=0)
            print(f"    {f:<10}" + "".join(f"{row[0][j]:>9.3f} /{row[1][j]:>7.3f}"
                                           for j in range(len(channels))))
        print()

    print("  compression is the measured range over the predicted range, 5th-95th percentile.")
    print("  **A head that has never seen an embodiment returns that embodiment's mean** and shows")
    print("  a large compression with a near-zero correlation -- which is F129, not a broken head.")


if __name__ == "__main__":
    main()
