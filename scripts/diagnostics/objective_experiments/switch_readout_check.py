"""At a behaviour switch, does the read-out report the NEW motion or the OLD one? (F266 on real data)

Switching babble (`collect_babble_hex.py`) contains natural counterfactuals: at a switch the start
frame belongs to the old behaviour and the next k frames to the new one. For every start t in the
held-out babble clips (not used by the adapted checkpoint's Stage 1):
  reading   body(ITM(e_t, e_{t+k}))                   k = the checkpoint's stride
  now       true Froude over [t, t+k)                 what the reading should report
  prev      true Froude over [t-k, t)                 what a start-state-driven reading reports
On "switch" samples (top 30% of |now - prev|) and on steady ones, R2 of the reading against now and
against prev, per channel. A reader driven by its start frame tracks prev on switch samples.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/switch_readout_check.py \\
        --ckpt zeroshot=wm/runs/beh24_stride5_cleansplit/c08_zeroshot/ckpt_lib_zeroshot.pt \\
        --ckpt babble=wm/runs/beh24_stride5_cleansplit/c08_babble/ckpt_s12.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

CH = ("fwd", "lat", "yaw")


def r2(p, y):
    return 1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--data", default="data/egocentric/babble_c08f09t09_flat")
    ap.add_argument("--adapted", default="wm/runs/beh24_stride5_cleansplit/c08_babble/adapted.pt",
                    help="its Stage-1 train_paths are excluded")
    ap.add_argument("--cache", default="results/wm/cache/babble_c08.pt")
    args = ap.parse_args()
    train = set(torch.load(os.path.join(ROOT, args.adapted), map_location="cpu",
                           weights_only=False)["adapted"]["train_paths"])
    paths = [p for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
             if os.path.basename(p) not in train]
    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path) if os.path.exists(cache_path) else {}
    if any(p not in cache for p in paths):
        from vjepa2_encoder import VJEPA2FrameEncoder
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        for p in paths:
            if p not in cache:
                cache[p] = encode_clip(enc, load(p, REGISTRY["hexapod"])["frames"], 2).cpu().half()
        del enc
        torch.save(cache, cache_path)
    print(f"{len(paths)} held-out babble clips")
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), "hexapod", 18, "cuda")
        ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        k = m.stride
        R, NOW, PREV = [], [], []
        for p in paths:
            e = cache[p].float().cuda()
            bm = np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3]
            ts = list(range(k, len(e) - k))
            z = m.itm(e[ts], e[[t + k for t in ts]])
            R.append(m.md.body(None, z).cpu().numpy() * std + mean)
            NOW.append(np.stack([bm[t:t + k].mean(0) for t in ts]))
            PREV.append(np.stack([bm[t - k:t].mean(0) for t in ts]))
        R, NOW, PREV = map(np.concatenate, (R, NOW, PREV))
        change = np.linalg.norm((NOW - PREV) / NOW.std(0), axis=1)
        sw = change >= np.percentile(change, 70)
        print(f"\n=== {name} (stride {k}), {len(R)} samples, {sw.sum()} switch samples")
        for tag, msk in (("switch", sw), ("steady", ~sw)):
            print(f"  {tag:<7} R2 vs NOW  " + " / ".join(f"{v:+.2f}" for v in r2(R[msk], NOW[msk]))
                  + "   R2 vs PREV " + " / ".join(f"{v:+.2f}" for v in r2(R[msk], PREV[msk])))
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
