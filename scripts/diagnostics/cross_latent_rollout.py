"""Drive one robot's forward model with the other robot's latent, and see if the prediction holds.

LAC-WM's section 5.2, adapted. Condition the FDM on observations from one embodiment while feeding
it action embeddings derived from **another**, and score the predicted next embedding:

    e_hat = FTM( e_t from B1 ,  z_t from the hexapod )   vs   e_{t+1} from B1

**This is not F51.** That asked whether the *forward model* survives a change of robot, using each
robot's own latent, and measured 0.57-0.71x -- worse than predicting no motion. This asks whether
the **latent** transfers, which is what the shared body head is claimed to produce (F83). Different
component, different question.

**No decoder is needed and none exists.** LAC-WM reports PSNR/LPIPS/FID/FVD because they trained a
custom V-JEPA2 RGB decoder; their FDM predicts a visual embedding exactly as ours does. The
measurement itself lives in embedding space either way.

**Two baselines, because the raw number is uninterpretable.** The clips are not synchronised in gait
phase -- that is F45's pairing problem, and it does not go away here -- so a cross-robot latent is
mismatched in phase even if it is perfectly shared in behaviour. Bracketing it:

    own z         the same robot's own latent            an upper bound
    shuffled z    a latent from a random other clip      a lower bound, phase-matched to nothing

The cross-robot number is only meaningful as a position between those two. Reported as the fraction
of the gap closed: 1.0 means the other robot's latent is as good as your own, 0.0 means it is worth
no more than a random one.

  .venv/bin/python3 scripts/diagnostics/cross_latent_rollout.py --ckpt wm/runs/beh12_body_fwd/best.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.models.ftm import ForwardTransitionModel  # noqa: E402
from wm.models.itm import InverseTransitionModel  # noqa: E402


def latents(itm, e, chunk=8):
    n = len(e) - 1
    with torch.no_grad():
        return torch.cat([itm(e[t:min(t + chunk, n)], e[t + 1:min(t + chunk, n) + 1])
                          for t in range(0, n, chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hex_dir", default="data/allocentric/beh12_c10f10t10_flat")
    ap.add_argument("--b1_dir", default="data/allocentric/beh12_b1_flat",
                    help="**`beh12_b1_flat`, not `beh12_b1_flat`.** The old set clips the robot in 61% of frames, never pins its camera, files the forward clip under `turn_wz0.00`, and turns the opposite way from the insect (F113-F115).")
    ap.add_argument("--cache", default="results/wm/cache/beh12_embeddings.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, args.ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    enc = None

    def load(directory, name):
        nonlocal enc
        out = {}
        for p in sorted(glob.glob(os.path.join(ROOT, directory, "*.npz"))):
            if p not in cache:
                if enc is None:
                    enc = VJEPA2FrameEncoder(dtype=torch.float32)
                with np.load(p, allow_pickle=True) as c:
                    cache[p] = encode_clip(enc, c["frames"], 2).cpu().half()
            e = cache[p].float().to(device)
            off = offset_for(ck, name)
            if off is not None:
                e = e - off.to(device)
            # **Pair by the episode slot, not the condition name.** The two robots name their
            # conditions after their own commands -- `speed_c5.8` against `speed_vx0.30` -- so a
            # name match finds only the four sideways conditions, which happen to share a label,
            # and silently drops the other eight while reporting a number as if it were all of
            # them. `merge_behaviour_dirs.py` assigns `axis*1000 + condition*100 + clip` exactly so
            # the matched pair carries the same slot.
            slot = int(os.path.basename(p).split("_ep")[1][:-4]) // 100
            out.setdefault(slot, []).append(e)
        return out

    H, B = load(args.hex_dir, "hexapod"), load(args.b1_dir, "b1")
    rng = np.random.default_rng(args.seed)

    def err(e_ctx, z):
        n = min(len(z), len(e_ctx) - 1)
        with torch.no_grad():
            pred = ftm(e_ctx[:n], z[:n])
        return float(((pred - e_ctx[1:n + 1]) ** 2).mean())

    print(f"{'direction':<18}{'own z':>10}{'other z':>10}{'random z':>11}{'gap closed':>13}")
    for lbl, src, dst, dn in (("hexapod -> B1", H, B, "b1"), ("B1 -> hexapod", B, H, "hexapod")):
        own, other, rand = [], [], []
        conds = sorted(set(src) & set(dst))
        for cond in conds:
            for e_dst in dst[cond]:
                z_own = latents(itm, e_dst)
                e_src = src[cond][rng.integers(len(src[cond]))]
                z_other = latents(itm, e_src)
                bad = dst[sorted(set(conds) - {cond})[rng.integers(len(conds) - 1)]][0]
                own.append(err(e_dst, z_own))
                other.append(err(e_dst, z_other))
                rand.append(err(e_dst, latents(itm, bad)))
        o, x, r = np.mean(own), np.mean(other), np.mean(rand)
        # 1.0 = the other robot's latent is as useful as this robot's own; 0.0 = no better than random
        print(f"{lbl:<18}{o:>10.4f}{x:>10.4f}{r:>11.4f}{(r - x) / max(r - o, 1e-9):>13.3f}"
              f"   {len(conds)} conditions")

    if enc is not None:
        torch.save(cache, cache_path)
    print("\nThe clips are not phase-synchronised (F45), so the cross-robot latent is mismatched in")
    print("gait phase even where behaviour is shared. Read the gap-closed column, not the raw MSE.")


if __name__ == "__main__":
    main()
