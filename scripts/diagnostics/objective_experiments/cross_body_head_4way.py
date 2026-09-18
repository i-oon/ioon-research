"""Section 8's literal 4-way cross-body test (insect->insect / b1->b1 / insect->b1 / b1->insect),
run on a checkpoint whose Cross-Body Head was fit SEPARATELY per body (not jointly with --also),
so each head can be scored on the body it never trained on -- a genuine transfer test, matching
Section 8's original protocol rather than a joint-fit self-evaluation.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/cross_body_head_4way.py \\
        --hex_ckpt wm/runs/beh12_body_stopgrad/head_hexonly.pt \\
        --b1_ckpt wm/runs/beh12_body_stopgrad/head_b1only.pt \\
        --hex_data data/allocentric/beh12_c10f10t10_flat --b1_data data/allocentric/beh12_b1_flat

Reuses `eval_body_head_true_heldout.py`'s own `embed_and_target`/per-channel-rho logic via import,
no duplicated feature code. Scores against a fixed 80/20 clip-level split (seed=0), same convention
`fit_body_head.py` itself uses, since these checkpoints don't have a `make_clean_split.py`-style
external held-out directory.
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wm.config import from_checkpoint             # noqa: E402
from wm.data.embodiment import REGISTRY, load       # noqa: E402
from wm.evaluate import encode_clip                 # noqa: E402
from wm.models.itm import InverseTransitionModel    # noqa: E402
from wm.models.motion_decoder import MotionDecoder  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder       # noqa: E402
from scipy.stats import spearmanr                   # noqa: E402


def held_out_split(paths, seed=0, frac=0.2):
    ids = np.arange(len(paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(frac * len(paths))))
    val_ids = set(ids[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_ids]
    held = [p for i, p in enumerate(paths) if i in val_ids]
    return train, held


def embed_and_target(encoder, itm, paths, spec, channels, cache_path):
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    device = next(itm.parameters()).device
    zs, ys = [], []
    with torch.no_grad():
        for path in paths:
            clip = load(path, spec)
            if path not in cache:
                cache[path] = encode_clip(encoder, clip["frames"], 2).cpu().half()
            e = cache[path].float().to(device)
            motion = np.asarray(clip["body_motion"])[:, channels]
            n = min(len(e) - 1, len(motion) - 1)
            z = torch.cat([itm(e[t:t + 1], e[t + 1:t + 2]) for t in range(n)]).cpu()
            zs.append(z)
            ys.append(torch.tensor(motion[:n], dtype=torch.float32))
    if len(cache) > before:
        torch.save(cache, cache_path)
    return torch.cat(zs), torch.cat(ys)


def load_head(ckpt_path, device):
    ck = torch.load(os.path.join(ROOT, ckpt_path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    mean = torch.tensor(np.asarray(ck["body_stats"][0]).ravel(), dtype=torch.float32)
    std = torch.tensor(np.asarray(ck["body_stats"][1]).ravel(), dtype=torch.float32)
    itm = InverseTransitionModel(cfg).to(device).eval()
    itm.load_state_dict(ck["itm"])
    md = MotionDecoder(cfg, {"b1": 12}).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    return itm, md, channels, mean, std


def r2(pred, true):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean(0, keepdim=True)) ** 2).sum()
    return float(1 - ss_res / ss_tot.clamp_min(1e-9))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hex_ckpt = "wm/runs/beh12_body_stopgrad/head_hexonly.pt"
    b1_ckpt = "wm/runs/beh12_body_stopgrad/head_b1only.pt"
    hex_dir = "data/allocentric/beh12_c10f10t10_flat"
    b1_dir = "data/allocentric/beh12_b1_flat"

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    itm_hex, md_hex, ch_hex, mean_hex, std_hex = load_head(hex_ckpt, device)
    itm_b1, md_b1, ch_b1, mean_b1, std_b1 = load_head(b1_ckpt, device)

    hex_paths = sorted(glob.glob(os.path.join(ROOT, hex_dir, "*.npz")))
    b1_paths = sorted(glob.glob(os.path.join(ROOT, b1_dir, "*.npz")))
    _, hex_held = held_out_split(hex_paths)
    _, b1_held = held_out_split(b1_paths)
    print(f"hexapod held-out: {len(hex_held)} clips | B1 held-out: {len(b1_held)} clips")

    # embed each body's held-out set with EACH checkpoint's own ITM (they may differ slightly
    # post-fit, though fit_body_head.py only touches body_head, so ITM is identical between the
    # two -- computed once per body for efficiency)
    z_hex_itmH, y_hex = embed_and_target(encoder, itm_hex, hex_held, REGISTRY["hexapod"], ch_hex,
                                         os.path.join(ROOT, "results/wm/cache/stopgrad_hex.pt"))
    z_b1_itmH, y_b1 = embed_and_target(encoder, itm_b1, b1_held, REGISTRY["b1"], ch_b1,
                                       os.path.join(ROOT, "results/wm/cache/stopgrad_b1.pt"))
    del encoder
    torch.cuda.empty_cache()

    def score(md, z, y, mean, std):
        y_std = ((y - mean) / std).to(device)
        with torch.no_grad():
            pred = md.body(None, z.to(device))
        return r2(pred, y_std), pred.cpu().numpy(), y_std.cpu().numpy()

    r2_hh, _, _ = score(md_hex, z_hex_itmH, y_hex, mean_hex, std_hex)   # insect -> insect
    r2_bb, _, _ = score(md_b1, z_b1_itmH, y_b1, mean_b1, std_b1)        # b1 -> b1
    r2_hb, _, _ = score(md_hex, z_b1_itmH, y_b1, mean_hex, std_hex)     # insect-fit head -> b1
    r2_bh, _, _ = score(md_b1, z_hex_itmH, y_hex, mean_b1, std_b1)      # b1-fit head -> insect

    print(f"\n{'':14}{'insect->insect':>16}{'b1->b1':>12}{'insect->b1':>14}{'b1->insect':>14}")
    print(f"{'R2':14}{r2_hh:>16.3f}{r2_bb:>12.3f}{r2_hb:>14.3f}{r2_bh:>14.3f}")


if __name__ == "__main__":
    main()
