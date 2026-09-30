"""Can one egocentric frame's V-JEPA2 embedding tell the hexapod from the B1? (render-confound check)

Grouped-CV logistic probe (PCA 64 of the token-mean embedding), c10 vs B1, on the same frames as the
F275 probe (24 clips each, every 4th frame): c10 from `anova_hex_beh24val.pt`, the old B1 renders from
`selection_eval_cands.pt`, and the re-rendered B1 (`data/egocentric_v3/beh12_b1_ego_flat_cleantrain`)
encoded here -- on the CPU by default, float32 like the caches, so it can run beside a training job.
Whole frame and four row bands of the 16x16 token grid. Target for matched rendering: near 0.5.

    .venv/bin/python3 scripts/diagnostics/egocentric_view/body_id_render_check.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

REGIONS = {"whole frame": slice(0, 16), "top rows (ceiling)": slice(0, 6), "wall": slice(6, 9),
           "floor/wall edge": slice(9, 11), "floor": slice(11, 16)}


def probe(A, B):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = {}
    for name, sl in REGIONS.items():
        X, Y, G = [], [], []
        for bi, clips in enumerate((A, B)):
            for ci, e in enumerate(clips):
                X.append(e[:, sl].mean((1, 2)).numpy()); Y += [bi] * len(e); G += [f"{bi}_{ci}"] * len(e)
        X = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(np.concatenate(X)))
        Y, G = np.array(Y), np.array(G)
        acc = [(LogisticRegression(max_iter=3000).fit(X[tr], Y[tr]).predict(X[te]) == Y[te]).mean()
               for tr, te in GroupKFold(5).split(X, Y, G)]
        out[name] = float(np.mean(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--new_dir", default="data/egocentric_v3/beh12_b1_ego_flat_cleantrain")
    args = ap.parse_args()
    grid = lambda e: e.float().reshape(-1, 16, 16, e.shape[-1])
    C = torch.load(os.path.join(ROOT, "results/wm/cache/anova_hex_beh24val.pt"), map_location="cpu", mmap=True)
    hexa = [grid(e[::4]) for _, e in list(C.items())[:24]]
    O = torch.load(os.path.join(ROOT, "results/wm/cache/selection_eval_cands.pt"), map_location="cpu", mmap=True)
    old = [grid(e[::4]) for _, e in list(O.items())[:24]]
    enc = VJEPA2FrameEncoder(device=args.device, dtype=torch.float32)
    new = []
    for i, p in enumerate(sorted(glob.glob(os.path.join(ROOT, args.new_dir, "*.npz")))[:24]):
        frames = np.load(p, allow_pickle=True)["frames"][::4]
        new.append(grid(encode_clip(enc, frames, 4).cpu()))
        print(f"  encoded {i + 1}/24", flush=True)
    for label, B in (("old B1 renders", old), ("re-rendered B1", new)):
        r = probe(hexa, B)
        print(f"c10 vs {label}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    print("BODY_ID_CHECK_DONE")


if __name__ == "__main__":
    main()
