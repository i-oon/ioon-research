"""Review renders for a babble set (`collect_babble_hex.py`): coverage against beh24, per-clip
top-down paths coloured by segment, and ego-view clips with the commanded drive and the achieved
Froude burned in.

    .venv/bin/python3 scripts/figures/render_babble_review.py --babble data/egocentric/babble_c08f09t09
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import REGISTRY, load  # noqa: E402

CH = ("forward", "lateral", "yaw")


def mode_of(d, t):
    if d["plan_a0"][t] < 0.125:
        return "side L" if d["plan_strafe"][t] < 0 else "side R"
    return "backward" if d["plan_lead"][t] > 0.5 else "forward"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--babble", required=True)
    ap.add_argument("--reference", default="data/egocentric/beh24_c10f10t10_ego_flat_cleantrain")
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--out", default="results/deck/babble_review")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(ROOT, args.babble, "clip*", "*.npz")))
    bm = [np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3] for p in paths]
    ref = np.concatenate([np.asarray(load(p, REGISTRY["hexapod"])["body_motion"])[:, :3]
                          for p in sorted(glob.glob(os.path.join(ROOT, args.reference, "*.npz")))])
    B = np.concatenate(bm)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for j in range(3):
        lo, hi = np.percentile(np.concatenate([ref[:, j], B[:, j]]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax[j].hist(ref[:, j], bins, alpha=0.5, density=True, color="#888", label="beh24 (c10, pretrain)")
        ax[j].hist(B[:, j], bins, alpha=0.6, density=True, color="#1565c0", label=f"babble ({len(paths)} clips)")
        p_r, p_b = np.percentile(ref[:, j], [2, 98]), np.percentile(B[:, j], [2, 98])
        ax[j].set_title(f"{CH[j]}: beh24 {p_r[0]:+.2f}..{p_r[1]:+.2f} | babble {p_b[0]:+.2f}..{p_b[1]:+.2f}", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.suptitle("Froude coverage (body_motion), 2-98 percentile in titles")
    fig.tight_layout(); fig.savefig(os.path.join(out, "coverage.png"), dpi=110); plt.close(fig)

    colours = {"forward": "#2e7d32", "backward": "#c62828", "side L": "#1565c0", "side R": "#6a1b9a"}
    n = min(12, len(paths))
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for k, p in enumerate(paths[:n]):
        d = np.load(p, allow_pickle=True)
        h = d["head"]
        a = axes.flat[k]
        for t in range(len(h) - 1):
            a.plot(h[t:t + 2, 0], h[t:t + 2, 1], color=colours[mode_of(d, t)], lw=2)
        a.plot(*h[0, :2], "ko", ms=4)
        a.set_aspect("equal"); a.set_title(os.path.basename(os.path.dirname(p)), fontsize=9)
    for m, c in colours.items():
        axes.flat[0].plot([], [], color=c, label=m)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Top-down head path per clip, coloured by commanded segment (black dot = start)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "paths.png"), dpi=100); plt.close(fig)

    import cv2
    import imageio.v2 as imageio
    for k, p in enumerate(paths[:args.videos]):
        d = np.load(p, allow_pickle=True)
        fr, m = d["frames"], bm[k]
        video = []
        for t in range(len(fr)):
            img = cv2.resize(np.ascontiguousarray(fr[t]), (384, 384))
            bar = np.full((70, 384, 3), 20, np.uint8)
            cv2.putText(bar, f"t {t:2d}  cmd: {mode_of(d, t)}  spin {d['plan_spin'][t]:+.2f}  "
                             f"pace {d['plan_pace'][t]:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            ti = min(t, len(m) - 1)
            cv2.putText(bar, f"achieved Froude  fwd {m[ti, 0]:+.3f}  lat {m[ti, 1]:+.3f}  yaw {m[ti, 2]:+.3f}",
                        (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 230, 150), 1, cv2.LINE_AA)
            video.append(np.vstack([img, bar]))
        imageio.mimwrite(os.path.join(out, f"{os.path.basename(os.path.dirname(p))}.mp4"),
                         np.stack(video), fps=10, codec="libx264", quality=8, macro_block_size=1)
    print("->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
