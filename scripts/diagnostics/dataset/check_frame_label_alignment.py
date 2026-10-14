"""Are rendered frames and labels aligned in TIME? (2026-10-14, before the wall-data retrain)

For every file with a recorded camera pose (`cam_pose`, one per rendered frame) the camera's planar SPEED is compared
with the speed of the body part it is mounted on, read from the same pose rows the labels use (hexapod: `head`; B1:
`base_pos`), at lags -3..+3 frames (speed: independent of the room / world frame). Aligned data: best lag 0.
(First version compared the hexapod camera with the centre of mass: the head sways with every step, so lags spread
1-3 and correlation was ~0.5 -- a test artefact, not a timing error; 2026-10-14: head reference, lag 0 in 94-100 %.)
A consistent non-zero lag would mean frame t shows another time than label t (a bug, fixed before training); this is
separate from the question of which averaging window the Froude label should use (a design choice, own experiment).
Also checks: one frame per pose row, dt.

    .venv/bin/python3 scripts/diagnostics/dataset/check_frame_label_alignment.py [--dirs DIR ...] [--max 200]
"""
import argparse
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT = ["data/wall_b1/train", "data/wall_b1/test", "data/wall_c10/train", "data/wall_c10/test",
           "data/counterfactual_walks/b1_clips_train", "data/counterfactual_walks/c10_clips_train",
           "data/counterfactual_walks/b1_branches_train", "data/counterfactual_walks/c10_branches_train"]
LAGS = range(-3, 4)


def best_lag(cam, body):
    vc, vb = np.linalg.norm(np.diff(cam, axis=0), axis=1), np.linalg.norm(np.diff(body, axis=0), axis=1)
    vc, vb = vc - vc.mean(), vb - vb.mean()
    sc = {}
    for L in LAGS:                                    # frame t of the camera vs body t+L
        a = vc[max(0, -L):len(vc) - max(0, L)]; b = vb[max(0, L):len(vb) - max(0, -L)]
        if len(a) < 5:
            continue
        num = (a * b).sum(); den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
        sc[L] = num / den
    return max(sc, key=sc.get), sc.get(0, np.nan), max(sc.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT)
    ap.add_argument("--max", type=int, default=200, help="files per dir (evenly spaced)")
    a = ap.parse_args()
    bad_all = 0
    for d in a.dirs:
        fs = sorted(glob.glob(os.path.join(ROOT, d, "*.npz")))
        if not fs:
            print(f"{d}: no files"); continue
        fs = fs[::max(1, len(fs) // a.max)][:a.max]
        lags, c0, n_bad, n_skip, dts, rows = [], [], 0, 0, set(), 0
        for f in fs:
            x = np.load(f, allow_pickle=True)
            if "cam_pose" not in x.files:
                n_skip += 1; continue
            key = "head" if "head" in x.files else ("base_pos" if "base_pos" in x.files else "com_pos")
            cam = np.asarray(x["cam_pose"], float)[:, :2]; body = np.asarray(x[key], float)[:, :2]
            n_fr = len(x["frames"]) if "frames" in x.files else len(cam)
            if not (len(cam) == len(body) == n_fr):
                rows += 1
            n = min(len(cam), len(body))
            if np.abs(np.diff(body[:n], axis=0)).sum() < 1e-6:            # body did not move: no timing signal
                n_skip += 1; continue
            L, r0, rb = best_lag(cam[:n], body[:n]); lags.append(L); c0.append(r0)
            n_bad += int(abs(L) > 1)                                   # +-1: flat speed curve (gait), not a shift
            if "dt" in x.files:
                dts.add(float(x["dt"]))
        bad_all += n_bad + rows
        hist = {L: lags.count(L) for L in sorted(set(lags))}
        print(f"{d}: {len(lags)} files checked ({n_skip} skipped: no cam_pose / no motion) | best lag histogram {hist} | "
              f"velocity correlation at lag 0 median {np.median(c0) if c0 else np.nan:.3f} | row-count mismatches {rows} | "
              f"dt {sorted(dts)} -> {'PASS' if n_bad == 0 and rows == 0 else 'CHECK'}", flush=True)
    print("ALIGNMENT", "PASS" if bad_all == 0 else f"{bad_all} files to inspect")


if __name__ == "__main__":
    main()
