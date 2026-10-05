"""Does the model read speed from how close the wall is? (F314 follow-up, 2026-10-05, user's question)

Every clip starts at its room's centre and walks toward a wall, so in the training data wall distance goes with time in the
clip. If the model learned "wall close = fast" instead of the real visual motion, its speed read at a CONSTANT true speed
drifts with wall proximity. Measured on straight-walking heldout clips (speed families, forward and backward), steady part
(frames >= 20), real-future read head(ITM(e_t, e_t+5)), forward channel, in Froude units:
  proximity p = distance travelled from the first frame / (room size / 2)   (0 = centre, ~1 = at the wall)
  error = read - true (true = the training label, mean 1 s CoM Froude over [t, t+5))
  report: corr(error, p) pooled within clips (both demeaned per clip), and the slope d(error)/dp, original vs random-size.
A wall-distance shortcut shows as a clearly non-zero within-clip correlation that changes when the room size changes.
Room size: stored `room_size` (rs files; hexapod originals 8.0), B1 originals 17.65 m (the camera-height-scaled room).

    .venv/bin/python3 scripts/diagnostics/egocentric_view/wall_distance_bias.py
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from rollout_state_action_anova import Models  # noqa: E402
from wm.data.emb_cache import load_cache  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402

CW = "data/counterfactual_walks"
B1_ORIG_ROOM = 17.65
RUNS = [("c10", "hexapod", "results/eval/round1_branches_s0/ckpt/hex.pt", "results/wm/cache/test_v4_hex_heldout.pt"),
        ("c08", "hexapod", "results/eval/round1_branches_s0/ckpt/hex.pt", "results/wm/cache/test_v4_c08_heldout.pt"),
        ("b1", "b1", "results/eval/round1_branches_s0/ckpt/b1.pt", "results/wm/cache/test_v4_b1_heldout.pt")]


@torch.no_grad()
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lines = ["within-clip corr(read error, wall proximity) and slope (Froude per unit proximity), forward channel, "
             "straight-walking heldout clips, steady part; round1_branches_s0"]
    for body, emb, ckpt, cache in RUNS:
        m = Models(os.path.join(ROOT, ckpt), emb, 18 if emb == "hexapod" else 12, dev)
        ck = torch.load(os.path.join(ROOT, ckpt), map_location="cpu", weights_only=False)
        mean, std = [np.asarray(x).ravel()[:3] for x in ck["body_stats"]]
        del ck
        E = load_cache(os.path.join(ROOT, cache))
        K = m.stride
        for tag, d in (("original", f"{CW}/{body}_clips_heldout"), ("random size", f"{CW}/rs_{body}_clips_heldout")):
            errs, prox, n_clips, n_missing = [], [], 0, 0
            for p in sorted(glob.glob(os.path.join(ROOT, d, "*.npz"))):
                c = load(p, REGISTRY[emb], lazy_frames=True)
                fam = str(c.get("family", c.get("behaviour", "")))
                cond = str(c.get("condition", ""))
                if fam not in ("fwd", "bwd"):
                    continue
                if p not in E:
                    n_missing += 1
                    continue
                with np.load(p, allow_pickle=True) as z:
                    size = float(z["room_size"]) if "room_size" in z.files else (B1_ORIG_ROOM if body == "b1" else 8.0)
                    pos = np.asarray(z["cam_pose"])[:, :2]
                e = E[p].float().to(dev)
                if m.offset is not None:
                    e = e - m.offset.float().to(dev).reshape(e.shape[1:])
                bm = np.asarray(c["body_motion"])[:, 0]
                ts = np.arange(20, len(e) - K)
                if len(ts) < 5:
                    continue
                z_ = m.itm(e[ts], e[ts + K])
                read = m.md.body(None, z_).float().cpu().numpy()[:, 0] * std[0] + mean[0]
                true = np.array([bm[t:t + K].mean() for t in ts])
                pr = np.linalg.norm(pos[ts] - pos[0], axis=1) / (size / 2)
                er = read - true
                errs.append(er - er.mean()); prox.append(pr - pr.mean()); n_clips += 1
            if not errs:
                lines.append(f"  {body:<4} {tag:<12} no clips (missing cache entries: {n_missing})")
                continue
            er, pr = np.concatenate(errs), np.concatenate(prox)
            r = float(np.corrcoef(er, pr)[0, 1])
            slope = float((er * pr).sum() / max((pr * pr).sum(), 1e-12))
            lines.append(f"  {body:<4} {tag:<12} clips {n_clips:2d}  pairs {len(er):4d}  corr {r:+.3f}  slope {slope:+.3f}  "
                         f"proximity range {np.concatenate([x for x in prox]).std():.3f} (std, demeaned)")
        del E, m
        torch.cuda.empty_cache()
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(ROOT, "results/check/wall_distance_bias")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
