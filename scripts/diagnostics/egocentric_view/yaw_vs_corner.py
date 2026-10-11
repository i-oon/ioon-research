"""Does the yaw read drop when no wall corner / edge is in view? (2026-10-10, F337 open question)

Held-out c10 TURN clips (steady, one command per clip, no switch): per 5-frame pair, the real-frame read head(ITM(e_t, e_t+5))
of yaw vs the true 1 s label, and a corner score of the frame = strength of the strongest vertical edge in the wall band
(column-wise mean |d/dx| of the grayscale image over rows 15-55 % of the height, max over columns, divided by the median).
Pairs split into corner-visible / plain-wall halves by the score's median per clip; reported: mean read / true yaw ratio and
r(read, true) in each half. Model given by --ckpt (merged hex.pt).

    .venv/bin/python3 scripts/diagnostics/egocentric_view/yaw_vs_corner.py --ckpt results/eval/round1_branches_s0_gray_imread/ckpt/hex.pt
"""
import argparse, glob, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))


def corner_score(f):
    g = f.astype(float).mean(-1)
    h = g.shape[0]
    band = g[int(0.15 * h):int(0.55 * h)]
    col = np.abs(np.diff(band, axis=1)).mean(0)
    return float(col.max() / (np.median(col) + 1e-6))


@torch.no_grad()
def main():
    from rollout_state_action_anova import Models
    from vjepa2_encoder import VJEPA2FrameEncoder
    from wm.data.embodiment import REGISTRY, load
    from wm.evaluate import encode_clip
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); a = ap.parse_args()
    m = Models(os.path.join(ROOT, a.ckpt), "hexapod", 18, "cuda"); K = m.stride
    st = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)["body_stats"]
    mean, std = [np.asarray(x).ravel()[:3] for x in st]
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data/counterfactual_walks/c10_clips_heldout/*.npz"))):
        c = load(p, REGISTRY["hexapod"], lazy_frames=True)
        cond = str(np.load(p, allow_pickle=True)["condition"])
        if not cond.startswith("turn"):
            continue
        fr = np.asarray(c["frames"]); bm = np.asarray(c["body_motion"])[:, :3]
        e = encode_clip(enc, fr, 8).float().cuda()
        n = len(e) - K
        r = (m.md.body(e[:n], m.itm(e[:n], e[K:K + n])).float().cpu().numpy() * std + mean)[:, 2]
        cs = np.array([corner_score(fr[t]) for t in range(n)])
        hi = cs >= np.median(cs)
        for t in range(n):
            rows.append((cond, bool(hi[t]), r[t], bm[t:t + K, 2].mean(), cs[t]))
    rows = np.array([(int(h), rd, tr, s) for _, h, rd, tr, s in rows], float)
    for lab, msk in (("corner / edge visible (upper half of score)", rows[:, 0] == 1), ("plain wall (lower half)", rows[:, 0] == 0)):
        x = rows[msk]
        print(f"{lab:45s} n={len(x)}  mean |read| / |true| yaw {np.abs(x[:, 1]).mean() / np.abs(x[:, 2]).mean():.2f}  "
              f"r(read, true) {np.corrcoef(x[:, 1], x[:, 2])[0, 1]:+.2f}")


if __name__ == "__main__":
    main()
