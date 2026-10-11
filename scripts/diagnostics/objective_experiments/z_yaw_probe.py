"""Does z carry the motion at full size, i.e. is the Froude head's shrink (F330 / F337: yaw read ~0.5x) in z or in the head?
(2026-10-10)

z = ITM(e_t, e_t+5) of REAL frames (model --ckpt). Linear ridge z -> standardised Froude (the head's training target: mean of the
labels t .. t+4) fitted on TRAINING data (c10 clips train + N random c10 branches train, pairs from first_pair), tested on
held-out c10 clips (steady) and held-out c10 branches (P11 window after the switch). Reported per channel: slope of the read on
the truth (1 = right size) and Pearson r, for the model's Froude head and for the probe. Ridge alpha chosen by CV on the fit set.

    .venv/bin/python3 scripts/diagnostics/objective_experiments/z_yaw_probe.py --ckpt results/eval/round1_branches_s0_gray_imread/ckpt/hex.pt
"""
import argparse, glob, os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))


@torch.no_grad()
def pairs(m, enc, path, mean, std):
    from wm.data.embodiment import REGISTRY, load
    from wm.data.strided import body_targets
    from wm.evaluate import encode_clip
    c = load(path, REGISTRY["hexapod"], lazy_frames=True)
    b = int(c.get("first_pair", 0) or 0); K = m.stride
    fr = np.asarray(c["frames"])[b:]
    e = encode_clip(enc, fr, 8).float().cuda()
    n = len(e) - K
    if n <= 0:
        return None
    z = m.itm(e[:n], e[K:K + n])
    head = m.md.body(e[:n], z).float().cpu().numpy()
    y = body_targets(np.asarray(c["body_motion"])[:, :3], K, n, start=b)
    return z.float().flatten(1).cpu().numpy(), head, (y - mean) / std


def slope_r(pred, true):
    out = []
    for j in range(3):
        t, p = true[:, j] - true[:, j].mean(), pred[:, j] - pred[:, j].mean()
        out.append(((t * p).sum() / (t * t).sum(), np.corrcoef(true[:, j], pred[:, j])[0, 1]))
    return out


def main():
    from sklearn.linear_model import RidgeCV
    from rollout_state_action_anova import Models
    from vjepa2_encoder import VJEPA2FrameEncoder
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n_branch", type=int, default=200)
    a = ap.parse_args()
    m = Models(os.path.join(ROOT, a.ckpt), "hexapod", 18, "cuda")
    st = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu", weights_only=False)["body_stats"]
    mean, std = [np.asarray(x).ravel()[:3] for x in st]
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    CW = os.path.join(ROOT, "data/counterfactual_walks")
    rng = np.random.default_rng(0)
    fit_files = sorted(glob.glob(f"{CW}/c10_clips_train/*.npz")) + list(rng.choice(sorted(glob.glob(f"{CW}/c10_branches_train/*.npz")), a.n_branch, replace=False))
    Z, Y = [], []
    for p in fit_files:
        r = pairs(m, enc, p, mean, std)
        if r is not None:
            Z.append(r[0]); Y.append(r[2])
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    probe = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(Z, Y)
    print(f"probe fitted on {len(Z)} training pairs (alpha {probe.alpha_:.3g})")
    tests = {"held-out clips (steady)": sorted(glob.glob(f"{CW}/c10_clips_heldout/*.npz")),
             "held-out branches (after a switch)": list(np.random.default_rng(1).choice(sorted(glob.glob(f"{CW}/c10_branches_heldout/*.npz")), 300, replace=False))}
    for name, files in tests.items():
        H, P, T = [], [], []
        for p in files:
            r = pairs(m, enc, p, mean, std)
            if r is None:
                continue
            z, h, y = r
            if "branches" in name:          # the P11 window after the switch, as the read-out
                z, h, y = z[:11], h[:11], y[:11]
            H.append(h); P.append(probe.predict(z)); T.append(y)
        H, P, T = (np.concatenate(x) for x in (H, P, T))
        print(f"{name}: n {len(T)} pairs; slope (r) fwd / lat / yaw")
        for lab, x in (("Froude head", H), ("linear probe on z", P)):
            sr = slope_r(x, T)
            print(f"   {lab:18s} " + " / ".join(f"{s:.2f} ({r:+.2f})" for s, r in sr))


if __name__ == "__main__":
    main()
