"""Can one still egocentric frame reveal the body's Froude motion without the room giving it away?

Question (2026-10-03, before round 2's camera-vs-body options). A head that reads the current frame
as well as z only keeps z grounded if the frame alone cannot tell the motion (F57: it could, R2 0.676,
on the old speed7 data with no room split). In data/counterfactual_walks every main clip has its own
room (train seeds 0-47, behaviour i -> rooms 2i, 2i+1; heldout 200-223), so a probe can learn
"this room = this speed" on train rooms; only rooms it never saw show whether the motion is visible in
the frame itself (legs, tilt, blur).

Probe: frozen V-JEPA2 on ONE frame (the encoder duplicates it into its 2-frame tubelet, so no motion
across time), patch tokens pooled two ways -- mean (1408-d) and 2x2 quadrants (5632-d) -- then ridge
-> Froude (fwd, lat, yaw; the loader's 1 s CoM label at that frame). Alpha chosen on the val rooms.

  seen rooms   fit on half the frames of every train clip, test on the other half (rooms seen):
               the room-memorising ceiling
  unseen rooms fit on all train clips, test on heldout clips (rooms 200-223, never seen): the answer

R2 near 0 on unseen rooms -> the frame alone does not show the motion; a frame-conditioned head is
safe from this shortcut. High R2 on unseen rooms -> the shortcut is real.

    .venv/bin/python3 scripts/diagnostics/egocentric_view/single_frame_speed_probe.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, p))
from wm.data.embodiment import REGISTRY, load  # noqa: E402

CW = "data/counterfactual_walks"
BODIES = (("c10", "hexapod"), ("b1", "b1"))
SPLITS = ("train", "val", "heldout")
CH = ("fwd", "lat", "yaw")


@torch.no_grad()
def features(encoder, frames, chunk=16):
    """Per frame: mean over the 256 patch tokens, and the mean of each 8x8-patch quadrant."""
    mean, quad = [], []
    for s in range(0, len(frames), chunk):
        e = encoder.encode(list(frames[s:s + chunk])).float()          # (n, 256, 1408)
        mean.append(e.mean(1).cpu())
        g = e.reshape(len(e), 16, 16, -1)
        quad.append(torch.cat([g[:, i:i + 8, j:j + 8].mean((1, 2)) for i in (0, 8) for j in (0, 8)], 1).cpu())
    return torch.cat(mean).numpy(), torch.cat(quad).numpy()


def gather(get_encoder, body, emb, split, out):
    path = os.path.join(out, f"feats_{body}_{split}.npz")
    files = sorted(glob.glob(os.path.join(ROOT, CW, f"{body}_clips_{split}", "*.npz")))
    if not files:
        raise SystemExit(f"no clips in {CW}/{body}_clips_{split}")
    stamp = np.array([f"{os.path.basename(f)}:{os.stat(f).st_size}:{os.stat(f).st_mtime_ns}" for f in files])
    if os.path.exists(path):
        d = np.load(path)
        if list(d["stamp"]) == list(stamp):
            return d["mean"], d["quad"], d["y"], d["clip"]
    encoder = get_encoder()
    M, Q, Y, C = [], [], [], []
    for ci, f in enumerate(files):
        clip = load(f, REGISTRY[emb])
        y = np.asarray(clip["body_motion"])[:, :3]
        m, q = features(encoder, np.asarray(clip["frames"]))
        n = min(len(m), len(y))
        M.append(m[:n]); Q.append(q[:n]); Y.append(y[:n]); C.append(np.full(n, ci))
    M, Q, Y, C = map(np.concatenate, (M, Q, Y, C))
    tmp = path + ".tmp.npz"
    np.savez(tmp, mean=M, quad=Q, y=Y, clip=C, stamp=stamp)
    os.replace(tmp, path)
    print(f"  {body} {split}: {len(files)} clips, {len(M)} frames")
    return M, Q, Y, C


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def ridge(Xtr, Ytr, alphas):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym = Ytr.mean(0)
    X = (Xtr - mu) / sd
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ (Ytr - ym)
    fits = {}
    for a in alphas:
        W = Vt.T @ ((S / (S ** 2 + a))[:, None] * UtY)
        fits[a] = lambda Z, W=W: ((Z - mu) / sd) @ W + ym
    return fits


def best_fit(Xtr, Ytr, Xva, Yva, alphas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5)):
    fits = ridge(Xtr, Ytr, alphas)
    a = max(alphas, key=lambda a: r2(Yva, fits[a](Xva)).mean())
    return fits[a], a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/check/single_frame_probe")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    held = {}

    def get_encoder():                 # loaded only if some split has no current feature file
        if "enc" not in held:
            from vjepa2_encoder import VJEPA2FrameEncoder
            held["enc"] = VJEPA2FrameEncoder(dtype=torch.float32)
        return held["enc"]

    data = {(body, split): gather(get_encoder, body, emb, split, out) for body, emb in BODIES for split in SPLITS}
    held.clear()
    torch.cuda.empty_cache()

    lines = ["R2 per channel fwd / lat / yaw (mean); ridge on frozen V-JEPA2 features of ONE frame -> 1 s CoM Froude"]
    for body, _ in BODIES:
        for fi, feat in enumerate(("mean-pooled 1408-d", "2x2 quadrants 5632-d")):
            Xtr, Ytr, Ctr = data[body, "train"][fi], data[body, "train"][2], data[body, "train"][3]
            Xva, Yva = data[body, "val"][fi], data[body, "val"][2]
            Xte, Yte = data[body, "heldout"][fi], data[body, "heldout"][2]
            # seen rooms: alternate frames of every train clip (same rooms on both sides)
            idx = np.arange(len(Xtr))
            fit, a = best_fit(Xtr[idx % 2 == 0], Ytr[idx % 2 == 0], Xva, Yva)
            seen = r2(Ytr[idx % 2 == 1], fit(Xtr[idx % 2 == 1]))
            fit, a2 = best_fit(Xtr, Ytr, Xva, Yva)
            unseen = r2(Yte, fit(Xte))
            fmt = lambda v: " / ".join(f"{x:+.2f}" for x in v) + f" ({v.mean():+.2f})"
            lines.append(f"{body:<4} {feat:<22} seen rooms {fmt(seen)}   unseen rooms {fmt(unseen)}   "
                         f"alpha {a:g}/{a2:g}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
