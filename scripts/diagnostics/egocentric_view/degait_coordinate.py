"""Does removing the gait-locked component of an egocentric view improve the cross-body coordinate?

    .venv/bin/python3 scripts/diagnostics/egocentric_view/degait_coordinate.py

**The question Q2 left open.** Egocentric moved yaw transfer from 0.07 to 0.64 and moved forward
and lateral the other way, 0.63 to 0.50 and 0.43 to 0.39. A head camera carries two things at once:
**where the body is going**, which both robots share, and **how the body shakes getting there**,
which is a six-legged tripod on one and a quadruped trot on the other. If the second can be removed,
the first should cross better -- and that would be a method rather than a setting.

The gait sits at about **6 cycles per clip on both bodies** while the net turn sits at 0 to 1, so
they are separable in frequency. Per clip this fits `sin` and `cos` at the measured stride frequency
and two harmonics, subtracts what they explain from every embedding dimension, and **keeps the
clip's mean** -- removing that too would delete the scene, not the shake.

The coordinate is then fitted on the insect and applied to the B1 **without refitting**, exactly as
in Q2, so the two numbers are comparable.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from diagnostics.objective_experiments.residual_structure import gram, ridge_r2  # noqa: E402
from wm.adapt3 import gather  # noqa: E402
from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, heading, load  # noqa: E402

CH = ("forward", "lateral", "yaw")
QUAT = {"hexapod": "body_quat", "b1": "base_quat"}


def stride_cycles(psi):
    """Cycles per clip of the gait oscillation, from the yaw residual's own spectrum."""
    n = len(psi)
    t = np.arange(n)
    osc = psi - np.polyval(np.polyfit(t, psi, 1), t)
    P = np.abs(np.fft.rfft(osc * np.hanning(n))) ** 2
    return max(1, int(np.argmax(P[1:])) + 1)


def degait(E, cycles, harmonics=3):
    """Subtract what the stride explains, keep the mean."""
    n = len(E)
    t = np.arange(n) / n
    cols = [np.ones(n)]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * cycles * t), np.cos(2 * np.pi * k * cycles * t)]
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, E, rcond=None)
    fitted = X @ beta
    return E - fitted + beta[0][None, :]        # the intercept row is the clip mean


def collect(name, data, ck, cache_path, chunk=2):
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    clips = gather(os.path.join(ROOT, data), name, enc, ck, cache, chunk, 1,
                   torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if len(cache) > before:
        torch.save(cache, cache_path)
    del enc
    torch.cuda.empty_cache()
    # **Half precision from the first clip, and one representation at a time.** Held as float32
    # this is 4.6 GB per body per version and four of them do not fit in 31 GB; the first attempt
    # died silently with no traceback, which is what an out-of-memory kill looks like from inside.
    raw, deg, B, cid = [], [], [], []
    for ci, c in enumerate(clips):
        e = c["e"].float().numpy()
        clip = load(os.path.join(ROOT, data, c["path"]), REGISTRY[name])
        bm = np.asarray(clip["body_motion"], float)
        psi = np.unwrap(heading(np.asarray(
            np.load(os.path.join(ROOT, data, c["path"]), allow_pickle=True)[QUAT[name]], float), name))
        n = min(len(e), len(bm), len(psi))
        flat = e[:n].reshape(n, -1)
        raw.append(torch.from_numpy(flat).half())
        deg.append(torch.from_numpy(degait(flat.astype(np.float64),
                                           stride_cycles(psi[:n])).astype(np.float32)).half())
        B.append(bm[:n, :3])
        cid.append(np.full(n, ci))
    return (torch.cat(raw), torch.cat(deg),
            np.concatenate(B), np.concatenate(cid), clips)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(ROOT, "wm/runs/beh12_hex-b1_body3/best.pt"),
                    map_location="cpu", weights_only=False)
    from_checkpoint(ck["config"])
    H = collect("hexapod", "data/egocentric/beh12_c10f10t10_ego_flat", ck,
                os.path.join(ROOT, "results/wm/cache/ego_hex.pt"))
    Bd = collect("b1", "data/egocentric/beh12_b1_ego_flat", ck,
                 os.path.join(ROOT, "results/wm/cache/ego_b1.pt"))

    clips_h = sorted(set(H[3].tolist()))
    te = np.array([c in set(clips_h[1::2]) for c in H[3]])
    tr = ~te
    folds = np.array([hash(int(c)) % 4 for c in H[3][tr]])
    mu, sd = H[2][tr].mean(0), H[2][tr].std(0) + 1e-9
    yh, yb = (H[2] - mu) / sd, (Bd[2] - mu) / sd

    print(f"{tr.sum()} insect train / {te.sum()} insect held-out / {len(Bd[2])} B1 frames\n")
    print(f"  {'features':>22}{'test':>20}" + "".join(f"{c:>10}" for c in CH))
    for label, Xh, Xb in (("egocentric (Q2)", H[0], Bd[0]),
                          ("gait removed", H[1], Bd[1])):
        centre = Xh[tr].float().mean(0, keepdim=True).half()
        xh, xb = Xh - centre, Xb - centre
        Kh = gram(xh, xh, dev).numpy()
        Kb = gram(xb, xh, dev).numpy()
        _, ph, _ = ridge_r2(Kh[np.ix_(tr, tr)], Kh[np.ix_(te, tr)], yh[tr], yh[te], folds)
        _, pb, _ = ridge_r2(Kh[np.ix_(tr, tr)], Kb[:, tr], yh[tr], yb, folds)
        for tag, pred, truth in (("insect held-out", ph, yh[te]), ("**B1 unrefitted**", pb, yb)):
            row = "".join(f"{np.corrcoef(pred[:, j], truth[:, j])[0, 1]:>10.2f}" for j in range(3))
            print(f"  {label:>22}{tag:>20}{row}")


if __name__ == "__main__":
    main()
