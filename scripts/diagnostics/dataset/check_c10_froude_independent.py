"""Independent Froude check of the hexapod (c10) main clips, data/counterfactual_walks/c10_clips_* (DATA_PLAN stage 2, step 1).

The loader (`wm.data.embodiment._hexapod`) labels from the `head` position + `body_quat` (/abdomen
quaternion, aft-pointing z axis). Here the same quantities are rebuilt from other recorded data, with no
quaternion at all:
  heading  = direction of the horizontal vector abdomen -> head (recorded link poses `state_link_pose`),
  position = (a) /abdomen origin (`state_abdomen_pos`), (b) /T1 (thorax segment next to the head),
             (c) /abdomen corrected to the head point by the rigid lever arm v + omega x r.
Velocities are resolved on that geometric heading; smoothing, Froude height (median head z, the loader's)
and dt follow wm.data.embodiment (E.smooth, 1 s window, 0.05 s frames). Reported per condition: clip-mean
(fwd, lat, yaw) loader vs independent, and per-frame max |diff|.

    .venv/bin/python3 scripts/diagnostics/dataset/check_c10_froude_independent.py
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
import wm.data.embodiment as E  # noqa: E402


def independent(d, ref):
    dt = float(d["dt"])
    names = [str(n) for n in d["state_link_names"]]
    L = d["state_link_pose"].astype(np.float64)
    head = L[:, names.index("/head"), :3]
    abd = d["state_abdomen_pos"].astype(np.float64)
    h = float(np.median(d["head"][:, 2]))
    ax = head[:, :2] - abd[:, :2]
    psi = np.unwrap(np.arctan2(ax[:, 1], ax[:, 0]))
    f = np.stack([np.cos(psi), np.sin(psi)], 1)
    left = np.stack([-f[:, 1], f[:, 0]], 1)
    om = np.gradient(psi, dt)
    if ref == "abdomen":
        p = abd[:, :2]
        v = np.gradient(p, dt, axis=0)
    elif ref == "T1":
        p = L[:, names.index("/T1"), :2]
        v = np.gradient(p, dt, axis=0)
    else:                                          # abdomen + omega x r (r = abdomen -> head, horizontal)
        v = np.gradient(abd[:, :2], dt, axis=0) + om[:, None] * np.stack([-ax[:, 1], ax[:, 0]], 1)
    s = np.sqrt(E.G * h)
    W = int(round(E.BODY_WINDOW_S / dt))
    fw = E.smooth((v * f).sum(1) / s, W)
    lt = E.smooth((v * left).sum(1) / s, W)
    yw = E.smooth(om, W) * np.sqrt(h / E.G)
    return np.stack([fw, lt, yw], 1)


def main():
    files = sorted(p for s in ("train", "val", "heldout")
                   for p in glob.glob(os.path.join(ROOT, f"data/counterfactual_walks/c10_clips_{s}/*.npz")))
    rows = {}
    for p in files:
        lab = E.load(p, E.HEXAPOD)["body_motion"].astype(np.float64)
        with np.load(p, allow_pickle=True) as d:
            c, i = str(d["condition"]), int(d["cond_index"])
            ind = {r: independent(d, r) for r in ("abdomen", "T1", "lever")}
        rows.setdefault((i, c), []).append((lab, ind))
    print(f"{len(files)} clips. clip-mean (fwd, lat, yaw): loader | abdomen raw | T1 | abdomen+lever ; "
          "per-frame max|diff| loader vs abdomen+lever")
    worst = {r: np.zeros(3) for r in ("abdomen", "T1", "lever")}
    worst_pf = np.zeros(3)
    for (i, c), rs in sorted(rows.items()):
        lab = np.mean([r[0].mean(0) for r in rs], 0)
        ind = {k: np.mean([r[1][k].mean(0) for r in rs], 0) for k in worst}
        pf = np.max([np.abs(r[0] - r[1]["lever"]).max(0) for r in rs], 0)
        for k in worst:
            worst[k] = np.maximum(worst[k], np.max([np.abs(r[0].mean(0) - r[1][k].mean(0)) for r in rs], 0))
        worst_pf = np.maximum(worst_pf, pf)
        f = lambda a: "(" + ", ".join(f"{x:+.3f}" for x in a) + ")"  # noqa: E731
        print(f"{i:2d} {c:<16} {f(lab)} | {f(ind['abdomen'])} | {f(ind['T1'])} | {f(ind['lever'])} ; {f(pf)}")
    for k, w in worst.items():
        print(f"max over clips |clip-mean diff| loader vs {k:<8}: fwd {w[0]:.4f} lat {w[1]:.4f} yaw {w[2]:.4f}")
    print(f"max per-frame |diff| loader vs lever: fwd {worst_pf[0]:.4f} lat {worst_pf[1]:.4f} yaw {worst_pf[2]:.4f}")


if __name__ == "__main__":
    main()
