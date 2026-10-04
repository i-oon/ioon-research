"""How much of the egocentric camera's motion is NOT explained by the planar Froude labels (fwd, lat, yaw at the CoM)?

Question
    The shared Froude head is supervised on three planar numbers at the centre of mass (CoM): forward and lateral
    velocity in the body frame and the yaw rate, each smoothed over 1 s (`wm.data.embodiment`). The images come from
    a camera mounted ahead of / above the CoM that also bobs, rolls and pitches with the gait. How big is the part of
    the camera's 6-DoF motion outside those labels, per body (hexapod c10 / c08, B1) and per behaviour family, at the
    label scale (1 s) and at the 5-frame step the model predicts?

Method (non-frame keys only; frames never loaded)
    Data: data/counterfactual_walks/{c10,b1}_clips_{train,heldout}, c08_clips_heldout (66 frames, dt 0.05 s).
    * Labels: exactly `embodiment.body_velocity` + `yaw_rate` at `com_pos`, Froude height h = median CoM z.
    * Camera (`cam_pose`, world, quat xyzw; camera axes +z optical, +y image up, +x image left):
      - planar velocity of the camera point in the camera's heading frame (optical axis projected on the floor):
        cam_fwd, cam_lat (left +); vertical velocity = world z rate; yaw / pitch / roll rates = rotation vector of
        R_t^T R_{t+k} about camera +y (up, yaw-left +), +x (pitch), +z (roll), divided by k*dt.
      - angles: pitch = elevation of the optical axis, roll = elevation of the image-left axis (rad).
      All in Froude units: velocities / sqrt(g h), rates * sqrt(h / g), same h as the labels.
    * Three scales: "gait" = central difference per frame (k=1, unsmoothed, what one frame sees);
      "step5" = displacement / rotation over 5 frames (0.25 s, the model's step, unsmoothed);
      "label1s" = per-frame rates smoothed with the labels' 1 s window (`embodiment.smooth`).
    * Rigid prediction: camera planar velocity from CoM labels + a fixed lever arm r (camera minus CoM in the body
      planar frame, per-body median over all clips): v_cam = v_com + w x r, rotated by the fixed camera-vs-body
      heading offset (median). The yaw prediction is the body yaw rate. Compared with the actual camera
      fwd / lat / yaw at the same scale (the CoM rates are computed at the same k / smoothing as the camera's).
      R^2 pooled over all clips of a body (1 - SS_res / SS_tot about the pooled mean); per family the residual RMS
      relative to the signal RMS (|actual| RMS) is reported because within a family the variance is small.
    * Outside the labels: std of vertical velocity, roll / pitch rates (per-frame values pooled), std of z / roll /
      pitch about each clip's mean (angles and positions), dominant frequency of z, compared with the planar signal
      RMS (sqrt(fwd^2 + lat^2) and |yaw| of the labels).
    * CoM vs base: CoM minus base (hexapod /head, B1 base) in the body frame -- std over time within clip, and its
      rate's RMS in Froude vs the CoM speed (how much leg motion moves the CoM relative to the body).

How to read
    results/check/camera_vs_body/summary.txt: per body R^2 of the rigid prediction per channel/scale, residual/signal
    per family, out-of-label amplitudes as a fraction of the planar signal. R^2 ~1 at label1s means the labels plus a
    constant lever arm fully determine the camera's planar motion (no camera target needed). Large out-of-label
    amplitudes at step5 but ~0 at label1s mean the sway is a gait oscillation the model sees in pixels but that
    averages out -- relevant to the FTM's reconstruction, not to the Froude head. Plots: per-body bars and time
    series of one clip per family.

    .venv/bin/python3 scripts/diagnostics/egocentric_view/camera_vs_body_motion.py
"""
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from wm.data import embodiment as E  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
OUT = os.path.join(ROOT, "results/check/camera_vs_body")
BODIES = {"c10": ("hexapod", ["c10_clips_train", "c10_clips_heldout"]),
          "c08": ("hexapod", ["c08_clips_heldout"]),
          "b1": ("b1", ["b1_clips_train", "b1_clips_heldout"])}
FAMS = ["fwd", "bwd", "turn_left", "turn_right", "side_L", "side_R"]
SCALES = ["gait", "step5", "label1s"]
CH = ["fwd", "lat", "vert", "roll", "pitch", "yaw"]
WIN = int(round(E.BODY_WINDOW_S / 0.05))


def body_pose(d, emb):
    if emb == "hexapod":
        q = d["body_quat"].astype(np.float64)
        return d["head"].astype(np.float64), q
    return d["base_pos"].astype(np.float64), d["base_quat"].astype(np.float64)


def rates(pos, Rot, heading, dt, k):
    """Velocity of `pos` in a heading frame and rotation rates in the camera frame, over k frames.
    k=1: central difference (np.gradient-like for position, forward rotvec averaged); k>1: forward displacement
    (last k frames dropped, padded with nan)."""
    n = len(pos)
    f = np.stack([np.cos(heading), np.sin(heading)], 1)
    left = np.stack([-f[:, 1], f[:, 0]], 1)
    if k == 1:
        v = np.gradient(pos, dt, axis=0)
        rv = (Rot[:-1].inv() * Rot[1:]).as_rotvec() / dt          # (n-1, 3) in camera frame
        w = np.empty((n, 3)); w[0] = rv[0]; w[-1] = rv[-1]; w[1:-1] = 0.5 * (rv[:-1] + rv[1:])
        hd = np.gradient(np.unwrap(heading), dt)
    else:
        v = np.full((n, 3), np.nan); w = np.full((n, 3), np.nan); hd = np.full(n, np.nan)
        v[:-k] = (pos[k:] - pos[:-k]) / (k * dt)
        w[:-k] = (Rot[:-k].inv() * Rot[k:]).as_rotvec() / (k * dt)
        uh = np.unwrap(heading); hd[:-k] = (uh[k:] - uh[:-k]) / (k * dt)
    fwd = (v[:, :2] * f).sum(1); lat = (v[:, :2] * left).sum(1)
    return fwd, lat, v[:, 2], w, hd


def process(path, emb, lever=None, yaw_off=None):
    with np.load(path, allow_pickle=True) as d:
        com = d["com_pos"].astype(np.float64)
        cam = d["cam_pose"].astype(np.float64)
        bpos, bq = body_pose(d, emb)
        fam = str(d["family"]); dt = float(d["dt"])
    h = float(np.median(com[:, 2])); vs = np.sqrt(E.G * h); ws = np.sqrt(h / E.G)
    labels = np.concatenate([E.body_velocity(com, bq, dt, emb, height=h),
                             E.yaw_rate(bq, dt, emb, h)], 1).astype(np.float64)
    bf = E.forward_axis(bq, emb); bhead = np.arctan2(bf[:, 1], bf[:, 0])
    Rc = Rt.from_quat(cam[:, 3:7]); M = Rc.as_matrix()
    opt = M[:, :, 2]; lft = M[:, :, 0]
    chead = np.arctan2(opt[:, 1], opt[:, 0])
    pitch = np.arcsin(np.clip(opt[:, 2], -1, 1)); roll = np.arcsin(np.clip(lft[:, 2], -1, 1))
    # lever arm (camera - CoM in body planar frame) and camera heading offset
    rel = cam[:, :2] - com[:, :2]
    left = np.stack([-bf[:, 1], bf[:, 0]], 1)
    arm = np.stack([(rel * bf).sum(1), (rel * left).sum(1), cam[:, 2] - com[:, 2]], 1)
    hoff = np.angle(np.exp(1j * (chead - bhead)))
    # CoM relative to base, body frame
    relb = com[:, :2] - bpos[:, :2]
    cb = np.stack([(relb * bf).sum(1), (relb * left).sum(1), com[:, 2] - bpos[:, 2]], 1)
    out = dict(fam=fam, h=h, arm=arm, hoff=hoff, labels=labels, z=cam[:, 2], pitch=pitch, roll=roll, cb=cb)
    if lever is None:
        return out
    for sc in SCALES:
        k = 5 if sc == "step5" else 1
        cf, cl, cz, cw, _ = rates(cam[:, :3], Rc, chead, dt, k)
        mf, ml, _, _, mh = rates(com, Rc, bhead, dt, k)              # CoM planar, body heading frame
        # camera rates about camera axes: +y up -> yaw, +x left -> pitch (nose down +), +z optical -> roll
        actual = np.stack([cf, cl, cz, cw[:, 2], cw[:, 0], cw[:, 1]], 1)
        if sc == "label1s":
            actual = np.stack([E.smooth(actual[:, c], WIN) for c in range(6)], 1)
            mf, ml, mh = (E.smooth(x, WIN) for x in (mf, ml, mh))
        # rigid: v_cam(body frame) = v_com + w x r ; then rotate by camera heading offset
        pf = mf - mh * lever[1]; pl = ml + mh * lever[0]
        c, s = np.cos(yaw_off), np.sin(yaw_off)
        pred = np.stack([c * pf + s * pl, -s * pf + c * pl, mh], 1)
        sc_vec = np.array([vs, vs, vs, 1 / ws, 1 / ws, 1 / ws])
        out[sc] = actual / sc_vec
        out[sc + "_pred"] = pred / sc_vec[[0, 1, 5]]
    out["cbv"] = np.gradient(cb, dt, axis=0) / vs
    return out


def r2(a, p):
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    return 1 - ((a - p) ** 2).sum() / max(((a - a.mean()) ** 2).sum(), 1e-12)


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt((x ** 2).mean()))


def dom_freq(x, dt=0.05):
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), dt)
    return fr[1 + np.argmax(sp[1:])]


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    P = lambda s="": (print(s), lines.append(s))  # noqa: E731
    P("Camera vs body motion (cam_pose vs CoM Froude labels). Units: Froude (v/sqrt(gh), w*sqrt(h/g)); angles deg.")
    P("Scales: gait = per-frame central diff, step5 = over 5 frames (0.25 s), label1s = 1 s smoothing (= labels).")
    res = {}
    for name, (emb, dirs) in BODIES.items():
        paths = sorted(p for dd in dirs for p in glob.glob(os.path.join(CW, dd, "*.npz")))
        pre = [process(p, emb) for p in paths]
        arm = np.concatenate([x["arm"] for x in pre]); lever = np.median(arm, 0)
        yaw_off = float(np.median(np.concatenate([x["hoff"] for x in pre])))
        clips = [process(p, emb, lever[:2], yaw_off) for p in paths]
        res[name] = clips
        hs = np.array([c["h"] for c in clips])
        P(f"\n=== {name} ({emb}), {len(clips)} clips, CoM height h median {np.median(hs):.3f} m")
        P(f"lever arm camera-CoM (body frame fwd, left, up) median {np.round(lever, 4)} m, within-clip std "
          f"{np.round(np.mean([x['arm'].std(0) for x in pre], 0), 4)} m; camera heading - body heading "
          f"median {np.degrees(yaw_off):.2f} deg, std {np.degrees(np.std(np.concatenate([x['hoff'] for x in pre]))):.2f} deg")
        # (2) rigid R^2 pooled
        P("Rigid prediction of camera planar motion from CoM fwd/lat/yaw + fixed lever arm: R^2 pooled (resid RMS / signal RMS)")
        for sc in SCALES:
            A = np.concatenate([c[sc] for c in clips]); Pp = np.concatenate([c[sc + "_pred"] for c in clips])
            s = "  ".join(f"{ch} {r2(A[:, i], Pp[:, j]):.3f} ({rms(A[:, i] - Pp[:, j]) / rms(A[:, i]):.2f})"
                          for ch, i, j in (("fwd", 0, 0), ("lat", 1, 1), ("yaw", 5, 2)))
            P(f"  {sc:8s} {s}")
        # also labels themselves vs label1s-camera (the actual Froude labels, not our recomputed com rates)
        A = np.concatenate([c["label1s"] for c in clips]); L = np.concatenate([c["labels"] for c in clips])
        P(f"  camera(label1s) vs raw CoM labels (no lever arm): fwd R^2 {r2(A[:, 0], L[:, 0]):.3f}  "
          f"lat {r2(A[:, 1], L[:, 1]):.3f}  yaw {r2(A[:, 5], L[:, 2]):.3f}")
        P("Per family: resid/signal RMS at step5 | label1s (fwd, lat, yaw)")
        for fam in FAMS:
            cc = [c for c in clips if c["fam"] == fam]
            row = []
            for sc in ("step5", "label1s"):
                A = np.concatenate([c[sc] for c in cc]); Pp = np.concatenate([c[sc + "_pred"] for c in cc])
                row.append(" ".join(f"{rms(A[:, i] - Pp[:, j]) / rms(A[:, i]):.2f}" for i, j in ((0, 0), (1, 1), (5, 2))))
            P(f"  {fam:10s} {row[0]}  |  {row[1]}")
        # (3) outside the labels
        P("Outside the labels: RMS (pooled over frames) of camera rates, Froude; planar signal = RMS of labels")
        for fam in FAMS + ["ALL"]:
            cc = [c for c in clips if fam == "ALL" or c["fam"] == fam]
            L = np.concatenate([c["labels"] for c in cc])
            sp = rms(np.hypot(L[:, 0], L[:, 1])); sy = rms(L[:, 2])
            row = [f"{fam:10s} planar |v| {sp:.3f} |yaw| {sy:.3f} ||"]
            for sc in SCALES:
                A = np.concatenate([c[sc] for c in cc])
                A = A - np.nanmean(A, 0) * np.array([0, 0, 1, 1, 1, 0])   # out-of-label channels about their mean
                row.append(f"{sc}: vert {rms(A[:, 2]):.3f} roll {rms(A[:, 3]):.3f} pitch {rms(A[:, 4]):.3f}")
            P("  " + "  ".join(row))
        zs = np.array([np.std(c["z"]) for c in clips]); ps = np.array([np.degrees(np.std(c["pitch"])) for c in clips])
        rs = np.array([np.degrees(np.std(c["roll"])) for c in clips])
        fz = np.array([dom_freq(c["z"]) for c in clips]); fr_ = np.array([dom_freq(c["roll"]) for c in clips])
        fp = np.array([dom_freq(c["pitch"]) for c in clips])
        P(f"Within-clip std (median over clips): camera z {1000 * np.median(zs):.1f} mm ({np.median(zs / hs) * 100:.2f}% of h), "
          f"pitch {np.median(ps):.2f} deg, roll {np.median(rs):.2f} deg; mean pitch {np.degrees(np.mean([c['pitch'].mean() for c in clips])):.1f} deg")
        P(f"Dominant frequency (median over clips): z {np.median(fz):.2f} Hz, roll {np.median(fr_):.2f} Hz, pitch {np.median(fp):.2f} Hz")
        # angle change over 5 frames vs yaw change over 5 frames (what one model step sees), degrees
        d5 = np.concatenate([c["step5"] for c in clips]) * 0.25          # Froude rate * 0.25 s -> need physical
        P("Per model step (5 frames, 0.25 s), physical: RMS |d yaw| vs |d pitch| |d roll| (deg), |d fwd/lat| vs |d z| (mm)")
        for fam in FAMS:
            cc = [c for c in clips if c["fam"] == fam]
            A = np.concatenate([c["step5"] * np.array([np.sqrt(9.81 * c["h"])] * 3 + [1 / np.sqrt(c["h"] / 9.81)] * 3)
                                for c in cc]) * 0.25
            P(f"  {fam:10s} yaw {np.degrees(rms(A[:, 5])):.2f}  pitch {np.degrees(rms(A[:, 4])):.2f}  roll "
              f"{np.degrees(rms(A[:, 3])):.2f} deg | planar {1000 * rms(np.hypot(A[:, 0], A[:, 1])):.1f}  vert "
              f"{1000 * rms(A[:, 2]):.1f} mm")
        del d5
        # (4) CoM vs base
        cbs = np.array([c["cb"].std(0) for c in clips]); cbm = np.median(np.concatenate([c["cb"] for c in clips]), 0)
        cbv = np.concatenate([c["cbv"] for c in clips]); L = np.concatenate([c["labels"] for c in clips])
        cbv1 = np.concatenate([np.stack([E.smooth(c["cbv"][:, i], WIN) for i in range(3)], 1) for c in clips])
        P(f"CoM - base (body frame fwd, left, up) median {np.round(cbm, 4)} m; within-clip std median "
          f"{np.round(1000 * np.median(cbs, 0), 2)} mm; its rate RMS (Froude) per-frame {np.round([rms(cbv[:, i]) for i in range(3)], 4)}"
          f", 1 s smoothed {np.round([rms(cbv1[:, i]) for i in range(3)], 4)}; label planar RMS {rms(np.hypot(L[:, 0], L[:, 1])):.3f}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    plots(res)


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res)
    # bar: out-of-label RMS at three scales vs planar signal
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ai, ch in enumerate([(2, "vertical velocity"), (3, "roll rate"), (4, "pitch rate")]):
        ax = axs[ai]
        for bi, n in enumerate(names):
            vals = []
            for sc in SCALES:
                A = np.concatenate([c[sc][:, ch[0]] for c in res[n]]); vals.append(rms(A - np.nanmean(A)))
            ax.bar(np.arange(3) + bi * 0.27, vals, 0.27, label=n)
        ax.set_xticks(np.arange(3) + 0.27); ax.set_xticklabels(SCALES); ax.set_title(ch[1] + " RMS (Froude)")
    axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "out_of_label_rms.png"), dpi=110); plt.close(fig)
    # time series: one clip per family per body, z/pitch/roll and planar
    for n in names:
        fig, axs = plt.subplots(len(FAMS), 2, figsize=(11, 12), sharex=True)
        for fi, fam in enumerate(FAMS):
            c = next(c for c in res[n] if c["fam"] == fam); t = np.arange(len(c["z"])) * 0.05
            ax = axs[fi, 0]
            for i, lab in ((0, "fwd"), (1, "lat"), (5, "yaw")):
                ax.plot(t, c["gait"][:, i], lw=0.8, label=f"cam {lab} gait")
                ax.plot(t, c["label1s"][:, i], lw=1.8, label=f"cam {lab} 1s")
            ax.set_ylabel(fam)
            ax = axs[fi, 1]
            ax.plot(t, 1000 * (c["z"] - c["z"].mean()), label="cam z (mm)")
            ax.plot(t, np.degrees(c["pitch"] - c["pitch"].mean()), label="pitch (deg)")
            ax.plot(t, np.degrees(c["roll"] - c["roll"].mean()), label="roll (deg)")
        axs[0, 0].legend(fontsize=6, ncol=3); axs[0, 1].legend(fontsize=7)
        axs[0, 0].set_title(f"{n}: camera planar (Froude)"); axs[0, 1].set_title(f"{n}: out-of-label sway about clip mean")
        axs[-1, 0].set_xlabel("s"); axs[-1, 1].set_xlabel("s")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"timeseries_{n}.png"), dpi=100); plt.close(fig)
    # scatter rigid pred vs actual at step5
    fig, axs = plt.subplots(len(names), 3, figsize=(11, 3.4 * len(names)))
    for bi, n in enumerate(names):
        A = np.concatenate([c["step5"] for c in res[n]]); Pp = np.concatenate([c["step5_pred"] for c in res[n]])
        for k, (i, j, lab) in enumerate(((0, 0, "fwd"), (1, 1, "lat"), (5, 2, "yaw"))):
            ax = axs[bi, k]; ax.scatter(Pp[:, j], A[:, i], s=1, alpha=0.3)
            lim = np.nanpercentile(np.abs(A[:, i]), 99.5); ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6)
            ax.set_title(f"{n} {lab} step5 R2 {r2(A[:, i], Pp[:, j]):.2f}", fontsize=9)
            ax.set_xlabel("rigid prediction"); ax.set_ylabel("camera")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "rigid_pred_step5.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
