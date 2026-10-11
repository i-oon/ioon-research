"""Feasibility check (no dataset): can 4-leg hexapod variants (c10f10t10 with one leg pair ghost-removed) walk the
beh24 commands, and can random 12-D babbling cover the c10 behaviours' CoM Froude range while staying upright?

QUESTION, per variant (front_loss = FL/FR removed, middle_loss = ML/MR, hind_loss = HL/HR):
  1. Plain c10 CPG tripod, the 24 beh24 commands (collect_switch_hex.COND, c10 centre pose), remaining 4 legs only:
     does it stay upright, and what CoM Froude (fwd, lat, yaw) does each command give?
  2. Random babbling in the 12-D action space (joint targets of the 4 remaining legs): fall rate, usable (upright)
     fraction, Froude coverage of the usable segments vs the c10 24-behaviour targets
     (data/counterfactual_walks/c10_walks/targets.npy).
  `six_leg` (no removal, 18-D) is run through the same command protocol as a check of this script's labels
  against targets.npy.

METHOD
  Body: medauroidea_c10f10t10.ttt, legs ghost-removed at runtime (collect_ik.ghost_remove_legs: shapes invisible
  and non-respondable, joint force 0 -- re-applied every step by drive_and_record(remove_legs=...)). The removed
  legs keep their mass and hang limp; the CoM (com_pos, recorded by drive_and_record from the scene's own shape
  masses, wm.data.com convention) is the physical CoM and so includes them. Action = 12 columns (the 4 remaining
  legs, collect_ik.LEGS order); drive_and_record(active_legs=...) expands them, the removed joints are never
  commanded. Physics only (capture_frames=False, no ego room), spawn (0, 0), warm-up 20 steps holding the first
  pose, scene loaded once per process and reused (scene_reuse.SceneReuse; bit-identity is NOT needed here).
  Labels: wm.data.embodiment.body_velocity / yaw_rate at com_pos (1 s edge-correct window, segment-aware), Froude
  height = median CoM z of the run's upright frames. Commands: 150-frame walks (7.5 s); a command's Froude =
  mean label over frames 20..end (or until a fall).
  Babble: 16 episodes x 300 frames (15 s) per variant = 4 min sim time, each from a fresh start. Segments of
  20-40 frames (1-2 s), 4-frame linear cross-fade of the joint targets at a switch. Per segment:
    - drive knobs sampled around the c10 families so the commanded motion spans their range: family in
      {fwd, bwd, turn, side, free} (equal odds); fwd pace 5.0-9.5 cycles/66 fr, lead 0.25; bwd pace 3.0-5.5,
      lead 0.75; turn pace 6, spin +-0..0.7; side a0 0, ft_phase 0.5, sym 1, strafe L -1.5..-0.4 / R 0.4..1.0,
      spin as c10 +-0.1; free: everything uniform (lead 0..1, spin, strafe, ft_phase, amps 0.1..0.3, pace 3..9.5);
    - per leg (the Egocentric-VSM-style CPG + noise part): amplitude gain U(0.7, 1.3) on all three joints, phase
      offset (cycles) from a per-segment pattern {inherited tripod signs + N(0, 0.05), diagonal trot + N(0, 0.05),
      uniform random}, joint offset N(0, 0.05) rad per joint;
    - OU command noise on all joints (drive_and_record cmd_noise 0.03 rad, tau 5 frames; the recorded action is the
      perturbed one).
  Oscillator = collect_ik.cpg_frame (same equation as cpg_commands) evaluated per leg on a shared clock that
  integrates the segment's pace, with the leg's phase offset added to the clock.

THRESHOLDS (fall, absorbing within an episode: frames after the first fallen frame are unusable)
  fallen  : (CoM z < 0.5 x H6  or  tilt > 45 deg) held for >= 10 consecutive frames (0.5 s); the fall frame is the
            first of them (single-frame dips while a sagging body's belly scrapes are not falls)   (tilt = beh24_conditions.tilt_deg, angle of world-up in the body
            frame from its first-frame value, yaw-free); H6 = median CoM z of the six-leg c10 walks
            (c10_walks/*.npz, measured here).
  usable segment: a babble segment whose frames are all before the episode's fall; its Froude = mean
            segment-aware label over its frames.

    .venv/bin/python3 scripts/dataset/four_leg_feasibility.py run --variant middle_loss --port 23110
    .venv/bin/python3 scripts/dataset/four_leg_feasibility.py video --variant middle_loss --port 23110
    .venv/bin/python3 scripts/dataset/four_leg_feasibility.py analyse
Output: results/check/four_leg_feasibility/<variant>/{cmd_<c>.npz, babble_ep<k>.npz, preview.mp4}, coverage_<variant>.png,
summary.json, summary.txt.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render"):
    sys.path.insert(0, os.path.join(ROOT, p))
import collect_ik as CI  # noqa: E402
from beh24_conditions import ORDER, FAMILY, EP, tilt_deg  # noqa: E402
from collect_switch_hex import COND, KEYS, BASE  # noqa: E402
from build_branches import CENTRE_NPZ, FADE  # noqa: E402

OUT = os.path.join(ROOT, "results/check/four_leg_feasibility")
WALKS = os.path.join(ROOT, "data/counterfactual_walks/c10_walks")
SCENE = "medauroidea_c10f10t10.ttt"
VARIANTS = {"six_leg": (), "front_loss": ("FL", "FR"), "middle_loss": ("ML", "MR"), "hind_loss": ("HL", "HR")}
DT = CI.STEP_DT
CMD_FRAMES, CMD_SKIP = 150, 20
N_EP, EP_FRAMES, SEG_LO, SEG_HI = 16, 300, 20, 40
NOISE, NOISE_TAU = 0.03, 5.0
FALL_H, FALL_TILT, FALL_HOLD = 0.5, 45.0, 10
DRIVE_KW = dict(travel=0.0, warmup=20, spawn=(0.0, 0.0), ego=False, capture_frames=False)
CPG_KW = dict(mirror_joints=(0, 1, 2), spin_amp=0.25, symmetric=False, legtune=None)
SAVE = ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat", "state_joint_pos",
        "state_sim_time", "com_pos")


def save_atomic(dst, **kw):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, dst)


def active(variant):
    return [l for l in CI.LEGS if l not in VARIANTS[variant]]


def cols(variant):
    return [CI.LEGS.index(l) * 3 + k for l in active(variant) for k in range(3)]


def six_leg_height():
    """H6: median CoM z over the 24 six-leg c10 walks."""
    z = [np.load(p)["com_pos"][:, 2] for p in sorted(glob.glob(os.path.join(WALKS, "*.npz")))]
    return float(np.median(np.concatenate(z)))


def labels(com, quat, segment=None, height=None):
    """(T, 3) CoM Froude (fwd, lat, yaw), loader conventions."""
    import wm.data.embodiment as E
    h = float(np.median(com[:, 2])) if height is None else height
    return np.concatenate([E.body_velocity(com, quat, DT, "hexapod", height=h, segment=segment),
                           E.yaw_rate(quat, DT, "hexapod", h, segment=segment)], axis=1), h


def fall_index(com, quat, h6):
    """First fallen frame (len if none) and the per-frame tilt (deg)."""
    tilt = tilt_deg(np.asarray(quat, np.float64))
    bad = ((com[:, 2] < FALL_H * h6) | (tilt > FALL_TILT)).astype(int)
    run = np.convolve(bad, np.ones(FALL_HOLD, int), "valid")      # run[t] = fallen frames in t .. t+HOLD-1
    hit = np.flatnonzero(run == FALL_HOLD)
    return (int(hit[0]) if len(hit) else len(com)), tilt


# ------------------------------------------------------------------------------------------------ babble
def sample_segment(rng, legs):
    fam = rng.choice(["fwd", "bwd", "turn", "side", "free"])
    k = dict(a0=0.25, a1=0.20, a2=0.20, ft_phase=0.0, strafe=0.0, sym=0.0, spin=0.0, lead=0.25)
    if fam == "fwd":
        k["pace"] = rng.uniform(5.0, 9.5) / BASE
    elif fam == "bwd":
        k.update(pace=rng.uniform(3.0, 5.5) / BASE, lead=0.75)
    elif fam == "turn":
        k.update(pace=6.0 / BASE, spin=rng.choice([-1, 1]) * rng.uniform(0.0, 0.7))
    elif fam == "side":
        left = rng.random() < 0.5
        k.update(a0=0.0, a2=0.30, ft_phase=0.5, sym=1.0, pace=6.0 / BASE,
                 strafe=-rng.uniform(0.4, 1.5) if left else rng.uniform(0.4, 1.0),
                 spin=(0.19 if left else -0.24) + rng.normal(0, 0.1))
    else:
        k.update(pace=rng.uniform(3.0, 9.5) / BASE, lead=rng.uniform(0, 1), spin=rng.uniform(-0.7, 0.7),
                 strafe=rng.uniform(-1.0, 1.0) * (rng.random() < 0.5), ft_phase=rng.uniform(0, 1),
                 a0=rng.uniform(0.0, 0.3), a1=rng.uniform(0.1, 0.3), a2=rng.uniform(0.1, 0.3), sym=float(rng.random() < 0.5))
    pat = rng.choice(["tripod", "trot", "random"])
    inherited = np.array([0.0 if l in CI.TRIPOD_A else 0.5 for l in legs])
    if pat == "tripod":
        want = inherited + rng.normal(0, 0.05, len(legs))
    elif pat == "trot":
        # diagonal pairs in phase: front-left with rear-right
        rows = sorted({l[0] for l in legs}, key="FMH".index)
        want = np.array([0.0 if (l[0] == rows[0]) == (l[1] == "L") else 0.5 for l in legs]) + rng.normal(0, 0.05, len(legs))
    else:
        want = rng.uniform(0, 1, len(legs))
    # cpg_frame already flips tripod B (sign -1 == half a cycle on the drive terms): offset relative to that
    k.update(family=fam, pattern=pat, phase=want - inherited, gain=rng.uniform(0.7, 1.3, len(legs)),
             joint_off=rng.normal(0, 0.05, 3 * len(legs)), length=int(rng.integers(SEG_LO, SEG_HI + 1)))
    return k


def segment_cmd(seg, clock, legs, bias_raw, bias_sym):
    """18-D command of one segment at clock (cycles); removed legs' columns stay at the bias."""
    bias = (1 - seg["sym"]) * bias_raw + seg["sym"] * bias_sym
    rec = dict(bias=bias, cycles=1.0, frames=1.0, lead=seg["lead"], ft_phase=seg["ft_phase"],
               amps=(seg["a0"], seg["a1"], seg["a2"]), mirror_joints=(0, 1, 2), spin_amp=0.25, strafe=seg["strafe"])
    out = bias.astype(np.float64).copy()
    for j, l in enumerate(legs):
        i = CI.LEGS.index(l)
        gain = np.ones(6, np.float32)
        gain[i] = seg["gain"][j]
        full = CI.cpg_frame(dict(rec, amps=tuple(a * (seg["gain"][j] if k == 0 else 1.0) for k, a in enumerate(rec["amps"]))),
                            gain, np.zeros(6, np.float32), clock + seg["phase"][j], seg["spin"])
        out[i * 3:i * 3 + 3] = full[i * 3:i * 3 + 3] + seg["joint_off"][3 * j:3 * j + 3]
    return out


def babble_plan(rng, legs, frames, bias_raw, bias_sym):
    segs, seg_id, t = [], np.zeros(frames, np.int64), 0
    while t < frames:
        s = sample_segment(rng, legs)
        s["start"] = t
        seg_id[t:t + s["length"]] = len(segs)
        segs.append(s)
        t += s["length"]
    clock = np.cumsum([segs[seg_id[f]]["pace"] for f in range(frames)]) - segs[0]["pace"]
    clock = clock * BASE / EP     # pace 1 = 8.8 cycles per 66 frames
    cmds = np.zeros((frames, 18))
    for f in range(frames):
        k = seg_id[f]
        c = segment_cmd(segs[k], clock[f], legs, bias_raw, bias_sym)
        d = f - segs[k]["start"]
        if k > 0 and d < FADE:               # causal cross-fade from the previous segment's command
            w = (d + 1) / (FADE + 1)
            c = (1 - w) * segment_cmd(segs[k - 1], clock[f], legs, bias_raw, bias_sym) + w * c
        cmds[f] = c
    return cmds, seg_id, segs


# ------------------------------------------------------------------------------------------------ physics
def run(a):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from scene_reuse import SceneReuse
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    v, legs, cl = a.variant, active(a.variant), cols(a.variant)
    d = os.path.join(OUT, v)
    os.makedirs(d, exist_ok=True)
    centre = np.load(CENTRE_NPZ)["actions"].astype(np.float64).mean(0)
    kw = dict(DRIVE_KW, remove_legs=list(VARIANTS[v]), active_legs=legs)
    R = SceneReuse(sim, SCENE, kw)
    N = CMD_FRAMES
    plans = [{k: np.full(N, float(COND[c][1][k])) for k in KEYS} for c in ORDER]
    cmds = R.commands(plans, N, centre, dict(CPG_KW, cycles=BASE * N / EP))
    from scene_reuse import _LoadOnce
    bias_sym = CI.cpg_commands(_LoadOnce(sim), SCENE, 2, centre, symmetric=True)[1]["bias"].astype(np.float64)
    t0 = time.time()
    for c, cm in zip(ORDER, cmds):
        dst = os.path.join(d, f"cmd_{c}.npz")
        if os.path.exists(dst) and not a.redo:
            continue
        out = R.run(cm[:, cl])
        save_atomic(dst, **{k: np.asarray(out[k]) for k in SAVE}, state_joint_names=out["state_joint_names"],
                    variant=v, legs=np.array(legs), condition=c, dt=DT)
        print(f"{v} {c}: {time.time() - t0:.0f} s", flush=True)
    if v == "six_leg":
        return
    rng = np.random.default_rng(a.seed)
    for e in range(N_EP):
        cm, seg_id, segs = babble_plan(rng, legs, EP_FRAMES, centre, bias_sym)
        dst = os.path.join(d, f"babble_ep{e:02d}.npz")
        if os.path.exists(dst) and not a.redo:
            continue
        R.kw.update(cmd_noise=NOISE, noise_tau=NOISE_TAU, noise_seed=a.seed * 1000 + e)
        out = R.run(cm[:, cl].astype(np.float32))
        meta = [{k: (np.asarray(x).tolist() if isinstance(x, np.ndarray) else x) for k, x in s.items()} for s in segs]
        save_atomic(dst, **{k: np.asarray(out[k]) for k in SAVE}, state_joint_names=out["state_joint_names"],
                    clean_actions=cm[:, cl].astype(np.float32), segment=seg_id, segments=np.array(json.dumps(meta, default=float)),
                    variant=v, legs=np.array(legs), dt=DT, noise=NOISE, noise_tau=NOISE_TAU)
        print(f"{v} babble ep{e}: {time.time() - t0:.0f} s", flush=True)


def video(a):
    """Few-second third-person preview: one babble episode (seed 99), 120 frames, render_leg_loss_walk's camera."""
    import imageio.v2 as imageio
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from render_leg_loss_walk import add_camera, label, SENSOR_NAME
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    v, legs, cl = a.variant, active(a.variant), cols(a.variant)
    centre = np.load(CENTRE_NPZ)["actions"].astype(np.float64).mean(0)
    from scene_reuse import _LoadOnce
    bias_sym = CI.cpg_commands(_LoadOnce(sim), SCENE, 2, centre, symmetric=True)[1]["bias"].astype(np.float64)
    walk = CI.cpg_commands(_LoadOnce(sim), SCENE, 60, centre, cycles=BASE * 60 / EP, amps=(0.25, 0.2, 0.2),
                           lead=0.25, pace=np.full(60, 7.1 / BASE), **CPG_KW)[0].astype(np.float64)
    bab, _, _ = babble_plan(np.random.default_rng(99), legs, 100, centre, bias_sym)
    cm = np.concatenate([walk, bab])[:, cl].astype(np.float32)
    CI.settle(sim)
    sim.loadScene(f"{CI.ENV}/{SCENE}")
    add_camera(sim)
    CI.SENSOR = SENSOR_NAME
    f, *_ = CI.drive_and_record(sim, SCENE, cm, cam_dx=0.0, remove_legs=list(VARIANTS[v]), active_legs=legs,
                                reuse={"built": True}, **dict(DRIVE_KW, capture_frames=True))
    frames = [label(x, f"{v}  {'c10 speed_c7.1' if t < 60 else 'babble'}  t={t:03d}") for t, x in enumerate(f)]
    dst = os.path.join(OUT, v, "preview.mp4")
    imageio.mimsave(dst, frames, fps=20, codec="libx264", macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(dst)


# ------------------------------------------------------------------------------------------------ analysis
def analyse(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tg = np.load(os.path.join(WALKS, "targets.npy"))
    h6 = six_leg_height()
    S = dict(thresholds=dict(H6=h6, fall_com_z=FALL_H * h6, fall_tilt_deg=FALL_TILT, hold_frames=FALL_HOLD), variants={})
    lines = [f"H6 (median CoM z, six-leg c10 walks) = {h6:.4f} m; fallen: CoM z < {FALL_H * h6:.4f} m or tilt > {FALL_TILT} deg, held {FALL_HOLD} frames",
             "Froude columns: fwd lat yaw"]
    ch = ("fwd", "lat", "yaw")
    for v in VARIANTS:
        d = os.path.join(OUT, v)
        if not os.path.isdir(d):
            continue
        V = {}
        lines.append(f"\n=== {v} ===")
        lines.append(f"{'command':18s} upright  fall@  CoMz   max_tilt |   Froude fwd   lat    yaw |  c10 target fwd lat yaw")
        cmd_fr, n_up = [], 0
        for i, c in enumerate(ORDER):
            p = os.path.join(d, f"cmd_{c}.npz")
            if not os.path.exists(p):
                cmd_fr.append([np.nan] * 3)
                continue
            r = np.load(p)
            com, q = r["com_pos"].astype(np.float64), r["body_quat"].astype(np.float64)
            fi, tilt = fall_index(com, q, h6)
            up = fi == len(com)
            n_up += up
            end = max(fi, CMD_SKIP + 21)
            lab, h = labels(com[:end], q[:end])
            fr = lab[CMD_SKIP:end].mean(0)
            cmd_fr.append(fr)
            lines.append(f"{c:18s} {'yes' if up else 'NO ':7s} {fi:5d}  {np.median(com[:, 2]):.3f}  {tilt.max():6.1f}   | "
                         + " ".join(f"{x:+.3f}" for x in fr) + " | " + " ".join(f"{x:+.3f}" for x in tg[i]))
        cmd_fr = np.array(cmd_fr)
        V["commands"] = dict(upright=int(n_up), froude=cmd_fr.tolist(), abs_err_vs_target=np.abs(cmd_fr - tg).tolist())
        lines.append(f"upright {n_up}/24; mean |Froude - c10 target| fwd/lat/yaw: "
                     + " ".join(f"{x:.3f}" for x in np.nanmean(np.abs(cmd_fr - tg), 0)))
        eps = sorted(glob.glob(os.path.join(d, "babble_ep*.npz")))
        if eps:
            seg_fr, seg_fam, n_fall, up_frames, tot, n_seg = [], [], 0, 0, 0, 0
            for p in eps:
                r = np.load(p)
                com, q, seg = r["com_pos"].astype(np.float64), r["body_quat"].astype(np.float64), r["segment"]
                segs = json.loads(str(r["segments"]))
                fi, _ = fall_index(com, q, h6)
                n_fall += fi < len(com)
                up_frames += fi
                tot += len(com)
                n_seg += len(segs)
                if fi < 2 * CMD_SKIP:
                    continue
                lab, _ = labels(com[:fi], q[:fi], segment=seg[:fi])
                for k, s in enumerate(segs):
                    a_, b_ = s["start"], s["start"] + s["length"]
                    if b_ <= fi:
                        seg_fr.append(lab[a_:b_].mean(0))
                        seg_fam.append(s["family"])
            seg_fr = np.array(seg_fr).reshape(-1, 3)
            minutes = tot * DT / 60
            # coverage: per behaviour, distance (Froude, 3-D and per channel) from its target to the nearest segment
            dist = np.linalg.norm(tg[:, None, :] - seg_fr[None], axis=2) if len(seg_fr) else np.full((24, 1), np.nan)
            near = dist.min(1)
            per_ch = np.abs(tg[:, None, :] - seg_fr[None]).min(1) if len(seg_fr) else np.full((24, 3), np.nan)
            within = [int((dist < r_).any(1).sum()) for r_ in (0.02, 0.03, 0.05)]
            V["babble"] = dict(episodes=len(eps), sim_minutes=minutes, falls=int(n_fall), fall_rate_per_ep=n_fall / len(eps),
                               falls_per_upright_minute=n_fall / max(up_frames * DT / 60, 1e-9),
                               usable_fraction=up_frames / tot, segments=n_seg, usable_segments=len(seg_fr),
                               nearest_dist=near.tolist(), nearest_per_channel=per_ch.tolist(),
                               behaviours_within_0p02_0p03_0p05=within,
                               seg_range=dict(lo=seg_fr.min(0).tolist(), hi=seg_fr.max(0).tolist()) if len(seg_fr) else None,
                               family_counts={f: seg_fam.count(f) for f in sorted(set(seg_fam))})
            tr = tg.min(0), tg.max(0)
            lines.append(f"babble: {len(eps)} episodes x {EP_FRAMES} fr ({minutes:.1f} min), falls {n_fall}/{len(eps)}, "
                         f"{n_fall / max(up_frames * DT / 60, 1e-9):.2f} falls per upright minute, usable fraction {up_frames / tot:.2f}, "
                         f"usable segments {len(seg_fr)}/{n_seg}")
            if len(seg_fr):
                lines.append("segment Froude range  " + "  ".join(f"{n} {lo:+.3f}..{hi:+.3f}" for n, lo, hi in
                                                                  zip(ch, seg_fr.min(0), seg_fr.max(0))))
                lines.append("c10 target range      " + "  ".join(f"{n} {lo:+.3f}..{hi:+.3f}" for n, lo, hi in zip(ch, *tr)))
                lines.append(f"behaviours with a segment within 0.02 / 0.03 / 0.05 (3-D Froude): {within[0]} / {within[1]} / {within[2]} of 24;"
                             f" median nearest {np.median(near):.3f}, max {near.max():.3f} ({ORDER[int(near.argmax())]})")
                lines.append("per behaviour nearest (3-D | fwd lat yaw): " + "; ".join(
                    f"{c} {near[i]:.3f}" for i, c in enumerate(ORDER)))
            fig, ax = plt.subplots(1, 4, figsize=(17, 4))
            fams = {"fwd": "C0", "bwd": "C1", "turn_left": "C2", "turn_right": "C3", "side_L": "C4", "side_R": "C5"}
            for (x, y), axx in zip(((0, 1), (0, 2), (1, 2)), ax[:3]):
                if len(seg_fr):
                    axx.scatter(seg_fr[:, x], seg_fr[:, y], s=6, c="0.6", label="babble segments (usable)")
                axx.scatter(cmd_fr[:, x], cmd_fr[:, y], s=25, marker="x", c="k", label=f"{v}: 24 c10 commands")
                for f_, col in fams.items():
                    m = np.array(FAMILY) == f_
                    axx.scatter(tg[m, x], tg[m, y], s=40, facecolors="none", edgecolors=col, label=f"c10 target {f_}")
                axx.set_xlabel(f"Froude {ch[x]}"); axx.set_ylabel(f"Froude {ch[y]}")
            ax[0].legend(fontsize=6)
            if len(seg_fr):
                for j in range(3):
                    ax[3].hist(seg_fr[:, j], bins=40, histtype="step", label=f"babble {ch[j]}")
                for j in range(3):
                    for x in tg[:, j]:
                        ax[3].axvline(x, color=f"C{j}", lw=0.3, alpha=0.6)
                ax[3].legend(fontsize=7); ax[3].set_title("segment Froude (lines: c10 targets)")
            fig.suptitle(f"{v}: usable babble segments vs c10 24-behaviour targets; falls {n_fall}/{len(eps)} episodes, "
                         f"usable {up_frames / tot:.2f}")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"coverage_{v}.png"), dpi=110)
            plt.close(fig)
        S["variants"][v] = V
    tmp = os.path.join(OUT, "summary.json.tmp")
    json.dump(S, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(OUT, "summary.json"))
    tmp = os.path.join(OUT, "summary.txt.tmp")
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(OUT, "summary.txt"))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "video", "analyse"))
    ap.add_argument("--variant", choices=tuple(VARIANTS))
    ap.add_argument("--port", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    if a.mode != "analyse" and (a.port in (None, 23000) or a.variant is None):
        raise SystemExit("--variant and a --port other than 23000 are required")
    {"run": run, "video": video, "analyse": analyse}[a.mode](a)


if __name__ == "__main__":
    main()
