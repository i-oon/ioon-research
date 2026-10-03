"""DATA_PLAN v2 stage 2: B1 main clips matched to the hexapod v4 clips -> data/counterfactual_walks/b1_clips_{train,val,heldout}
(`cut` renders with the corrected camera mount straight into b1_clips_*; the clips there were made 2026-10-02 by a
legacy-mount cut + corrected-mount re-render, field `remount_from`, same state and same frames rule).

Reference point (2026-10-01, FINDINGS F301): every label here is CoM-based -- hexapod clips and B1 rollouts
carry `com_pos` and the loader measures velocity and Froude height at it. The head-referenced version of
this stage is kept with the superseded data (doc/DATA.md, b1_head_reference_targets).

Targets: for each hexapod condition c (beh24_conditions.ORDER, index i), the mean clip-mean Froude
(fwd, lat, yaw) of its 4 v4 windows (loader labels, wm.data.embodiment). The B1 condition paired with c is
the same family and level index (DATA_PLAN section 0 item 3).

Tuning (`tune`): per condition, the B1 command (vx, vy, wz) of the canonical rollout
(sim/collect/rollout_b1_mujoco.py; wz drives the heading controller's target, heading = integral of wz) is
searched so that the steady-state Froude matches the target on all three channels: Newton steps with a
finite-difference Jacobian, then Broyden updates. Measurement: rollout of 45 unlogged + 900 recorded
policy steps (50 Hz); 4 mid-rollout windows of 165 steps (start 150 + 165 k), each subsampled to 66
frames at 0.05 s exactly as render_b1_replay --fps 20 does, put in its own face-forward frame (F293-correct
`recollect_b1_more._face_forward`) and labelled by the loader (`wm.data.embodiment._b1`); score = mean of
the 4 clip means. Acceptance per channel max(10 % of |target|, 0.005); the search
aims at max(3 %, 0.002).

Policy / heading gains: one policy for every condition (gait3, `base_gait3/model_600.pt`, clock 2.0 Hz, the
rollout default; the existing beh24 collectors used both gait3 and sym, two clips each, which a single long
rollout per condition cannot do); heading gains as the existing collectors per family (forward speed: PI
kp 2.5 / ki 1.0, F69; every other family: kp 0.5 / ki 0).

Walks (`walks`): one long rollout per condition with the tuned command; 4 non-overlapping 165-step windows
(66 frames) cut after STEADY steps, gap chosen from 20..75 steps to spread the start gait phase (phase from
the FR/RL foot-contact touchdowns). Split / room rule as stage 1 (beh24_conditions: window w of
condition i -> ROLES[(w + i) % 4], seeds train 2i+k, val 100+i, heldout 200+i).

Cut (`cut`): each window face-forwarded, re-centred at the room centre (render_b1_replay --spawn 0 0), rendered
with render_b1_replay --ego --match_floor --ground_uv_mult 1.0 --fps 20 --ego_seed <seed> (default = corrected mount,
camera on the base's forward axis) -> b1_clips_<split>/b1_ep<ep>.npz; existing files are skipped. `cut --dry_run`
lists every walk read and every target clip with its (split, seed, window start) checked against the existing clip.

    .venv/bin/python3 scripts/dataset/collect_b1_walks.py targets|tune|walks|cut|check|video
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
import wm.data.embodiment as E  # noqa: E402
from beh24_conditions import ORDER as HEX_ORDER, FAMILY, ROLES, seed_of, split_of  # noqa: E402
from recollect_b1_more import _face_forward  # noqa: E402
from wm.data.com import b1_com  # noqa: E402

OUT = "data/counterfactual_walks"
WALKS = os.path.join(OUT, "b1_walks")
CUT = "b1_clips_{}"            # `cut` output = the current B1 main clips (corrected camera mount)
TUNE_JSON = os.path.join(OUT, "b1_walks", "tuning.json")
PY = os.path.join(ROOT, ".venv/bin/python3")
ROLL = os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py")
PER = 165
KEEP = np.unique(np.round(np.arange(0, PER, 2.5)).astype(int))
assert len(KEEP) == 66
TUNE_STEPS, TUNE_STARTS = 900, (150, 315, 480, 645)
STEADY = 150
GAPS = range(20, 76)
CKPT = "sim/assets/b1_policy/base_gait3/model_600.pt"
GAIT = 2.0
GAINS = {"fwd": (2.5, 1.0)}           # others (0.5, 0.0)
B1_NAME = {"fwd": ["speed_vx0.30", "speed_vx0.38", "speed_vx0.40", "speed_vx0.50"],
           "bwd": ["speed_vx-0.30", "speed_vx-0.38", "speed_vx-0.40", "speed_vx-0.50"],
           "turn_left": ["turn_w0.008", "turn_w0.024", "turn_w0.037", "turn_w0.075"],
           "turn_right": ["turn_w0.008_neg", "turn_w0.024_neg", "turn_w0.037_neg", "turn_w0.075_neg"],
           "side_L": [f"side_L_lvl{k}" for k in range(4)], "side_R": [f"side_R_lvl{k}" for k in range(4)]}
B1_BEH = {"fwd": "speed", "bwd": "speed", "turn_left": "turn", "turn_right": "turn", "side_L": "side",
          "side_R": "side"}


def gains(i):
    return GAINS.get(FAMILY[i], (0.5, 0.0))


def targets():
    T = {}
    for s in ("train", "val", "heldout"):
        for p in glob.glob(os.path.join(ROOT, OUT, f"c10_clips_{s}", "*.npz")):
            with np.load(p, allow_pickle=True) as d:
                i = int(d["cond_index"])
            T.setdefault(i, []).append(E.load(p, E.HEXAPOD)["body_motion"].mean(0))
    assert all(len(T[i]) == 4 for i in range(24))
    return np.array([np.mean(T[i], 0) for i in range(24)], np.float64)


def rollout(cmd3, i, steps, out):
    kp, ki = gains(i)
    c = [PY, ROLL, "--vx", f"{cmd3[0]:.6f}", "--vy", f"{cmd3[1]:.6f}", "--wz", f"{cmd3[2]:.6f}",
         "--steps", str(steps), "--checkpoint", os.path.join(ROOT, CKPT), "--gait_freq", str(GAIT),
         "--head_kp", str(kp), "--head_ki", str(ki), "--out", out]
    r = subprocess.run(c, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    if r.returncode:
        raise RuntimeError(r.stderr[-1000:])
    with np.load(out, allow_pickle=True) as T:
        return {k: T[k] for k in T.files}


def window_clip(T, st):
    """Window [st, st + PER) -> the clip fields as the cut/render path stores them (face-forwarded, 66 frames)."""
    idx = st + KEEP
    out = {k: T[k][idx] for k in ("joint_pos", "joint_vel", "action", "command", "foot_contact", "base_pos",
                                  "base_quat")}
    out["base_pos"], out["base_quat"] = _face_forward(out["base_pos"].astype(np.float64),
                                                      out["base_quat"].astype(np.float64))
    # CoM (F301) of the face-forwarded pose: the subtree CoM is rigid-motion equivariant, so recomputing it
    # from the stored pose equals rotating the rollout's com_pos
    out["com_pos"] = b1_com(out["base_pos"], out["base_quat"], out["joint_pos"])
    return out


def label(clip):
    class D(dict):
        files = property(lambda self: list(self.keys()))
    d = D(base_pos=clip["base_pos"], base_quat=clip["base_quat"], action=clip["action"],
          foot_contact=clip["foot_contact"], dt=np.float64(0.05))
    if "com_pos" in clip:
        d["com_pos"] = clip["com_pos"]        # labels at the CoM (F301), as the loader reads the stored clips
    return E._b1(d)["body_motion"].astype(np.float64)


def health(T):
    z = T["base_pos"][:, 2]
    q = T["base_quat"].astype(np.float64)
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return dict(z_min=float(z.min()), up_min=float(up.min()), fell=bool(z.min() < 0.35 or up.min() < 0.7))


def measure(cmd3, i, tmp):
    T = rollout(cmd3, i, TUNE_STEPS, os.path.join(tmp, f"m{i}.npz"))
    ms = np.array([label(window_clip(T, s)).mean(0) for s in TUNE_STARTS])
    return ms.mean(0), ms.std(0), health(T)


def ok(ach, tgt, rel=0.1, floor=0.005):
    return np.abs(ach - tgt) <= np.maximum(rel * np.abs(tgt), floor)


def tight(y, tgt):
    # the search aims tighter than the acceptance tolerance (3 % / 0.002) so the walks land well inside it
    return np.abs((y - tgt) / np.maximum(0.03 * np.abs(tgt), 0.002)).max()


def tune_one(i):
    tgt = TGT[i]
    tmp = tempfile.mkdtemp(prefix=f"b1tune{i}_")
    sc = np.array([2.34, 2.34, 4.19])              # Froude -> command (h ~ 0.56 m)
    x = tgt * sc
    hist = []
    y, sd, hl = measure(x, i, tmp); hist.append((x.tolist(), y.tolist(), hl))
    # finite-difference Jacobian
    J = np.zeros((3, 3))
    hstep = np.array([0.05, 0.05, 0.08])
    for k in range(3):
        xk = x.copy(); xk[k] += hstep[k]
        yk, _, hk = measure(xk, i, tmp); hist.append((xk.tolist(), yk.tolist(), hk))
        J[:, k] = (yk - y) / hstep[k]
    best = (tight(y, tgt), x, y, sd, hl)
    for it in range(14):
        if tight(y, tgt) <= 1 and not hl["fell"]:
            break
        dx = np.linalg.lstsq(J, tgt - y, rcond=None)[0]
        dx = np.clip(dx, -0.25, 0.25)
        xn = x + dx
        yn, sdn, hn = measure(xn, i, tmp); hist.append((xn.tolist(), yn.tolist(), hn))
        dy = yn - y
        if np.linalg.norm(dx) > 1e-9:                # Broyden rank-1 update
            J = J + np.outer(dy - J @ dx, dx) / (dx @ dx)
        x, y, sd, hl = xn, yn, sdn, hn
        sc_ = tight(y, tgt)
        if sc_ < best[0] and not hl["fell"]:
            best = (sc_, x, y, sd, hl)
    sc_, x, y, sd, hl = best
    return dict(i=i, hex=HEX_ORDER[i], target=tgt.tolist(), cmd=x.tolist(), achieved=y.tolist(),
                window_sd=sd.tolist(), ok=ok(y, tgt).tolist(), score=float(sc_), health=hl,
                gains=gains(i), evals=len(hist), history=hist)


def init_tgt():
    global TGT
    TGT = np.load(os.path.join(ROOT, WALKS, "targets.npy"))


def do_targets(a):
    os.makedirs(os.path.join(ROOT, WALKS), exist_ok=True)
    T = targets()
    np.save(os.path.join(ROOT, WALKS, "targets.npy"), T)
    for i, c in enumerate(HEX_ORDER):
        print(f"{i:2d} {c:<16} -> {B1_NAME[FAMILY[i]][i % 4]:<16} target {np.round(T[i], 4)}")


def do_tune(a):
    only = a.only if a.only else list(range(24))
    t0 = time.time()
    with Pool(a.workers, initializer=init_tgt) as P:
        res = P.map(tune_one, only, chunksize=1)
    old = json.load(open(os.path.join(ROOT, TUNE_JSON))) if os.path.exists(os.path.join(ROOT, TUNE_JSON)) else {}
    for r in res:
        old[str(r["i"])] = r
    json.dump(old, open(os.path.join(ROOT, TUNE_JSON), "w"), indent=1)
    print_tune(old)
    print(f"{time.time() - t0:.0f}s")


def print_tune(R):
    print(f"{'i':>2} {'hex':<16} {'b1':<16} {'target fwd/lat/yaw':<26} {'achieved':<26} {'err':<26} cmd vx/vy/wz  ok")
    f = lambda a: " ".join(f"{x:+.3f}" for x in a)  # noqa: E731
    for k in sorted(R, key=int):
        r = R[k]; i = r["i"]
        e = np.array(r["achieved"]) - np.array(r["target"])
        print(f"{i:2d} {r['hex']:<16} {B1_NAME[FAMILY[i]][i % 4]:<16} {f(r['target']):<26} {f(r['achieved']):<26} "
              f"{f(e):<26} {f(r['cmd'])}  {'OK' if all(r['ok']) else 'NO'} fell={r['health']['fell']} n={r['evals']}")


def walk_path(i):
    return os.path.join(ROOT, WALKS, f"{HEX_ORDER[i]}.npz")


def contact_phase(fc, starts):
    """Gait phase in [0, 1) at each start step from foot-contact touchdowns (foot column 0), period =
    median touchdown interval."""
    c = fc[:, 0] > 0.5
    td = np.flatnonzero(c[1:] & ~c[:-1]) + 1
    P = float(np.median(np.diff(td)))
    ph = []
    for st in starts:
        prev = td[td <= st]
        ph.append(((st - prev[-1]) / P) % 1.0 if len(prev) else np.nan)
    return np.array(ph), P


def walk_one(i):
    R = json.load(open(os.path.join(ROOT, TUNE_JSON)))[str(i)]
    n = STEADY + 4 * PER + 3 * max(GAPS) + 10
    tmp = tempfile.mkdtemp(prefix=f"b1walk{i}_")
    T = rollout(np.array(R["cmd"]), i, n, os.path.join(tmp, "w.npz"))
    best = None
    for g in GAPS:
        st = [STEADY + w * (PER + g) for w in range(4)]
        ph, P = contact_phase(T["foot_contact"], st)
        d = np.abs(ph[:, None] - ph[None, :]); d = np.minimum(d, 1 - d)
        sc = d[np.triu_indices(4, 1)].min()
        if best is None or sc > best[0] + 1e-9:
            best = (sc, g, st, ph, P)
    sc, g, st, ph, P = best
    T.update(hex_condition=HEX_ORDER[i], b1_condition=B1_NAME[FAMILY[i]][i % 4], cond_index=i, family=FAMILY[i],
             family_level=i % 4, walk_gap=g, window_starts=np.array(st), start_phase=ph, gait_period_steps=P,
             tune_cmd=np.array(R["cmd"]), tune_target=np.array(R["target"]), tune_achieved=np.array(R["achieved"]))
    np.savez_compressed(walk_path(i), **T)
    clips = [label(window_clip(T, s)).mean(0) for s in st]
    return i, g, st, ph, P, health(T), np.mean(clips, 0)


def do_walks(a):
    with Pool(a.workers) as P:
        res = P.map(walk_one, range(24), chunksize=1)
    tg = np.load(os.path.join(ROOT, WALKS, "targets.npy"))
    for i, g, st, ph, Pp, hl, m in res:
        print(f"{i:2d} {HEX_ORDER[i]:<16} gap {g} starts {st} phase {np.round(ph, 2)} period {Pp:.1f} steps "
              f"fell {hl['fell']} zmin {hl['z_min']:.3f} upmin {hl['up_min']:.3f} windows mean {np.round(m, 3)} "
              f"target {np.round(tg[i], 3)}")


SEND = ("joint_pos", "joint_vel", "action", "command", "foot_contact", "base_pos", "base_quat", "com_pos")


def do_cut(a):
    if a.dry_run:
        return cut_dry_run()
    for s in ("train", "val", "heldout"):
        os.makedirs(os.path.join(ROOT, OUT, CUT.format(s)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="b1cut_")
    t0 = time.time()
    for i in range(24):
        T = dict(np.load(walk_path(i), allow_pickle=True))
        for w, st in enumerate(T["window_starts"]):
            st = int(st)
            role = ROLES[(w + i) % 4]
            split, seed = split_of(role), seed_of(i, role)
            ep = 50000 + 10 * i + w
            dst = os.path.join(ROOT, OUT, CUT.format(split), f"b1_ep{ep}.npz")
            if os.path.exists(dst):
                continue
            # the 50 Hz window, translated so its first base position is the room centre (orientation kept,
            # as the hexapod v4 windows); render_b1_replay builds the room around frame 0 (--spawn 0 0)
            piece = {k: T[k][st:st + PER].copy() for k in SEND}
            off = -piece["base_pos"][0, :2].astype(np.float64)
            piece["base_pos"][:, :2] += off
            piece["com_pos"][:, :2] += off
            piece.update({k: T[k] for k in T if k.startswith("rollout_")}, dt=T["dt"],
                         joint_order_sdk=T["joint_order_sdk"])
            src = os.path.join(tmp, f"w{ep}.npz")
            np.savez(src, **piece)
            outd = os.path.join(tmp, f"o{ep}")
            r = subprocess.run([PY, os.path.join(ROOT, "sim/render/render_b1_replay.py"), "--port", str(a.port),
                                "--scene", "sim/env/b1_flat.ttt", "--traj", src, "--out", outd, "--fps", "20",
                                "--ego", "--match_floor", "--ground_uv_mult", "1.0", "--ego_seed", str(seed),
                                "--spawn", "0", "0"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                raise SystemExit(r.stdout[-500:] + r.stderr[-1500:])
            room = [ln for ln in r.stdout.splitlines() if "ego camera" in ln]
            with np.load(os.path.join(outd, f"w{ep}.npz"), allow_pickle=True) as f:
                out = {k: f[k] for k in f.files}
            assert out["frames"].shape == (66, 256, 256, 3), out["frames"].shape
            assert np.abs(out["base_pos"] - piece["base_pos"][KEEP]).max() < 1e-6
            assert np.abs(out["com_pos"] - piece["com_pos"][KEEP]).max() < 1e-9
            beh = B1_BEH[FAMILY[i]]
            out.update(condition=np.array(B1_NAME[FAMILY[i]][i % 4]), behaviour=np.array(beh),
                       level=np.array(i % 4), family=np.array(FAMILY[i]), family_level=np.array(i % 4),
                       cond_index=np.array(i), hex_condition=np.array(HEX_ORDER[i]), embodiment=np.array("b1"),
                       expert_episode=np.array(ep), policy=np.array("gait3"), room_seed=np.array(seed),
                       ego_seed=np.array(seed), window_start=np.array(st), window_index=np.array(w),
                       copy=np.array(role), split=np.array(split),
                       source_walk=np.array(os.path.relpath(walk_path(i), ROOT)),
                       start_phase=np.array(T["start_phase"][w]), offset_xy=off,
                       tune_cmd=T["tune_cmd"], tune_target=T["tune_target"],
                       render=np.array("render_b1_replay --ego --match_floor --ground_uv_mult 1.0 --fps 20 "
                                       "--spawn 0 0, re-centred (translation only), fov 90; " + (room[0].strip() if room else "")))
            tmpd = dst[:-4] + ".tmp.npz"
            np.savez_compressed(tmpd, **out)
            os.replace(tmpd, dst)
            print(f"{HEX_ORDER[i]:<16} w{w} start {st:3d} -> {split:<7} seed {seed:3d} ep{ep}  {time.time() - t0:.0f}s",
                  flush=True)


def cut_dry_run():
    """`cut` without rendering: the walks it reads and the clips it would write; each existing clip must carry the
    window's (episode, condition, split, copy, seed, window start, source walk)."""
    n = bad = 0
    for i in range(24):
        wp = walk_path(i)
        with np.load(wp, allow_pickle=True) as T:
            starts = [int(x) for x in T["window_starts"]]
        print(f"read {os.path.relpath(wp, ROOT)}")
        for w, st in enumerate(starts):
            role = ROLES[(w + i) % 4]
            split, seed, ep = split_of(role), seed_of(i, role), 50000 + 10 * i + w
            dst = os.path.join(ROOT, OUT, CUT.format(split), f"b1_ep{ep}.npz")
            n += 1
            if not os.path.exists(dst):
                print(f"  would render {os.path.relpath(dst, ROOT)}")
                continue
            with np.load(dst, allow_pickle=True) as d:
                got = (int(d["expert_episode"]), int(d["cond_index"]), str(d["split"]), str(d["copy"]),
                       int(d["room_seed"]), int(d["window_start"]), str(d["source_walk"]))
            want = (ep, i, split, role, seed, st, os.path.relpath(wp, ROOT))
            bad += got != want
            if got != want:
                print(f"  MISMATCH {os.path.relpath(dst, ROOT)}: {got} != {want}")
    print(f"cut --dry_run: {n} windows, existing {sum(len(v) for v in b1_files().values())}, mismatches {bad}")


def b1_files():
    return {s: sorted(glob.glob(os.path.join(ROOT, OUT, CUT.format(s), "*.npz"))) for s in ("train", "val", "heldout")}


def do_check(a):
    files = b1_files()
    print("counts", {s: len(v) for s, v in files.items()})
    good = True
    rows = []
    for s, ps in files.items():
        for p in ps:
            clip = E.load(p, E.B1)
            bm = clip["body_motion"].astype(np.float64)
            fin = bool(np.isfinite(bm).all() and np.isfinite(clip["actions"]).all())
            with np.load(p, allow_pickle=True) as d:
                r = dict(split=s, i=int(d["cond_index"]), copy=str(d["copy"]), seed=int(d["room_seed"]),
                         start=int(d["window_start"]), phase=float(d["start_phase"]), dt=float(d["dt"]),
                         act=d["action"].astype(np.float64), mean=bm.mean(0), fin=fin,
                         shape=clip["frames"].shape, xy=np.abs(d["base_pos"][:, :2]).max())
            rows.append(r)
            good &= fin and r["shape"] == (66, 256, 256, 3) and abs(r["dt"] - 0.05) < 1e-9
    print(f"(i) loader ok on {len(rows)} files, finite labels {all(r['fin'] for r in rows)}, frames 66x256x256x3, dt 0.05: {good}")
    tg = np.load(os.path.join(ROOT, WALKS, "targets.npy"))
    print("\n(ii) clip-mean Froude (fwd, lat, yaw): hexapod target (mean of its 4 clips) | B1 mean of 4 clips | error | "
          "B1 per-clip range | within max(10%, 0.005)")
    f = lambda a: " ".join(f"{x:+.3f}" for x in a)  # noqa: E731
    nbad = 0
    for i in range(24):
        m = np.array([r["mean"] for r in rows if r["i"] == i])
        e = m.mean(0) - tg[i]
        okc = ok(m.mean(0), tg[i])
        nbad += int(not okc.all())
        print(f"{i:2d} {HEX_ORDER[i]:<16} {B1_NAME[FAMILY[i]][i % 4]:<16} {f(tg[i])} | {f(m.mean(0))} | {f(e)} | "
              f"{f(m.min(0))} .. {f(m.max(0))} | {'OK' if okc.all() else 'OUT ' + str(okc)}")
    print(f"  conditions out of tolerance: {nbad}")
    good &= nbad == 0
    # (iii) seeds identical to the hexapod
    hx = {}
    for s in ("train", "val", "heldout"):
        for p in glob.glob(os.path.join(ROOT, OUT, f"c10_clips_{s}", "*.npz")):   # deterministic hexapod (F305)
            with np.load(p, allow_pickle=True) as d:
                hx[(s, int(d["cond_index"]), str(d["copy"]))] = int(d["room_seed"])
    bx = {(r["split"], r["i"], r["copy"]): r["seed"] for r in rows}
    print(f"\n(iii) room seed per (split, condition, copy) identical to the hexapod: {hx == bx} ({len(bx)} keys)")
    good &= hx == bx
    # (iv) distinct
    dm = min(min(np.abs(x["act"] - y["act"]).max() for k, x in enumerate(rs) for y in rs[k + 1:])
             for rs in ([r for r in rows if r["i"] == i] for i in range(24)))
    print(f"(iv) smallest same-condition max|action diff| {dm:.3f}; start phases per condition: "
          f"{sorted({tuple(sorted(round(r['phase'], 2) for r in rows if r['i'] == i)) for i in range(24)})}")
    good &= dm > 1e-3
    # (v) falls
    hl = [health(dict(np.load(walk_path(i), allow_pickle=True))) for i in range(24)]
    print(f"(v) falls in walks: {[HEX_ORDER[i] for i, h in enumerate(hl) if h['fell']] or 'none'}; base z min "
          f"{min(h['z_min'] for h in hl):.3f} m, min upright {min(h['up_min'] for h in hl):.3f}; window max |x|,|y| "
          f"from room centre {max(r['xy'] for r in rows):.2f} m")
    good &= not any(h["fell"] for h in hl)
    print("\nALL GATES", "PASS" if good else "FAIL")


def do_video(a):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    sel = a.video_conds
    W, H = 256, 90
    def load(body, i, split, copy):
        pat = "c10_clips" if body == "hex" else "b1_clips"
        for p in glob.glob(os.path.join(ROOT, OUT, f"{pat}_{split}", "*.npz")):
            with np.load(p, allow_pickle=True) as d:
                if int(d["cond_index"]) == i and str(d["copy"]) == copy:
                    bm = E.load(p, E.HEXAPOD if body == "hex" else E.B1)["body_motion"]
                    return d["frames"], bm, int(d["room_seed"]), p
        raise FileNotFoundError((body, i, split, copy))
    rows = []
    for c, copy in sel:
        i = HEX_ORDER.index(c)
        split = split_of(copy)
        hf, hb, hs, _ = load("hex", i, split, copy)
        bf, bb, bs, _ = load("b1", i, split, copy)
        assert hs == bs
        rows.append((c, B1_NAME[FAMILY[i]][i % 4], split, copy, hs, hf, hb, bf, bb))
    cols = [(255, 80, 80), (80, 200, 80), (90, 140, 255)]

    def trace(hb, bb, t):
        im = Image.new("RGB", (2 * W, H), (20, 20, 20)); dr = ImageDraw.Draw(im)
        lo, hi = min(hb.min(), bb.min(), -0.05), max(hb.max(), bb.max(), 0.05)
        y = lambda v: H - 4 - (v - lo) / (hi - lo) * (H - 8)  # noqa: E731
        dr.line([(0, y(0)), (2 * W, y(0))], fill=(80, 80, 80))
        for ch in range(3):
            for arr, dash in ((hb, False), (bb, True)):
                pts = [(k * (2 * W - 1) / 65, y(arr[k, ch])) for k in range(66)]
                if dash:
                    for k in range(0, 65, 2):
                        dr.line([pts[k], pts[k + 1]], fill=cols[ch])
                else:
                    dr.line(pts, fill=cols[ch], width=2)
        dr.line([(t * (2 * W - 1) / 65, 0), (t * (2 * W - 1) / 65, H)], fill=(255, 255, 255))
        dr.text((3, 2), f"fwd red / lat green / yaw blue; hex solid, B1 dashed  hex {hb[t, 0]:+.3f} {hb[t, 1]:+.3f} "
                f"{hb[t, 2]:+.3f}", fill=(230, 230, 230))
        dr.text((3, 14), f"B1 {bb[t, 0]:+.3f} {bb[t, 1]:+.3f} {bb[t, 2]:+.3f}", fill=(230, 230, 230))
        return np.asarray(im)

    os.makedirs(os.path.join(ROOT, "results/check"), exist_ok=True)
    dst = os.path.join(ROOT, "results/check/b1_hex_matched_samples.mp4")
    wr = imageio.get_writer(dst, fps=10)
    for t in range(66):
        blocks = []
        for c, b1n, split, copy, seed, hf, hb, bf, bb in rows:
            tiles = []
            for fr, lab in ((hf, f"hexapod {c}"), (bf, f"B1 {b1n}")):
                im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, W - 1, 26], fill=(0, 0, 0))
                dr.text((3, 2), lab, fill=(255, 255, 255))
                dr.text((3, 14), f"{split} {copy} room {seed} t{t}", fill=(255, 255, 255))
                tiles.append(np.asarray(im))
            blocks.append(np.concatenate([np.concatenate(tiles, 1), trace(hb, bb, t)], 0))
        top = np.concatenate(blocks[:2], 1)
        bot = np.concatenate(blocks[2:4], 1)
        wr.append_data(np.concatenate([top, bot], 0))
    wr.close()
    print("->", os.path.relpath(dst, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("targets", "tune", "show", "walks", "cut", "check", "video"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", type=int, nargs="*", default=[])
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--dry_run", action="store_true", help="cut: list reads / writes, check existing clips, no render")
    ap.add_argument("--video_conds", nargs=8, default=["speed_c5.8_bwd", "val", "turn_s0.56_neg", "heldout",
                                                       "side_L_lvl3", "train0", "speed_c8.15", "train1"])
    a = ap.parse_args()
    a.video_conds = list(zip(a.video_conds[::2], a.video_conds[1::2]))
    os.chdir(ROOT)
    if a.step == "show":
        print_tune(json.load(open(TUNE_JSON)))
        return
    {"targets": do_targets, "tune": do_tune, "walks": do_walks, "cut": do_cut,
     "check": do_check, "video": do_video}[a.step](a)


if __name__ == "__main__":
    main()
