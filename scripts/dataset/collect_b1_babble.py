"""B1 babbling data (DATA_PLAN section 10): B1 as a new robot, adapted with NO task knowledge.
-> data/counterfactual_walks/b1_babble_{train,val,heldout_pool}/b1_babble_ep<code>.npz

Commands: (vx, vy, wz) drawn uniform over the walking policy's own training command range
(sim/assets/b1_policy/base_gait3/train_config.yaml `commands`: lin_vel_x [-0.5, 0.5], lin_vel_y [-0.4, 0.4],
ang_vel_z [-0.6, 0.6]) -- never from the hexapod-tuned commands (b1_walks/tuning.json) or any Froude target. Piecewise
constant, a switch every U{50..100} policy steps (1-2 s at 50 Hz). Applied exactly as every B1 collector: vx / vy step
changes straight into the policy observation, wz integrated into the heading target of the heading controller
(rollout_b1_mujoco.py loop, here via build_b1_cf_branches.Roll, step for step identical). Heading gains PI kp 2.5 /
ki 1.0 (F69, the faster of the two existing B1 controller settings: time constant 0.4 s vs 2 s for kp 0.5, so the
commanded yaw rate is reached inside a 1-2 s segment).

Episode: 45 unlogged policy-warmup steps (rollout convention, first command held), then LEAD = 50 + 4 x 165 + 1 steps.
Windows = steps [LEAD0 + 165 j, LEAD0 + 165 (j + 1)), j = 0..3, subsampled to 66 frames at 0.05 s (KEEP, as every B1
clip). A window is usable if no step of it nor the 25 steps (10 frames) after it is fallen (base z < 0.35 m or upright
< 0.7, collect_b1_walks.health rule); a fall ends the episode. Episodes per split are disjoint (episode seeds: train
0.., val 1000.., heldout_pool 2000..); usable windows kept in (episode, j) order until the split's count, the rest
unused. `segment` per frame = index of the command applied in the step after the frame's state (a switch frame = the
first state of the new segment, the branch convention `first_pair`), renumbered from 0 per window.

Froude: labels by the loader at `com_pos` with `froude_height` = 0.5274 m = median over b1_clips_* of the per-clip
median CoM z (the scale the B1 clips use), one constant for every babble window (DATA_PLAN 9.5 D6).

Rooms: one per window; seeds train 300-491, val 500-523, heldout_pool 600-695. Size log-uniform 8.0 .. 26.47 m
(render_shift_heldout.size_range), stratified per split, SIZE_SEED + split index; offset uniform over the valid box
(camera >= rr margin_b1 = 2 x B1 max camera height inside every wall, camera xy estimated as base xy +- 0.40 m, the
largest camera-base distance of the B1 clips; checked afterwards on the rendered cam_pose), midpoint if empty. Rendering
= render_shift_heldout.b1_render rs path, the same as rr_b1_* clips.

    PY=.venv/bin/python3; S=scripts/dataset/collect_b1_babble.py
    $PY $S physics --workers 4
    $PY $S place
    $PY $S launch --ports ...; $PY $S render --ports ...; $PY $S stop
    $PY $S check; $PY $S review
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render", "sim/scene"):
    sys.path.insert(0, os.path.join(ROOT, p))

CW = os.path.join(ROOT, "data/counterfactual_walks")
WALKS = os.path.join(CW, "b1_babble_walks")
ROOMS = os.path.join(CW, "b1_babble_rooms.json")
CHK = os.path.join(ROOT, "results/check/b1_babble")
PIDS = os.path.join(CHK, "pids.json")
COPPELIA = os.path.expanduser("~/CoppeliaSim")
MODEL_REL = "sim/assets/b1_mujoco/b1_flat.xml"
CFG = "sim/assets/b1_policy/base_gait3/train_config.yaml"
GAINS = (2.5, 1.0)
PER = 165
KEEP = np.unique(np.round(np.arange(0, PER, 2.5)).astype(int))
assert len(KEEP) == 66
LEAD0, NWIN, AFTER = 50, 4, 25
NSTEPS = LEAD0 + NWIN * PER + AFTER + 1
SEG_STEPS = (50, 100)
FROUDE_H = 0.5274
SPLITS = {"train": (192, list(range(300, 492)), 0), "val": (24, list(range(500, 524)), 1000),
          "heldout_pool": (96, list(range(600, 696)), 2000)}
SIZE_SEED, OFFSET_SEED, CMD_SEED = 20261010, 20261011, 20261012
CAM_BASE = 0.40
MARGIN_K = 2.0


def atomic_savez(dst, **arrays):
    """Durable atomic write: temp file written + fsynced, renamed, directory fsynced (a power cut leaves either the
    old state or the complete file, never a renamed-but-empty one)."""
    tmp = dst[:-4] + f".tmp{os.getpid()}_{threading.get_ident()}.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst)
    fd = os.open(os.path.dirname(dst) or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def complete(p):
    try:
        with np.load(p) as d:
            [d[k] for k in d.files]
        return True
    except Exception:  # noqa: BLE001
        return False


def cmd_range():
    import yaml
    c = yaml.safe_load(open(os.path.join(ROOT, CFG)))["commands"]
    return np.array([c["lin_vel_x"], c["lin_vel_y"], c["ang_vel_z"]], float)   # 3 x (lo, hi)


def plan(ep):
    """Per recorded step (vx, vy, wz) and segment index; the warmup holds plan[0]."""
    R = cmd_range()
    rng = np.random.default_rng([CMD_SEED, ep])
    P, seg, segs, s, k = np.zeros((NSTEPS, 3)), np.zeros(NSTEPS, np.int64), [], 0, 0
    while s < NSTEPS:
        n = int(rng.integers(SEG_STEPS[0], SEG_STEPS[1] + 1))
        c = R[:, 0] + rng.random(3) * (R[:, 1] - R[:, 0])
        P[s:s + n], seg[s:s + n] = c, k
        segs.append(dict(start=s, steps=n, cmd=c.tolist()))
        s += n; k += 1
    return P, seg, segs


def fallen(T):
    q = T["base_quat"].astype(np.float64)
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return (T["base_pos"][:, 2] < 0.35) | (up < 0.7)


def episode(ep):
    import torch
    torch.set_num_threads(1)
    from build_b1_cf_branches import Roll, POLICY_WARMUP, FIELDS
    from wm.data.com import b1_com
    dst = os.path.join(WALKS, f"ep{ep:04d}.npz")
    if os.path.exists(dst) and complete(dst):
        return ep, "skip"
    P, seg, segs = plan(ep)
    r = Roll("gait3", GAINS, os.path.join(ROOT, MODEL_REL))
    for _ in range(POLICY_WARMUP):
        r.step(*P[0])
    L = {f: [] for f in FIELDS}
    for s in range(NSTEPS):
        o = r.step(*P[s])
        for f in FIELDS:
            L[f].append(o[f])
    T = {f: np.asarray(v, np.float32) for f, v in L.items()}
    fl = fallen(T)
    T["fallen"] = fl
    T["com_pos"] = b1_com(T["base_pos"], T["base_quat"], T["joint_pos"], os.path.join(ROOT, MODEL_REL))
    T.update(cmd_plan=P.astype(np.float32), plan_segment=seg, segments=np.array(json.dumps(segs)),
             episode=np.int64(ep), dt_step=np.float64(0.02),
             **{f"rollout_{k}": np.array(v) for k, v in r.settings().items()},
             rollout_policy_warmup=np.int64(POLICY_WARMUP), rollout_cmd_seed=np.int64(CMD_SEED),
             cmd_range=cmd_range(), cmd_range_source=np.array(CFG + " commands"))
    atomic_savez(dst, **T)
    return ep, f"falls {int(fl.sum())} first {int(np.argmax(fl)) if fl.any() else -1}"


def usable_windows(T):
    fl = T["fallen"]
    first = int(np.argmax(fl)) if fl.any() else len(fl)
    return [j for j in range(NWIN) if LEAD0 + PER * (j + 1) + AFTER <= first]


def do_physics(a):
    os.makedirs(WALKS, exist_ok=True)
    t0 = time.time()
    for split, (n, _, base) in SPLITS.items():
        need = (n + NWIN - 1) // NWIN
        eps = list(range(base, base + need))
        while True:
            with Pool(a.workers) as Pl:
                for ep, msg in Pl.imap_unordered(episode, [e for e in eps if not os.path.exists(
                        os.path.join(WALKS, f"ep{e:04d}.npz"))]):
                    print(f"{split} ep{ep} {msg} {time.time() - t0:.0f}s", flush=True)
            got = sum(len(usable_windows(np.load(os.path.join(WALKS, f"ep{e:04d}.npz")))) for e in eps)
            if got >= n:
                break
            eps += list(range(eps[-1] + 1, eps[-1] + 1 + max(1, (n - got + NWIN - 1) // NWIN)))
        print(f"{split}: {len(eps)} episodes, {got} usable windows (need {n})", flush=True)


def windows():
    """split -> [(episode, j, room_seed)] in (episode, j) order."""
    out = {}
    for split, (n, seeds, base) in SPLITS.items():
        W = []
        for e in range(base, base + 999):
            p = os.path.join(WALKS, f"ep{e:04d}.npz")
            if len(W) >= n or not os.path.exists(p):
                break
            with np.load(p) as T:
                W += [(e, j) for j in usable_windows({"fallen": T["fallen"]})]
        W = W[:n]
        assert len(W) == n, (split, len(W))
        out[split] = [(e, j, seeds[k]) for k, (e, j) in enumerate(W)]
    return out


def window_piece(T, j):
    st = LEAD0 + PER * j
    idx = st + KEEP
    out = {k: T[k][idx].copy() for k in ("joint_pos", "joint_vel", "action", "command", "foot_contact", "base_pos",
                                         "base_quat", "com_pos")}
    off = -out["base_pos"][0, :2].astype(np.float64)
    out["base_pos"][:, :2] += off
    out["com_pos"][:, :2] += off
    gseg = T["plan_segment"][idx + 1]
    out["segment"] = (gseg - gseg[0]).astype(np.int8)
    out["cmd_plan"] = T["cmd_plan"][idx + 1]
    return out, off, st, gseg


def do_place(a):
    import render_shift_heldout as RS
    rr = json.load(open(os.path.join(CW, "rr_rooms.json")))
    margin = float(rr["margin"]["b1"])
    lo_s, hi_s = RS.size_range()
    W = windows()
    rooms = {}
    for si, (split, L) in enumerate(W.items()):
        n = len(L)
        rng = np.random.default_rng(SIZE_SEED + si)
        perm, u = rng.permutation(n), rng.random(n)
        S = np.exp(np.log(lo_s) + (perm + u) / n * (np.log(hi_s) - np.log(lo_s)))
        for k, (e, j, seed) in enumerate(L):
            T = np.load(os.path.join(WALKS, f"ep{e:04d}.npz"))
            piece, *_ = window_piece({k_: T[k_] for k_ in T.files}, j)
            xy = piece["base_pos"][:, :2].astype(np.float64)
            pmin, pmax = xy.min(0) - CAM_BASE, xy.max(0) + CAM_BASE
            h = S[k] / 2
            lo, hi = -(h - margin) - pmin, (h - margin) - pmax
            u2 = np.random.default_rng([OFFSET_SEED, seed]).random(2)
            o = np.where(lo <= hi, lo + u2 * (hi - lo), (lo + hi) / 2)
            rooms[str(seed)] = dict(size=float(S[k]), offset=o.tolist(), shrunk=[bool(lo[i] > hi[i]) for i in range(2)],
                                    split=split, episode=e, window=j)
    json.dump(dict(size_range=[lo_s, hi_s], size_seed=SIZE_SEED, offset_seed=OFFSET_SEED, margin=margin,
                   cam_base_bound=CAM_BASE, windows={s: L for s, L in W.items()},
                   rule="as render_random_room.place: size stratified log-uniform per split; offset uniform over the "
                        "valid box (camera >= margin inside every wall), midpoint if an axis has none",
                   rooms=rooms), open(ROOMS + ".tmp", "w"), indent=1)
    os.replace(ROOMS + ".tmp", ROOMS)
    print(f"{len(rooms)} rooms, shrunk {sum(any(r['shrunk']) for r in rooms.values())}, sizes "
          f"{min(r['size'] for r in rooms.values()):.2f}..{max(r['size'] for r in rooms.values()):.2f}")


def out_path(split, e, j):
    return os.path.join(CW, f"b1_babble_{split}", f"b1_babble_ep{70000 + 10 * e + j}.npz")


def do_launch(a):
    os.makedirs(CHK, exist_ok=True)
    pids = json.load(open(PIDS)) if os.path.exists(PIDS) else {}
    for port in a.ports:
        pr = subprocess.Popen(f"tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort={port} "
                              f"-GzmqRemoteApi.cntPort={port + 1}", shell=True, cwd=COPPELIA, start_new_session=True,
                              stdout=open(os.path.join(CHK, f"csim_{port}.log"), "a"), stderr=subprocess.STDOUT)
        pids[str(port)] = pr.pid
    json.dump(pids, open(PIDS, "w"))
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    for port in a.ports:
        for _ in range(60):
            try:
                RemoteAPIClient("localhost", port=port).require("sim").getSimulationState()
                print(f"port {port} up"); break
            except Exception:  # noqa: BLE001
                time.sleep(2)


def do_stop(a):
    pids = json.load(open(PIDS)) if os.path.exists(PIDS) else {}
    for port, pg in list(pids.items()):
        for s in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(int(pg), s)
            except ProcessLookupError:
                break
            time.sleep(2)
        print(f"stopped {port}"); pids.pop(port)
    json.dump(pids, open(PIDS, "w"))


def do_render(a):
    import render_shift_heldout as RS
    from render_b1_replay import CAM_POSE_CONVENTION
    RM = json.load(open(ROOMS))
    J = [(s, e, j, sd) for s, L in RM["windows"].items() for e, j, sd in L if not complete(out_path(s, e, j))]
    if a.limit:
        J = J[:a.limit]
    print(f"{len(J)} windows to render", flush=True)
    cache, lock = {}, threading.Lock()

    def ep_data(e):
        with lock:
            if e not in cache:
                with np.load(os.path.join(WALKS, f"ep{e:04d}.npz")) as T:
                    cache[e] = {k: T[k] for k in T.files}
            return cache[e]

    def fn(sim, job):
        split, e, j, seed = job
        T = ep_data(e)
        piece, off, st, gseg = window_piece(T, j)
        room = RM["rooms"][str(seed)]
        d = dict(piece, room_seed=np.int64(seed))
        fr, cp, R = RS.b1_render(sim, d, -np.asarray(room["offset"], float), RS.rs_room(room["size"]))
        assert fr.shape == (66, 256, 256, 3)
        h = room["size"] / 2
        rel = cp[:, :2] + np.asarray(room["offset"])            # camera relative to the room centre
        clear = float((h - np.abs(rel)).min())
        out = dict(piece)
        out.update(
            frames=fr, cam_pose=cp, cam_pose_convention=np.array(CAM_POSE_CONVENTION), cam_pose_parent=np.array("base_visual"),
            joint_order_sdk=np.array([f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL") for s in ("hip", "thigh", "calf")]),
            dt=np.float64(0.05), fps=np.float64(20.0), froude_height=np.float64(FROUDE_H),
            condition=np.array("babble"), family=np.array("babble"), behaviour=np.array("babble"),
            level=np.int64(-1), family_level=np.int64(-1), cond_index=np.int64(-1), embodiment=np.array("b1"),
            expert_episode=np.int64(70000 + 10 * e + j), policy=np.array("gait3"), room_seed=np.int64(seed),
            ego_seed=np.int64(seed), window_start=np.int64(st), window_index=np.int64(j), split=np.array(split),
            babble_episode=np.int64(e), source_walk=np.array(os.path.relpath(os.path.join(WALKS, f"ep{e:04d}.npz"), ROOT)),
            offset_xy=off, segment_global=gseg.astype(np.int64),
            segments=np.array(json.dumps([sg for sg in json.loads(str(T["segments"]))
                                          if sg["start"] < st + PER and sg["start"] + sg["steps"] > st])),
            cmd_range=T["cmd_range"], cmd_range_source=T["cmd_range_source"], cmd_seed=np.int64(CMD_SEED),
            **{k: T[k] for k in T if k.startswith("rollout_")},
            room_size=np.float64(room["size"]), rr_room_size=np.float64(room["size"]),
            rr_room_offset=np.asarray(room["offset"], np.float64), rr_room_size_original=np.float64(np.nan),
            rr_size_seed=np.int64(SIZE_SEED), rr_offset_seed=np.int64(OFFSET_SEED),
            rr_offset_shrunk=np.asarray(room["shrunk"], bool), rr_margin=np.float64(RM["margin"]),
            rr_room_height=np.float64(R["height"]), rr_ground_uv=np.float64(R["ground_uv"]),
            rr_cam_clearance=np.float64(clear), rr_source=np.array("babble (rendered directly)"),
            rr_render=np.array("render_shift_heldout.b1_render rs path (as rr_b1_*): room size rr_room_size, window start "
                               "at rr_room_offset from the room centre, heading kept, FOV 90"))
        dst = out_path(split, e, j)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        atomic_savez(dst, **out)
        return f"S {room['size']:.1f} clear {clear:.2f}"
    RS.run_pool(J, a.ports, fn, "babble")


def files(split):
    return sorted(glob.glob(os.path.join(CW, f"b1_babble_{split}", "*.npz")))


def do_check(a):
    import wm.data.embodiment as E
    os.makedirs(CHK, exist_ok=True)
    rep, good = [], True
    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); rep.append(s)
    RM = json.load(open(ROOMS))
    # counts, splits, rooms
    eps, seeds, keys = {}, {}, set()
    for s in SPLITS:
        fs = files(s)
        say(f"{s}: {len(fs)} files (want {SPLITS[s][0]}), tmp leftovers {len(glob.glob(os.path.join(CW, f'b1_babble_{s}', '*.tmp*')))}")
        good &= len(fs) == SPLITS[s][0]
        eps[s], seeds[s] = set(), []
        for p in fs:
            with np.load(p) as d:
                eps[s].add(int(d["babble_episode"])); seeds[s].append(int(d["room_seed"]))
                keys.add((int(d["babble_episode"]), int(d["window_index"])))
    ov = [(x, y) for x in SPLITS for y in SPLITS if x < y and eps[x] & eps[y]]
    say(f"episodes per split {[len(eps[s]) for s in SPLITS]}, episode overlap between splits: {ov or 'none'}; "
        f"distinct (episode, window) {len(keys)}")
    exist = set()
    for p in glob.glob(os.path.join(CW, "*_clips_*", "*.npz")) + glob.glob(os.path.join(CW, "rr_*_clips_*", "*.npz")):
        if "babble" in p:
            continue
        with np.load(p) as d:
            exist.add(int(d["room_seed"]))
    for s in SPLITS:
        ok = sorted(seeds[s]) == SPLITS[s][1]
        say(f"{s} room seeds {min(seeds[s])}-{max(seeds[s])} each once: {ok}; overlap with existing clip rooms: "
            f"{sorted(set(seeds[s]) & exist) or 'none'}")
        good &= ok and not (set(seeds[s]) & exist)
    # labels
    rng = np.random.default_rng(0)
    nsw, bad_ind, hs, cross, fin, falls, clear = 0, 0, set(), 0, True, 0, []
    allf = [p for s in SPLITS for p in files(s)]
    seg_means = []
    for p in allf:
        with np.load(p, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files if k != "frames"}
        class D(dict):
            files = property(lambda self: list(self.keys()))
        lab = E._b1(D(d))["body_motion"].astype(np.float64)
        fin &= bool(np.isfinite(lab).all())
        hs.add(float(d["froude_height"]))
        seg = d["segment"].astype(int)
        assert seg[0] == 0 and np.all(np.diff(seg) >= 0) and np.all(np.diff(seg) <= 1)
        # segment = command index; command constant inside each segment
        for k in np.unique(seg):
            cp_ = d["cmd_plan"][seg == k]
            cross += int(np.abs(cp_ - cp_[0]).max() > 0)
            m = seg == k
            if m.sum() >= 10:
                seg_means.append(lab[m][3:-3].mean(0) if m.sum() > 12 else lab[m].mean(0))
        for b in np.flatnonzero(np.diff(seg)) + 1:
            nsw += 1
            e = D(d); e["com_pos"] = d["com_pos"].copy(); e["base_quat"] = d["base_quat"].copy()
            e["com_pos"][:b] += rng.normal(0, 0.3, e["com_pos"][:b].shape)
            q = rng.normal(size=(b, 4)); e["base_quat"][:b] = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
            l2 = E._b1(e)["body_motion"].astype(np.float64)
            nxt = np.flatnonzero(seg > seg[b])
            end = nxt[0] if len(nxt) else len(seg)
            bad_ind += int(not np.array_equal(l2[b:end], lab[b:end]))
        # loader height = froude_height (labels scale 1/sqrt(h)): recompute with a different stored h
        e = D(d); e["froude_height"] = np.float64(4 * FROUDE_H)
        l3 = E._b1(e)["body_motion"].astype(np.float64)
        good &= np.allclose(l3[:, :2] * 2, lab[:, :2], rtol=1e-6, atol=1e-9)
        q = d["base_quat"].astype(np.float64)
        falls += int(((d["base_pos"][:, 2] < 0.35) | (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) < 0.7)).any())
        clear.append(float(d["rr_cam_clearance"]) - float(d["rr_margin"]))
    say(f"labels finite: {fin}; froude_height values {sorted(hs)} (B1 clips' median CoM z {FROUDE_H}); loader uses it "
        f"(fwd/lat scale 1/sqrt(h) check): {good}")
    say(f"segment switches inside windows: {nsw}; labels after a switch independent of the prefix (pose noise before "
        f"the switch): {nsw - bad_ind}/{nsw}; segments whose command changes inside: {cross}")
    say(f"windows with a fallen frame: {falls}; camera clearance beyond margin: min {min(clear):.2f} m "
        f"({sum(c < 0 for c in clear)} windows inside the margin)")
    good &= fin and bad_ind == 0 and cross == 0 and falls == 0 and len(hs) == 1
    # loader + pairs
    from wm.data.dataset import MultiEmbodimentPairs
    for s in SPLITS:
        ds = MultiEmbodimentPairs([(files(s), "b1")], lazy_frames=True, frame_stride=5, rollout_k=2)
        x = ds[0]
        say(f"MultiEmbodimentPairs({s}, lazy, stride 5, rollout 2): {len(ds)} pairs; item keys {sorted(x)[:8]}...")
    # coverage
    B = []
    for p in sorted(glob.glob(os.path.join(CW, "b1_clips_*", "*.npz"))):
        with np.load(p) as z:
            B.append((int(z["cond_index"]), str(z["condition"]), E.load(p, E.B1)["body_motion"].mean(0)))
    beh = {}
    for i, n, m in B:
        beh.setdefault((i, n), []).append(m)
    beh = {k: np.mean(v, 0) for k, v in beh.items()}
    tr = []
    for p in files("train"):
        with np.load(p) as z:
            d = {k: z[k] for k in z.files if k != "frames"}
        class D2(dict):
            files = property(lambda self: list(self.keys()))
        lab = E._b1(D2(d))["body_motion"].astype(np.float64)
        tr.append(lab)
    TR = np.concatenate(tr)
    SM = np.array(seg_means)
    say("\nCoverage (REPORTED only, never used to sample). Measured CoM Froude, channel ranges (1st..99th pct):")
    for nm, X in (("babble train frames", TR), ("babble all segment means", SM),
                  ("B1 clip behaviours (24)", np.array(list(beh.values())))):
        say(f"  {nm:<26} fwd {np.percentile(X[:, 0], 1):+.3f}..{np.percentile(X[:, 0], 99):+.3f}  "
            f"lat {np.percentile(X[:, 1], 1):+.3f}..{np.percentile(X[:, 1], 99):+.3f}  "
            f"yaw {np.percentile(X[:, 2], 1):+.3f}..{np.percentile(X[:, 2], 99):+.3f}")
    say("  per B1 behaviour: nearest babble segment mean (Euclidean in Froude, 3 channels)")
    within = 0
    for (i, n), m in sorted(beh.items()):
        dd = np.linalg.norm(SM - m, axis=1)
        within += int(dd.min() <= 0.05)
        say(f"   {i:2d} {n:<18} {np.round(m, 3)}  nearest {dd.min():.3f}  segments within 0.05: {int((dd <= 0.05).sum())}")
    say(f"  behaviours with a babble segment within 0.05: {within}/24")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BM = np.array(list(beh.values()))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax[k].scatter(TR[::3, x], TR[::3, y], s=2, alpha=0.15, c="tab:gray", label="babble train frames")
        ax[k].scatter(SM[:, x], SM[:, y], s=8, alpha=0.6, c="tab:blue", label="babble segment means")
        ax[k].scatter(BM[:, x], BM[:, y], s=40, marker="x", c="tab:red", label="B1 clip behaviours (24)")
        nm = ("fwd", "lat", "yaw")
        ax[k].set_xlabel(f"{nm[x]} Froude"); ax[k].set_ylabel(f"{nm[y]} Froude"); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("B1 babbling: measured CoM Froude vs the 24 B1 clip behaviours (reported only)")
    fig.tight_layout(); fig.savefig(os.path.join(CHK, "coverage.png"), dpi=110)
    say(f"\nALL CHECKS {'PASS' if good else 'FAIL'}")
    open(os.path.join(CHK, "summary.txt"), "w").write("\n".join(rep) + "\n")


def do_review(a):
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    import wm.data.embodiment as E
    rng = np.random.default_rng(0)
    pick = []
    for s, n in (("train", 3), ("val", 1), ("heldout_pool", 2)):
        fs = files(s)
        pick += [(s, fs[i]) for i in sorted(rng.choice(len(fs), n, replace=False))]
    rows, vids = [], []
    for s, p in pick:
        with np.load(p) as z:
            fr, seg, cmd, S, seed = z["frames"], z["segment"], z["cmd_plan"], float(z["room_size"]), int(z["room_seed"])
        lab = E.load(p, E.B1)["body_motion"]
        row = Image.new("RGB", (160 * 6 + 260, 160))
        for k, t in enumerate(np.linspace(0, 65, 6).astype(int)):
            im = Image.fromarray(fr[t]).resize((160, 160)); dr = ImageDraw.Draw(im)
            dr.text((3, 3), f"t{t} seg{seg[t]}", fill=(255, 255, 0))
            dr.text((3, 146), f"{lab[t, 0]:+.2f} {lab[t, 1]:+.2f} {lab[t, 2]:+.2f}", fill=(255, 255, 0))
            row.paste(im, (260 + 160 * k, 0))
        dr = ImageDraw.Draw(row)
        dr.text((4, 30), f"{s} {os.path.basename(p)}", fill=(255, 255, 255))
        dr.text((4, 50), f"room {seed} size {S:.1f} m", fill=(255, 255, 255))
        for k, sg in enumerate(np.unique(seg)[:4]):
            c = cmd[seg == sg][0]
            dr.text((4, 75 + 18 * k), f"seg{sg} cmd vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}", fill=(180, 220, 255))
        rows.append(row)
        vids.append((fr, seg, lab, cmd, s, os.path.basename(p)))
    sheet = Image.new("RGB", (rows[0].width, 160 * len(rows)))
    for k, r in enumerate(rows):
        sheet.paste(r, (0, 160 * k))
    sheet.save(os.path.join(CHK, "contact_sheet.png"))
    with imageio.get_writer(os.path.join(CHK, "babble_samples.mp4"), fps=10) as w:
        for t in range(66):
            tiles = []
            for fr, seg, lab, cmd, s, n in vids:
                im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 255, 40], fill=(0, 0, 0))
                dr.text((3, 2), f"{s} {n}", fill=(255, 255, 255))
                dr.text((3, 14), f"t{t} seg{seg[t]} cmd {cmd[t, 0]:+.2f} {cmd[t, 1]:+.2f} {cmd[t, 2]:+.2f}", fill=(180, 220, 255))
                dr.text((3, 26), f"Froude fwd {lab[t, 0]:+.3f} lat {lab[t, 1]:+.3f} yaw {lab[t, 2]:+.3f}", fill=(255, 255, 0))
                tiles.append(np.asarray(im))
            w.append_data(np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:6], 1)], 0))
    print("->", CHK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("physics", "place", "launch", "stop", "render", "check", "review"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ports", type=int, nargs="+", default=[25600])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    dict(physics=do_physics, place=do_place, launch=do_launch, stop=do_stop, render=do_render, check=do_check,
         review=do_review)[a.cmd](a)


if __name__ == "__main__":
    main()
