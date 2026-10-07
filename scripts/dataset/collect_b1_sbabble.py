"""B1 STRUCTURED babbling (STATUS "Adaptation direction (agreed 2026-10-07)", structured babbling rule): near-same-state,
many-futures data without resets, no task knowledge.
-> data/counterfactual_walks/b1_sbabble_{train,val}/b1_sbabble_ep<code>.npz (+ nested links b1_sbabble_n{44,88,176})

Bases: 9 commands on a coarse grid of B1's OWN policy command range (sim/assets/b1_policy/base_gait3/train_config.yaml
`commands`: vx [-0.5, 0.5], vy [-0.4, 0.4], wz [-0.6, 0.6]); "slow" = 0.3 x the range half-width, "medium" = 0.6 x
(never the hexapod-tuned b1_walks/tuning.json commands, never a Froude target):
    0 stand (0, 0, 0)          1 fwd slow (0.15, 0, 0)    2 fwd medium (0.30, 0, 0)   3 back slow (-0.15, 0, 0)
    4 turn L slow (0, 0, 0.18) 5 turn R slow (0, 0, -0.18) 6 side L slow (0, 0.12, 0)  7 side R slow (0, -0.12, 0)
    8 fwd slow + turn L slow (0.15, 0, 0.18)

Trial = one clip of 66 frames (165 policy steps at 50 Hz, subsampled at 0.05 s as every B1 clip, KEEP):
    frames 0..19 base (steady: >= SETTLE = 150 steps (3 s) of base since the last return / episode start before the
    switch), switch at frame 20 = a FR-foot touchdown (foot_contact[0] 0 -> 1; gait3 steps at 2 Hz = 25 steps even when
    standing, so every trial of every base switches at the same gait phase), then a random command of ONE family for
    75 steps (1.5 s, frames 20..49), then the base again (frames 50..65 and onwards = re-settle for the next trial).
    Family uniform over {speed: (vx, 0, 0), turn: (0, 0, wz), side: (0, vy, 0)}, level uniform over that axis' policy range.
Continuous episode per (split, base): warmup 45 policy steps (rollout convention), then trials back to back. A fall
(collect_b1_walks.health rule) anywhere in a trial window or the 25 steps after it, or during settling, drops the trial
and restarts the episode (fresh Roll); counted. Commands applied exactly as the B1 collectors (build_b1_cf_branches.Roll,
heading PI kp 2.5 / ki 1.0 as collect_b1_babble).

`segment` per frame = index of the command applied in the step after the frame's state (0 base, 1 command, 2 back);
`cmd_plan` per frame = that command. Froude via the loader at `com_pos`, froude_height 0.5274 (as b1_babble).
Counts: train 20 per base = 180 (176 + the 4 extra clips the nested n176 subset needs, like b1_babble_n*), val 24
(bases 0-5: 3, 6-8: 2). Room seeds train 700-879, val 900-923; room size / offset as collect_b1_babble.do_place.

    PY="nice -n 19 .venv/bin/python3"; S=scripts/dataset/collect_b1_sbabble.py
    $PY $S physics; $PY $S place; $PY $S launch --ports P; $PY $S render --ports P ...; $PY $S stop
    $PY $S subsets; $PY $S check; $PY $S review
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render", "sim/scene"):
    sys.path.insert(0, os.path.join(ROOT, p))
import collect_b1_babble as BB  # noqa: E402
from collect_b1_babble import atomic_savez, complete, cmd_range, fallen, KEEP, FROUDE_H, GAINS, MODEL_REL, CFG  # noqa

CW = os.path.join(ROOT, "data/counterfactual_walks")
WALKS = os.path.join(CW, "b1_sbabble_walks")
ROOMS = os.path.join(CW, "b1_sbabble_rooms.json")
CHK = os.path.join(ROOT, "results/check/b1_sbabble")
PIDS = os.path.join(CHK, "pids.json")
PER, PRE, CMD_STEPS, AFTER, SETTLE = 165, 50, 75, 25, 150
SWITCH_FRAME = 20
assert KEEP[SWITCH_FRAME] == PRE and KEEP[50] == PRE + CMD_STEPS
SLOW, MED = 0.3, 0.6
SPLITS = {"train": ([20] * 9, list(range(700, 880))), "val": ([3] * 6 + [2] * 3, list(range(900, 924)))}
SIZE_SEED, OFFSET_SEED, CMD_SEED = 20261020, 20261021, 20261022
FAMS = ("speed", "turn", "side")
FAM_AXIS = {"speed": 0, "side": 1, "turn": 2}
CAM_BASE = 0.40


def bases():
    h = cmd_range()[:, 1]                       # half-widths (symmetric range)
    s, m = SLOW * h, MED * h
    return [("stand", (0, 0, 0)), ("fwd_slow", (s[0], 0, 0)), ("fwd_medium", (m[0], 0, 0)),
            ("back_slow", (-s[0], 0, 0)), ("turnL_slow", (0, 0, s[2])), ("turnR_slow", (0, 0, -s[2])),
            ("sideL_slow", (0, s[1], 0)), ("sideR_slow", (0, -s[1], 0)), ("fwd_turnL_slow", (s[0], 0, s[2]))]


def run_base(split, b):
    """Continuous episode(s) for one (split, base) -> list of kept trials (step-resolution records)."""
    import torch
    torch.set_num_threads(1)
    from build_b1_cf_branches import Roll, POLICY_WARMUP, FIELDS
    si = list(SPLITS).index(split)
    need = SPLITS[split][0][b]
    base = np.array(bases()[b][1], float)
    R = cmd_range()
    kept, falls, restarts, attempt, ep_k = [], [], 0, 0, 0
    while len(kept) < need:
        r = Roll("gait3", GAINS, os.path.join(ROOT, MODEL_REL))
        for _ in range(POLICY_WARMUP + 37 * si + 13 * ep_k):   # extra steps: val / restarted episodes never
            r.step(*base)                                    # replay the train episode's deterministic start
        hist, since, prev_c = [], 0, None       # records with their next command; steps of base since start/return
        restart = False
        while len(kept) < need and not restart:
            # settle on base until SETTLE steps passed and a FR touchdown is seen
            while True:
                o = r.step(*base); o["step_i"] = r.step_i; since += 1
                hist.append(o); hist = hist[-(PRE + 1):]
                c = o["foot_contact"][0]
                td = prev_c is not None and prev_c == 0 and c > 0
                prev_c = c
                if fallen({k: np.asarray([o[k]]) for k in ("base_pos", "base_quat")})[0]:
                    restart = True; break
                if since >= SETTLE and td and len(hist) == PRE + 1:
                    break
            if restart:
                falls.append(dict(attempt=attempt, where="settle")); break
            rng = np.random.default_rng([CMD_SEED, si, b, attempt])
            fam = FAMS[int(rng.integers(3))]
            ax = FAM_AXIS[fam]
            lvl = float(R[ax, 0] + rng.random() * (R[ax, 1] - R[ax, 0]))
            cmd = np.zeros(3); cmd[ax] = lvl
            attempt += 1
            # hist[-1] = state at the switch frame (step k); window = hist[0] .. (record st = k - 50)
            recs = list(hist)
            nxt = [base] * PRE + [cmd]          # next command after record i (record PRE -> first new command)
            seg = [0] * PRE + [1]
            for i in range(CMD_STEPS):
                o = r.step(*cmd); o["step_i"] = r.step_i; recs.append(o)
                nxt.append(cmd if i < CMD_STEPS - 1 else base); seg.append(1 if i < CMD_STEPS - 1 else 2)
            for i in range(PER - PRE - 1 - CMD_STEPS + AFTER):
                o = r.step(*base); o["step_i"] = r.step_i; recs.append(o); nxt.append(base); seg.append(2)
            prev_c = recs[-1]["foot_contact"][0]
            since = PER - PRE - 1 - CMD_STEPS + AFTER
            hist = recs[-(PRE + 1):]
            T = {f: np.asarray([x[f] for x in recs], np.float32) for f in FIELDS}
            fl = fallen(T)
            if fl.any():
                falls.append(dict(attempt=attempt - 1, where="trial", family=fam, level=lvl, first=int(np.argmax(fl))))
                restart = True; break
            T["step_i"] = np.asarray([x["step_i"] for x in recs], np.int64)
            T["next_cmd"] = np.asarray(nxt, np.float32); T["next_seg"] = np.asarray(seg, np.int8)
            kept.append(dict(T=T, family=fam, level=lvl, cmd=cmd, attempt=attempt - 1, episode=ep_k))
        if restart:
            restarts += 1; ep_k += 1
    return kept, falls, restarts


def walk_path(split, b):
    return os.path.join(WALKS, f"{split}_b{b}.npz")


def do_physics(a):
    from wm.data.com import b1_com
    os.makedirs(WALKS, exist_ok=True)
    t0, log = time.time(), {}
    for split in SPLITS:
        for b in range(9):
            dst = walk_path(split, b)
            if os.path.exists(dst) and complete(dst):
                continue
            kept, falls, restarts = run_base(split, b)
            out = {}
            for f in kept[0]["T"]:
                out[f] = np.stack([k["T"][f] for k in kept])
            out["com_pos"] = np.stack([b1_com(k["T"]["base_pos"], k["T"]["base_quat"], k["T"]["joint_pos"],
                                              os.path.join(ROOT, MODEL_REL)) for k in kept]).astype(np.float32)
            out.update(family=np.array([k["family"] for k in kept]), level=np.array([k["level"] for k in kept]),
                       cmd=np.stack([k["cmd"] for k in kept]), attempt=np.array([k["attempt"] for k in kept]),
                       episode=np.array([k["episode"] for k in kept]), falls=np.array(json.dumps(falls)),
                       restarts=np.int64(restarts), base_id=np.int64(b), base_name=np.array(bases()[b][0]),
                       base_cmd=np.array(bases()[b][1], np.float64))
            atomic_savez(dst, **out)
            print(f"{split} base {b} {bases()[b][0]}: {len(kept)} trials, falls {len(falls)} "
                  f"({[f['where'] for f in falls]}), {time.time() - t0:.0f}s", flush=True)


def windows():
    """split -> [(base, trial, room_seed)], bases interleaved (trial-major) so room sizes spread over bases."""
    out = {}
    for split, (cnt, seeds) in SPLITS.items():
        W = [(b, t) for t in range(max(cnt)) for b in range(9) if t < cnt[b]]
        assert len(W) == len(seeds)
        out[split] = [(b, t, seeds[k]) for k, (b, t) in enumerate(W)]
    return out


def piece(W, t):
    idx = KEEP
    out = {k: W[k][t][idx].copy() for k in ("joint_pos", "joint_vel", "action", "command", "foot_contact", "base_pos",
                                            "base_quat", "com_pos")}
    off = -out["base_pos"][0, :2].astype(np.float64)
    out["base_pos"][:, :2] += off
    out["com_pos"][:, :2] += off
    out["segment"] = W["next_seg"][t][idx].astype(np.int8)
    out["cmd_plan"] = W["next_cmd"][t][idx]
    out["step_i"] = W["step_i"][t][idx]
    return out, off


_cache, _lock = {}, threading.Lock()


def walk(split, b):
    with _lock:
        if (split, b) not in _cache:
            with np.load(walk_path(split, b)) as z:
                _cache[(split, b)] = {k: z[k] for k in z.files}
        return _cache[(split, b)]


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
        for k, (b, t, seed) in enumerate(L):
            pc, _ = piece(walk(split, b), t)
            xy = pc["base_pos"][:, :2].astype(np.float64)
            pmin, pmax = xy.min(0) - CAM_BASE, xy.max(0) + CAM_BASE
            h = S[k] / 2
            lo, hi = -(h - margin) - pmin, (h - margin) - pmax
            u2 = np.random.default_rng([OFFSET_SEED, seed]).random(2)
            o = np.where(lo <= hi, lo + u2 * (hi - lo), (lo + hi) / 2)
            rooms[str(seed)] = dict(size=float(S[k]), offset=o.tolist(), shrunk=[bool(lo[i] > hi[i]) for i in range(2)],
                                    split=split, base=b, trial=t)
    tmp = ROOMS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(size_range=[lo_s, hi_s], size_seed=SIZE_SEED, offset_seed=OFFSET_SEED, margin=margin,
                       cam_base_bound=CAM_BASE, windows=W, bases=bases(),
                       rule="as collect_b1_babble.do_place (render_random_room.place): size stratified log-uniform per "
                            "split; offset uniform over the valid box, midpoint if an axis has none", rooms=rooms), f,
                  indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, ROOMS)
    print(f"{len(rooms)} rooms, shrunk {sum(any(r['shrunk']) for r in rooms.values())}")


def out_path(split, b, t):
    return os.path.join(CW, f"b1_sbabble_{split}", f"b1_sbabble_ep{code(split, b, t)}.npz")


def code(split, b, t):
    return 80000 + 1000 * list(SPLITS).index(split) + 100 * b + t


def do_launch(a):
    BB.CHK, BB.PIDS = CHK, PIDS
    for p in a.ports:
        assert p != 23000
    BB.do_launch(a)


def do_stop(a):
    BB.CHK, BB.PIDS = CHK, PIDS
    BB.do_stop(a)


def do_render(a):
    import render_shift_heldout as RS
    from render_b1_replay import CAM_POSE_CONVENTION
    RM = json.load(open(ROOMS))
    J = [(s, b, t, sd) for s, L in RM["windows"].items() for b, t, sd in L if not complete(out_path(s, b, t))]
    if a.limit:
        J = J[:a.limit]
    print(f"{len(J)} clips to render", flush=True)

    def fn(sim, job):
        split, b, t, seed = job
        W = walk(split, b)
        pc, off = piece(W, t)
        room = RM["rooms"][str(seed)]
        d = dict(pc, room_seed=np.int64(seed))
        fr, cp, R = RS.b1_render(sim, d, -np.asarray(room["offset"], float), RS.rs_room(room["size"]))
        assert fr.shape == (66, 256, 256, 3)
        h = room["size"] / 2
        clear = float((h - np.abs(cp[:, :2] + np.asarray(room["offset"]))).min())
        fam = str(W["family"][t])
        out = dict(pc)
        out.update(
            frames=fr, cam_pose=cp, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
            cam_pose_parent=np.array("base_visual"),
            joint_order_sdk=np.array([f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL") for s in ("hip", "thigh", "calf")]),
            dt=np.float64(0.05), fps=np.float64(20.0), froude_height=np.float64(FROUDE_H),
            condition=np.array("sbabble"), family=np.array("sbabble"), behaviour=np.array("sbabble"),
            level=np.int64(-1), family_level=np.int64(-1), cond_index=np.int64(-1), embodiment=np.array("b1"),
            expert_episode=np.int64(code(split, b, t)), policy=np.array("gait3"), room_seed=np.int64(seed),
            ego_seed=np.int64(seed), split=np.array(split), offset_xy=off,
            switch_frame=np.int64(SWITCH_FRAME), base_id=np.int64(b), base_name=np.array(bases()[b][0]),
            base_cmd=np.asarray(bases()[b][1], np.float64), cmd_family=np.array(fam),
            cmd_level=np.float64(W["level"][t]), trial_cmd=W["cmd"][t].astype(np.float64),
            trial_index=np.int64(t), trial_attempt=np.int64(W["attempt"][t]), sbabble_episode=np.int64(W["episode"][t]),
            switch_rule=np.array("FR foot touchdown (foot_contact[0] 0->1) after >= 150 base steps"),
            source_walk=np.array(os.path.relpath(walk_path(split, b), ROOT)),
            cmd_range=cmd_range(), cmd_range_source=np.array(CFG + " commands"), cmd_seed=np.int64(CMD_SEED),
            rollout_heading_kp=np.float64(GAINS[0]), rollout_heading_ki=np.float64(GAINS[1]),
            room_size=np.float64(room["size"]), rr_room_size=np.float64(room["size"]),
            rr_room_offset=np.asarray(room["offset"], np.float64), rr_room_size_original=np.float64(np.nan),
            rr_size_seed=np.int64(SIZE_SEED), rr_offset_seed=np.int64(OFFSET_SEED),
            rr_offset_shrunk=np.asarray(room["shrunk"], bool), rr_margin=np.float64(RM["margin"]),
            rr_room_height=np.float64(R["height"]), rr_ground_uv=np.float64(R["ground_uv"]),
            rr_cam_clearance=np.float64(clear), rr_source=np.array("sbabble (rendered directly)"),
            rr_render=np.array("render_shift_heldout.b1_render rs path (as rr_b1_*): room size rr_room_size, window start "
                               "at rr_room_offset from the room centre, heading kept, FOV 90"))
        dst = out_path(split, b, t)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        atomic_savez(dst, **out)
        return f"S {room['size']:.1f} clear {clear:.2f}"
    RS.run_pool(J, a.ports, fn, "sbabble")


def files(split):
    return sorted(glob.glob(os.path.join(CW, f"b1_sbabble_{split}", "*.npz")))


def subset_order():
    """Train clips in a fixed, base-balanced order: each base's trials permuted (seed 0), then round-robin over the
    bases with the base order of every round permuted (same rng)."""
    rng = np.random.default_rng(0)
    per = {b: list(rng.permutation(SPLITS["train"][0][b])) for b in range(9)}
    order = []
    for r in range(max(SPLITS["train"][0])):
        for b in rng.permutation(9):
            if r < len(per[b]):
                order.append((int(b), int(per[b][r])))
    return order


def do_subsets(a):
    order = subset_order()
    for n in (44, 88, 176):
        d = os.path.join(CW, f"b1_sbabble_n{n}")
        os.makedirs(d, exist_ok=True)
        for f in glob.glob(os.path.join(d, "*.npz")):
            os.remove(f)
        for b, t in order[:n + 4]:
            src = out_path("train", b, t)
            assert complete(src), src
            os.symlink(src, os.path.join(d, os.path.basename(src)))
        cnt = np.bincount([b for b, _ in order[:n + 4]], minlength=9)
        print(f"b1_sbabble_n{n}: {n + 4} links, per base {cnt.tolist()}")
    fd = os.open(CW, os.O_RDONLY); os.fsync(fd); os.close(fd)


def switch_state(d):
    """State at the switch frame: CoM velocity in the heading frame (fwd, lat m/s, central difference over +-1 frame),
    yaw rate (rad/s), clock gait phase (fraction of the 2 Hz cycle from the policy step counter), FR contact."""
    s = int(d["switch_frame"])
    q = d["base_quat"].astype(np.float64)
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    v = (d["com_pos"][s + 1, :2] - d["com_pos"][s - 1, :2]).astype(np.float64) / 0.1
    c, sn = np.cos(yaw[s]), np.sin(yaw[s])
    vf, vl = c * v[0] + sn * v[1], -sn * v[0] + c * v[1]
    wz = (yaw[s + 1] - yaw[s - 1]) / 0.1
    ph = (int(d["step_i"][s]) * 0.02 * 2.0) % 1.0
    return np.array([vf, vl, wz]), ph, d["joint_pos"][s].astype(np.float64)


class _D(dict):
    files = property(lambda self: list(self.keys()))


def do_check(a):
    import wm.data.embodiment as E
    rep, good = [], True

    def say(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); rep.append(s)
    os.makedirs(CHK, exist_ok=True)
    # physics bookkeeping
    say("Bases (vx, vy, wz):", "; ".join(f"{i} {n} {tuple(round(v, 3) for v in c)}" for i, (n, c) in enumerate(bases())))
    for s in SPLITS:
        tf, tr = 0, 0
        for b in range(9):
            with np.load(walk_path(s, b)) as z:
                tf += len(json.loads(str(z["falls"]))); tr += int(z["restarts"])
        say(f"{s}: dropped trials (falls) {tf}, episode restarts {tr}")
    seeds_all = {}
    for s in SPLITS:
        fs = files(s)
        say(f"{s}: {len(fs)} files (want {sum(SPLITS[s][0])}), tmp leftovers "
            f"{len(glob.glob(os.path.join(CW, f'b1_sbabble_{s}', '*.tmp*')))}")
        good &= len(fs) == sum(SPLITS[s][0])
        seeds_all[s] = []
        for p in fs:
            with np.load(p) as z:
                seeds_all[s].append(int(z["room_seed"]))
    exist = set()
    for p in glob.glob(os.path.join(CW, "*_clips_*", "*.npz")) + glob.glob(os.path.join(CW, "b1_babble_*", "*.npz")):
        if "sbabble" in p:
            continue
        try:
            with np.load(p) as z:
                exist.add(int(z["room_seed"]))
        except Exception:  # noqa: BLE001
            pass
    for s in SPLITS:
        ok = sorted(seeds_all[s]) == SPLITS[s][1]
        say(f"{s} room seeds {min(seeds_all[s])}-{max(seeds_all[s])} each once: {ok}; overlap with existing rooms: "
            f"{sorted(set(seeds_all[s]) & exist) or 'none'}")
        good &= ok and not (set(seeds_all[s]) & exist)
    # labels / segments
    rng = np.random.default_rng(0)
    nsw, bad, cross, fin, falls, clear, hs, fields_ok = 0, 0, 0, True, 0, [], set(), True
    steady = []
    SM, BMn, ST = [], [], {}
    need = ["segment", "switch_frame", "base_id", "cmd_family", "cmd_plan", "com_pos", "cam_pose", "rr_room_size",
            "rr_room_offset", "froude_height", "condition", "family", "behaviour"]
    for s in SPLITS:
        for p in files(s):
            with np.load(p, allow_pickle=True) as z:
                d = _D({k: z[k] for k in z.files if k != "frames"})
            fields_ok &= all(k in d for k in need)
            fields_ok &= str(d["condition"]) == str(d["family"]) == str(d["behaviour"]) == "sbabble"
            lab = E._b1(d)["body_motion"].astype(np.float64)
            fin &= bool(np.isfinite(lab).all())
            hs.add(float(d["froude_height"]))
            seg = d["segment"].astype(int)
            good &= seg.tolist() == [0] * 20 + [1] * 30 + [2] * 16
            for k in range(3):
                cp_ = d["cmd_plan"][seg == k]
                cross += int(np.abs(cp_ - cp_[0]).max() > 0)
            good &= np.allclose(d["cmd_plan"][0], d["base_cmd"]) and np.allclose(d["cmd_plan"][-1], d["base_cmd"])
            for b_ in (20, 50):
                nsw += 1
                e = _D(d); e["com_pos"] = d["com_pos"].copy(); e["base_quat"] = d["base_quat"].copy()
                e["com_pos"][:b_] += rng.normal(0, 0.3, e["com_pos"][:b_].shape)
                q = rng.normal(size=(b_, 4)); e["base_quat"][:b_] = (q / np.linalg.norm(q, axis=1, keepdims=True))
                l2 = E._b1(e)["body_motion"].astype(np.float64)
                end = 50 if b_ == 20 else 66
                bad += int(not np.array_equal(l2[b_:end], lab[b_:end]))
            q = d["base_quat"].astype(np.float64)
            falls += int(((d["base_pos"][:, 2] < 0.35) | (1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) < 0.7)).any())
            clear.append(float(d["rr_cam_clearance"]) - float(d["rr_margin"]))
            v, ph, jp = switch_state(d)
            b = int(d["base_id"])
            ST.setdefault(b, []).append(np.concatenate([v, [np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)], jp,
                                                        [d["foot_contact"][20].sum(), ph]]))
            if s == "train":
                SM.append(lab[23:47].mean(0)); BMn.append((b, lab[3:17].mean(0), lab[53:63].mean(0)))
            steady.append(np.abs(lab[:20].mean(0) - lab[3:17].mean(0)).max())
    say(f"fields present + condition=family=behaviour='sbabble': {fields_ok}; segments 20/30/16 frames each clip")
    say(f"labels finite: {fin}; froude_height {sorted(hs)}")
    say(f"segment switches checked {nsw}; labels after a switch independent of everything before it (pose noise "
        f"before the switch): {nsw - bad}/{nsw}; segments whose command changes inside: {cross}")
    say(f"clips with a fallen frame: {falls}; camera clearance beyond margin: min {min(clear):.2f} m "
        f"({sum(c < 0 for c in clear)} inside the margin)")
    good &= fields_ok and fin and bad == 0 and cross == 0 and falls == 0 and len(hs) == 1
    # near-same-state
    say("\nNear-same-state at the switch frame (all clips, train + val). Per base: std across its trials; "
        "'across bases' = std of the 9 base means. CoM vel in heading frame (m/s), yaw rate (rad/s), gait phase "
        "(clock, fraction of cycle), joint pos (rad, RMS over 12).")
    names = ["v_fwd", "v_lat", "yaw_rate"]
    within = {n: [] for n in names + ["phase", "joints"]}
    means = {n: [] for n in names + ["joints_vec"]}
    for b in range(9):
        X = np.array(ST[b])
        ph = X[:, -1]
        cs = np.angle(np.mean(np.exp(2j * np.pi * ph))) / (2 * np.pi) % 1
        phs = np.sqrt(max(0.0, -2 * np.log(min(1.0, np.abs(np.mean(np.exp(2j * np.pi * ph))))))) / (2 * np.pi)
        for k, n in enumerate(names):
            within[n].append(X[:, k].std()); means[n].append(X[:, k].mean())
        within["phase"].append(phs)
        jstd = np.sqrt((X[:, 5:17].std(0) ** 2).mean()); within["joints"].append(jstd)
        means["joints_vec"].append(X[:, 5:17].mean(0))
        say(f"  base {b} {bases()[b][0]:<15} n {len(X):2d}  v_fwd {X[:, 0].mean():+.3f}+-{X[:, 0].std():.3f}  "
            f"v_lat {X[:, 1].mean():+.3f}+-{X[:, 1].std():.3f}  yaw {X[:, 2].mean():+.3f}+-{X[:, 2].std():.3f}  "
            f"phase {cs:.3f}+-{phs:.3f}  joints rms-std {jstd:.3f}  FR..: contacts at switch {np.unique(X[:, -2]).tolist()}")
    say("  summary (mean within-base std | across-base std of means | ratio):")
    for n in names:
        w, ac = np.mean(within[n]), np.std(means[n])
        say(f"    {n:<9} {w:.3f} | {ac:.3f} | {w / max(ac, 1e-9):.2f}")
    jm = np.array(means["joints_vec"])
    acj = np.sqrt((jm.std(0) ** 2).mean())
    say(f"    joints    {np.mean(within['joints']):.3f} | {acj:.3f} | {np.mean(within['joints']) / acj:.2f}")
    say(f"    phase     {np.mean(within['phase']):.3f} cycles within (circular std); all bases switch at FR touchdown")
    say(f"  steadiness of the base before the switch: max |Froude mean frames 0-19 - frames 3-16| median "
        f"{np.median(steady):.3f}, max {np.max(steady):.3f}")
    # coverage
    SM = np.array(SM)
    B = []
    for p in sorted(glob.glob(os.path.join(CW, "b1_clips_*", "*.npz"))):
        with np.load(p) as z:
            B.append((int(z["cond_index"]), str(z["condition"]), E.load(p, E.B1)["body_motion"].mean(0)))
    beh = {}
    for i, n, m in B:
        beh.setdefault((i, n), []).append(m)
    beh = {k: np.mean(v, 0) for k, v in beh.items()}
    BM = np.array(list(beh.values()))
    BS = np.array([x[1] for x in BMn])
    say("\nCoverage (REPORTED only). Measured CoM Froude (fwd, lat, yaw), 1st..99th pct:")
    for nm, X in (("command-segment means", SM), ("base-segment means", BS), ("B1 clip behaviours (24)", BM)):
        say(f"  {nm:<24} fwd {np.percentile(X[:, 0], 1):+.3f}..{np.percentile(X[:, 0], 99):+.3f}  "
            f"lat {np.percentile(X[:, 1], 1):+.3f}..{np.percentile(X[:, 1], 99):+.3f}  "
            f"yaw {np.percentile(X[:, 2], 1):+.3f}..{np.percentile(X[:, 2], 99):+.3f}")
    ALL = np.concatenate([SM, BS])
    w = 0
    for (i, n), m in sorted(beh.items()):
        dd = np.linalg.norm(ALL - m, axis=1); w += int(dd.min() <= 0.05)
        say(f"   {i:2d} {n:<18} {np.round(m, 3)}  nearest segment mean {dd.min():.3f}")
    say(f"  B1 behaviours with a sbabble segment mean within 0.05: {w}/24")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax[k].scatter(SM[:, x], SM[:, y], s=8, alpha=0.6, c="tab:blue", label="command segments (train)")
        ax[k].scatter(BS[:, x], BS[:, y], s=8, alpha=0.6, c="tab:green", label="base segments (train)")
        ax[k].scatter(BM[:, x], BM[:, y], s=40, marker="x", c="tab:red", label="B1 clip behaviours (24)")
        nm = ("fwd", "lat", "yaw")
        ax[k].set_xlabel(f"{nm[x]} Froude"); ax[k].set_ylabel(f"{nm[y]} Froude"); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("B1 structured babbling: measured CoM Froude (reported only)")
    fig.tight_layout(); fig.savefig(os.path.join(CHK, "coverage.png"), dpi=110)
    # loader + pairs
    from wm.data.dataset import MultiEmbodimentPairs
    for s in ["train", "val", "n44"]:
        fs = files(s) if s != "n44" else sorted(glob.glob(os.path.join(CW, "b1_sbabble_n44", "*.npz")))
        ds = MultiEmbodimentPairs([(fs, "b1")], lazy_frames=True, frame_stride=5, rollout_k=2)
        x = ds[0]; _ = ds[len(ds) - 1]
        say(f"MultiEmbodimentPairs({s}: {len(fs)} files, lazy, stride 5, rollout 2): {len(ds)} pairs; keys {sorted(x)[:6]}...")
    say(f"\nALL CHECKS {'PASS' if good else 'FAIL'}")
    with open(os.path.join(CHK, "summary.txt"), "w") as f:
        f.write("\n".join(rep) + "\n"); f.flush(); os.fsync(f.fileno())


def do_review(a):
    from PIL import Image, ImageDraw
    import imageio.v2 as imageio
    import wm.data.embodiment as E
    pick = [out_path("train", b, t) for b, t in ((0, 0), (1, 1), (2, 2), (4, 0), (6, 3), (8, 1))]
    rows, vids = [], []
    for p in pick:
        with np.load(p) as z:
            fr, seg, cmd, S, b = z["frames"], z["segment"], z["cmd_plan"], float(z["room_size"]), int(z["base_id"])
            fam = str(z["cmd_family"])
        lab = E.load(p, E.B1)["body_motion"]
        row = Image.new("RGB", (160 * 6 + 260, 160))
        for k, t in enumerate((0, 19, 20, 35, 49, 65)):
            im = Image.fromarray(fr[t]).resize((160, 160)); dr = ImageDraw.Draw(im)
            dr.text((3, 3), f"t{t} seg{seg[t]}", fill=(255, 255, 0))
            dr.text((3, 146), f"{lab[t, 0]:+.2f} {lab[t, 1]:+.2f} {lab[t, 2]:+.2f}", fill=(255, 255, 0))
            row.paste(im, (260 + 160 * k, 0))
        dr = ImageDraw.Draw(row)
        dr.text((4, 20), os.path.basename(p), fill=(255, 255, 255))
        dr.text((4, 38), f"base {b} {bases()[b][0]}  room {S:.1f} m", fill=(255, 255, 255))
        for k in range(3):
            c = cmd[seg == k][0]
            dr.text((4, 62 + 18 * k), f"seg{k} vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}", fill=(180, 220, 255))
        dr.text((4, 120), f"family {fam}", fill=(180, 220, 255))
        rows.append(row)
        vids.append((fr, seg, lab, cmd, b, fam))
    sheet = Image.new("RGB", (rows[0].width, 160 * len(rows)))
    for k, r in enumerate(rows):
        sheet.paste(r, (0, 160 * k))
    sheet.save(os.path.join(CHK, "contact_sheet.png"))
    with imageio.get_writer(os.path.join(CHK, "sbabble_samples.mp4"), fps=10) as w:
        for t in range(66):
            tiles = []
            for fr, seg, lab, cmd, b, fam in vids:
                im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, 255, 40], fill=(0, 0, 0))
                dr.text((3, 2), f"base {b} {bases()[b][0]} -> {fam}", fill=(255, 255, 255))
                dr.text((3, 14), f"t{t} seg{seg[t]} cmd {cmd[t, 0]:+.2f} {cmd[t, 1]:+.2f} {cmd[t, 2]:+.2f}", fill=(180, 220, 255))
                dr.text((3, 26), f"Froude {lab[t, 0]:+.3f} {lab[t, 1]:+.3f} {lab[t, 2]:+.3f}", fill=(255, 255, 0))
                tiles.append(np.asarray(im))
            w.append_data(np.concatenate([np.concatenate(tiles[:3], 1), np.concatenate(tiles[3:6], 1)], 0))
    print("->", CHK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("physics", "place", "launch", "stop", "render", "subsets", "check", "review"))
    ap.add_argument("--ports", type=int, nargs="+", default=[25700])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    dict(physics=do_physics, place=do_place, launch=do_launch, stop=do_stop, render=do_render, subsets=do_subsets,
         check=do_check, review=do_review)[a.cmd](a)


if __name__ == "__main__":
    main()
