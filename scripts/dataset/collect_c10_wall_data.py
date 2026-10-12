"""c10 (six-legged hexapod) wall data -- the B1 design (`collect_b1_wall_data.py`, DATA_PLAN 13) on the hexapod.

Same rules as B1, imported from the B1 collector wherever the code is body-independent (per-frame SAFE / NEAR / CROSSED
labels, stratified train selection, report, stats, separability); hexapod-specific parts here:

**Matched to B1 (user 2026-10-12: "make the B1 and hex match").** Commands: the 24 beh24 conditions in B1's index order
(`beh24_conditions.ORDER`, Froude-matched pairwise) + 4 turn-on-the-spot recipes matched to B1's spins in Froude
(`hex_spin_pilot.py`: turn recipe, a0 0.05, 6 cycles, spin -0.67 / -0.95 / +0.50 / +0.80 = B1 wz +0.30 / +0.45 / -0.30 /
-0.45). Distances, bands and the margin in BODY LENGTHS: every B1 metre threshold x S = 0.82 / 1.13 (c10 footprint 0.82 x
0.46 m incl. legs vs B1 1.13 x 0.47); time in seconds as all data (3 s branches, 5 s in the far band; frames 0.05 s).
Room: the hexapod's original 8 m room (walls at +-3.975 m inner face).

**Physics exactly as F305 (deterministic scene reuse).** CoppeliaSim cannot restore a state, so: one scene load per
instance, 90 short warm-up runs, every run started identically (spawn (0, 0), heading as authored, 40 m empty room, no
camera), runs fall into bit-identical "classes"; a branch = the episode's commands replayed up to the branch frame T, then
one command (cross-fade over 4 frames, the `cpg_clock.Clock` rule) for 3 / 5 s; each branch is assigned to the class whose
episode replay matches its frames T-10..T-1 bit for bit; the episode's replay in the winning class is the stored episode.

**Virtual room per episode.** Because every physics run starts the same way, each episode's room is placed around the
path: a room centre c and rotation phi drawn so the hexapod starts facing a wall (approach angle as B1: 30-85 deg 75 % /
0-30 deg 25 %) or a corner (CORNER episodes, +-15 deg). The controller works in room coordinates; rendering transforms the
recorded link poses into the room (exact: the floor is flat) and builds the 8 m room at the origin.

**Episodes (one approach + one complete escape each, so that branches replay short prefixes):** approach at a random
forward level; branch states at clearance < d0 ~ U(1.2, 2.0) S, < d1 ~ U(0.8, 1.2) S, < d2 ~ U(0.4, 0.8) S (corner
episodes: only where the second wall is close too); at d2 the escape: back up / step aside until a spin on the spot is
safe (centre further than the spin radius + margin from every wall) -> spin by a random amount (small / medium / sharp,
along or away from the wall, > 4 S m free ahead) -> walk out 1-2.5 s. Decisions every 0.5 s from the hexapod's pose
(centre = mean of /head and /abdomen, heading = head - abdomen). No look-ahead (it would cost replays). The
approach heading is HELD with the gentlest library turn (level 1) when > 8 deg off: the open-loop gait drifts.

    PY=.venv/bin/python3; S=scripts/dataset/collect_c10_wall_data.py
    $PY $S physics --split pilot --n 3 --ep0 9000 --ports 25780 25782      # episodes + branches, CoppeliaSim
    $PY $S select --split train;  $PY $S report --splits ...;  $PY $S stats --split ...   (B1 code, hexapod geometry)
    $PY $S render --split pilot --ports ...
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts/dataset", "scripts/diagnostics/wall_pilot", "sim/control", "sim/collect", "sim/render", "."):
    sys.path.insert(0, os.path.join(ROOT, p))
import collect_b1_wall_data as W  # noqa: E402

# ---- hexapod geometry, swapped into the shared (B1) code ----
BL, BL_B1 = 0.82, 1.13
S = BL / BL_B1                                   # B1 metres -> c10 metres at equal body lengths
W.F.H, W.F.FOOT_L, W.F.FOOT_W = 3.975, 0.82, 0.46
W.MARGIN = W.MARGIN_BODY_LENGTHS * BL            # 0.13 body lengths = 0.107 m
W.DIST_SCALE = S                                 # report / stats bands and corner rule in body lengths
W.DATA = os.path.join(ROOT, "data/wall_c10")
W.CHK = os.path.join(ROOT, "results/check/wall_data_c10")
F = W.F
PREFIX = W.PREFIX
BRANCH_F, BRANCH_F_FAR, FAR_FROM = W.BRANCH_F, W.BRANCH_F_FAR, W.FAR_FROM * S
SCENE = "medauroidea_c10f10t10.ttt"
DRIVE_KW = dict(travel=0.0, warmup=20, cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), ego=True, ego_euler=None,
                ego_offset=None, ego_box=40.0, ego_seed=0, cam_fov=90.0, cmd_noise=0.0, capture_frames=False)
SHORT_WARM = 90


def recipes28():
    """28 hexapod recipes in B1's command order (index i = B1 command i), + the 4 matched spins"""
    from beh24_conditions import ORDER
    from collect_switch_hex import COND
    base = dict(COND["turn_s0.56"][1])
    spins = [dict(base, a0=0.05, spin=-0.67), dict(base, a0=0.05, spin=-0.95),     # spin left  ~ B1 wz +0.30 / +0.45
             dict(base, a0=0.05, spin=+0.50), dict(base, a0=0.05, spin=+0.80)]     # spin right ~ B1 wz -0.30 / -0.45
    return [dict(COND[n][1]) for n in ORDER] + spins


_, FAM = W.commands28()                          # family names, same order (fwd / bwd / turn_* / side_* / spin_*)
FWD, BWD, SL, SR, SPL, SPR = [0, 1, 2, 3], [4, 5, 6, 7], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25], [26, 27]


def rot(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


class Room:
    """world (physics) -> room frame: p_room = R(-phi) (p - c); heading_room = heading - phi"""

    def __init__(self, c, phi):
        self.c, self.phi = np.asarray(c, float), float(phi)

    def p(self, xy):
        return (rot(-self.phi) @ (np.asarray(xy, float)[..., :2] - self.c).T).T

    def yaw(self, y):
        return np.angle(np.exp(1j * (np.asarray(y) - self.phi)))


def pose(sim, H):
    """live controller pose: centre = mean(head, abdomen), heading = head - abdomen (world frame)"""
    h = np.array(sim.getObjectPosition(H["head"], sim.handle_world))
    a = np.array(sim.getObjectPosition(H["abd"], sim.handle_world))
    return (h[:2] + a[:2]) / 2, float(np.arctan2(h[1] - a[1], h[0] - a[0]))


def place_room(rng, p0, h0, corner):
    """room centre / rotation so the start pose faces a wall (or a corner) as the B1 episodes do (distances x S)"""
    if corner:
        d = rng.uniform(2.5, 4.0, 2) * S                                   # from both walls of corner (+H, +H)
        p_room = np.array([F.H - d[0], F.H - d[1]])
        tgt = np.array([F.H, F.H]) - rng.uniform(0.3, 1.0) * S
        head_room = float(np.arctan2(tgt[1] - p_room[1], tgt[0] - p_room[0]) + np.deg2rad(rng.uniform(-15, 15)))
    else:
        dist = rng.uniform(1.5, 4.0) * S                                   # from wall +x (normal +x)
        p_room = np.array([F.H - dist - F.FOOT_L / 2, rng.uniform(-F.H + 1.5, F.H - 1.5)])
        ang = rng.uniform(30, 85) if rng.random() < 0.75 else rng.uniform(0, 30)
        head_room = float(np.deg2rad(ang) * rng.choice([-1, 1]))
    phi = h0 - head_room
    c = np.asarray(p0) - rot(phi) @ p_room
    return Room(c, phi)


def episode_live(sim, R, rng, cmds_seed, corner, n_frames=600):
    """ONE live closed-loop episode (approach + complete escape), recipes chosen from the pose every 10 frames through
    `cpg_clock.Clock`. Returns (out, room, branch frames T, clock snapshots per frame, log)."""
    import collect_ik as CI
    from cpg_clock import Clock, centre_of
    recs = recipes28()
    centre = centre_of("c10f10t10")
    H = dict(head=sim.getObject("/head"), abd=sim.getObject("/abdomen"))
    st = dict(phase="approach", t=0, fwd=int(rng.choice(FWD)), d0=rng.uniform(1.2, 2.0) * S,
              d1=rng.uniform(0.8, 1.2) * S, d2=rng.uniform(0.4, 0.8) * S, marked=-1, room=None, target=None,
              walk_dur=float(rng.uniform(1.0, 2.5)), choice=None, done_at=None)
    clk = Clock(0.0, recs[st["fwd"]], centre)
    st["choice"] = st["fwd"]
    snaps, states, log = [], [], []
    spin_r = float(np.hypot(F.FOOT_L / 2, F.FOOT_W / 2)) + W.MARGIN

    def policy(observation, t):
        if t % 10 == 0:
            p_w, h_w = pose(sim, H)
            if st["room"] is None:
                st["room"] = place_room(rng, p_w, h_w, corner)
                st["aim"] = float(st["room"].yaw(h_w))           # approach heading to hold (room frame)
            rm = st["room"]; pos = rm.p(p_w); yw = float(rm.yaw(h_w))
            c = F.clearance(pos, yw, None)
            dws = sorted([F.H - pos[0], F.H + pos[0], F.H - pos[1], F.H + pos[1]])
            ok_state = (not corner) or W.is_corner(dws)
            ph = st["phase"]
            if ph == "approach":
                for key, nxt in (("d0", 0), ("d1", 1), ("d2", 2)):
                    if st["marked"] == nxt - 1 and c < st[key] and t >= PREFIX:
                        if ok_state and (not states or states[-1] != t):
                            states.append(t)
                        st["marked"] = nxt
                if st["marked"] == 2:
                    near = int(np.argmin([F.H - pos[0], F.H + pos[0], F.H - pos[1], F.H + pos[1]]))
                    away = (np.pi, 0.0, -np.pi / 2, np.pi / 2)[near]
                    cat = str(rng.choice(["small", "medium", "sharp"])); tgt = away
                    for cat_ in [cat] + [k for k in ("small", "medium", "sharp") if k != cat]:
                        lo, hi = {"small": (20, 60), "medium": (60, 120), "sharp": (120, 180)}[cat_]
                        ok = [yw + sg * np.deg2rad(a) for a in rng.uniform(lo, hi, 30) for sg in (-1, 1)
                              if W.free_dist(pos, yw + sg * np.deg2rad(a)) > 4.0 * S
                              and np.cos(yw + sg * np.deg2rad(a) - away) >= 0.0]
                        if ok:
                            tgt, cat = float(ok[int(rng.integers(len(ok)))]), cat_; break
                    st.update(phase="room", t=t, target=tgt, turn_cat=cat, wall=near)
            if st["phase"] == "room":
                rel = float(np.angle(np.exp(1j * (st["target"] - yw))))
                if min(dws) - 0.0 > spin_r + 0.05:                            # centre far enough: spin is safe
                    st.update(phase="turnout", t=t)
                else:                                                         # back up, or step to the open side
                    st["choice"] = int(rng.choice(BWD if abs(rel) > np.deg2rad(100) else (SL if rel > 0 else SR) + BWD))
            if st["phase"] == "turnout":
                rel = float(np.angle(np.exp(1j * (st["target"] - yw))))
                if abs(rel) < np.deg2rad(10) and F.front_dist(pos, yw) > 4.0 * S:
                    st.update(phase="walkout", t=t, walk0=t, c_walk0=c, wall_d0=dws[0]); st["choice"] = st["fwd"]
                else:
                    st["choice"] = int(rng.choice(SPL if rel > 0 else SPR))
            if st["phase"] == "walkout" and (t - st["walk0"]) * 0.05 >= st["walk_dur"] and st["done_at"] is None:
                st["done_at"] = t
            if st["phase"] == "approach":
                # STEER: the hexapod's open-loop gait drifts in heading (F75; measured +32 -> +7 deg over a 10 s corner
                # approach), so hold the approach heading with the gentlest library turn (level 1) when > 8 deg off
                err = float(np.angle(np.exp(1j * (st["aim"] - yw))))
                st["choice"] = (9 if err > 0 else 13) if abs(err) > np.deg2rad(8) else st["fwd"]
            clk.choose(recs[st["choice"]])
            log.append(dict(t=t, phase=st["phase"], cmd=st["choice"], clear=round(c, 3), pos=pos.round(3).tolist(),
                            yaw=round(yw, 3)))
        snaps.append(copy.deepcopy(clk))                    # clock BEFORE this frame's step: the branch start state
        return clk.step()

    out_st = {}
    f, a, fc, h, o = CI.drive_and_record(sim, SCENE, cmds_seed[:n_frames], state_out=out_st, state_from=0,
                                         reuse=R.reuse, policy=policy, **R.kw)
    R.n_runs += 1
    out = dict(actions=a, forces=fc, head=h, body_quat=o, frames=f, **out_st)
    return out, st, states, snaps, log


def branch_cmds(actions, snaps, recs, T, ci, nf):
    """episode commands for frames < T, then command ci from the episode's clock at T (4-frame cross-fade), nf frames"""
    clk = copy.deepcopy(snaps[T])
    clk.choose(recs[ci])
    tail = [clk.step() for _ in range(nf)]
    return np.concatenate([np.asarray(actions[:T], np.float32), np.asarray(tail, np.float32)])


def labels(out, room, a_from=0):
    """room-frame base_pos (CoM xy, z) + yaw quaternion (w, x, y, z) of heading head - CoM, per-frame clearance / front"""
    com = np.asarray(out["com_pos"], float); head = np.asarray(out["head"], float)[a_from:a_from + len(com)]
    xy = room.p(com[:, :2]); hd = room.p(head[:, :2])
    yaw = np.arctan2(hd[:, 1] - xy[:, 1], hd[:, 0] - xy[:, 0])
    quat = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1)
    clear = np.array([F.clearance(p, y, None) for p, y in zip(xy, yaw)])
    front = np.array([F.front_dist(p, y) for p, y in zip(xy, yaw)])
    return np.c_[xy, com[:, 2]], quat, clear, front


def save(path, out, room, meta, a_from=0, sl=None):
    """one file: hexapod physics fields (frames sl of the run) + room-frame labels (B1 field names) + room"""
    keep = ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat", "state_joint_pos",
            "state_link_pose", "state_sim_time", "com_pos")
    d = {}
    for k in keep:
        v = np.asarray(out[k])
        if k.startswith("state_") or k == "com_pos":
            d[k] = v if sl is None else v[sl.start - a_from:sl.stop - a_from]
        else:
            d[k] = v if sl is None else v[sl]
    bp, bq, cl, fr = labels(d, room, 0)
    crossed, kx, cb = W._terminal(cl)
    n = len(cl) if not crossed else kx + 1                                 # nothing after the crossing frame
    d = {k: v[:n] for k, v in d.items()}
    st = W.frame_status(cl[:n])
    tmp = path[:-4] + ".tmp.npz"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(tmp, **d, base_pos=bp[:n], base_quat=bq[:n], clearance=cl[:n], front_dist=fr[:n], status=st,
                        crossed=np.bool_(crossed), cross_frame=np.int64(kx), clear_before_cross=np.float64(cb),
                        first_near_frame=np.int64(np.flatnonzero(st >= 1)[0] if (st >= 1).any() else -1),
                        margin=np.float64(W.MARGIN), margin_body_lengths=np.float64(W.MARGIN_BODY_LENGTHS),
                        state_link_names=out["state_link_names"], state_joint_names=out["state_joint_names"],
                        room_c=room.c, room_phi=np.float64(room.phi), boundary_half=np.float64(F.H),
                        foot_lw=np.array([F.FOOT_L, F.FOOT_W]), dt=np.float64(0.05), fps=np.float64(20.0),
                        embodiment=np.array("hexapod"), morph=np.array("c10f10t10"), **meta)
    os.replace(tmp, path)
    return crossed, n


def worker(port, jobs, split, log):
    """one CoppeliaSim instance: warm up once, then each episode: live run, replay class reference, all branches"""
    import collect_ik as CI
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from scene_reuse import SceneReuse, state_hash
    from cpg_clock import Clock, centre_of
    sim = RemoteAPIClient("localhost", port=port).require("sim")
    R = SceneReuse(sim, SCENE, DRIVE_KW)
    recs = recipes28(); centre = centre_of("c10f10t10")
    clk = Clock(0.0, recs[0], centre); seed = np.array([clk.step() for _ in range(600)], np.float32)
    t0 = time.time()
    for _ in range(SHORT_WARM):
        R.run(seed[:30])
    log(f"port {port}: {SHORT_WARM} warm-up runs in {time.time() - t0:.0f} s")
    out_dir = os.path.join(W.DATA, "_physics", split)
    for ep, look, corner in jobs:
        if os.path.exists(os.path.join(out_dir, f"ep{ep:05d}.npz")):
            continue
        rng = np.random.default_rng(ep)
        t0 = time.time()
        live, st, states, snaps, elog = episode_live(sim, R, rng, seed, corner)
        n_ep = (st["done_at"] + 1) if st["done_at"] is not None else len(live["actions"])
        A = np.asarray(live["actions"][:n_ep], np.float32)
        room = st["room"]
        refs = {}                                            # class hash -> episode replay

        def new_ref():
            o = R.run(A)
            refs[state_hash(o)] = o
            return o
        new_ref(); new_ref()
        if not states:                                       # no branch state: keep the episode + its log, say why
            save(os.path.join(out_dir, f"ep{ep:05d}.npz"), refs[next(iter(refs))], room,
                 dict(episode=np.int64(ep), split=np.array(split), room_seed=np.int64(look),
                      style=np.array("corner" if corner else "approach"), kind=np.array("episode"),
                      log=np.array(json.dumps(elog)), escape_done=np.int64(st["done_at"] or -1)))
            log(f"ep{ep}: NO branch state; phases {[e['phase'] for e in elog][::4]}; clearances "
                f"{[e['clear'] for e in elog][::4]}")
            continue
        bl, nf_of = {}, {}
        for b, T in enumerate(states):
            nf = BRANCH_F_FAR if labels({"com_pos": live["com_pos"][T - 1:T], "head": live["head"][T - 1:T]},
                                        room)[2][0] >= FAR_FROM else BRANCH_F
            nf_of[b] = nf
            for ci in range(28):
                cm = branch_cmds(A, snaps, recs, T, ci, nf)
                for attempt in range(4):
                    o = R.run(cm, state_from=T - PREFIX)
                    hit = [k for k, ref in refs.items() if R.same_as(o, ref, range(T - PREFIX, T), a_from=T - PREFIX)]
                    if hit:
                        bl.setdefault(hit[0], {})[(b, ci)] = (o, nf); break
                    new_ref()
        need = {(b, ci) for b in range(len(states)) for ci in range(28)}
        win = next((k for k in refs if need <= set(bl.get(k, {}))), None)
        if win is None:                                    # complete the best class (F305 rule)
            win = max(refs, key=lambda k: len(bl.get(k, {})))
            for (b, ci) in sorted(need - set(bl.get(win, {}))):
                T = states[b]; nf = nf_of[b]
                for attempt in range(12):
                    o = R.run(branch_cmds(A, snaps, recs, T, ci, nf), state_from=T - PREFIX)
                    if R.same_as(o, refs[win], range(T - PREFIX, T), a_from=T - PREFIX):
                        bl.setdefault(win, {})[(b, ci)] = (o, nf); break
            if not need <= set(bl[win]):
                log(f"ep{ep}: could not complete one class -- episode skipped"); continue
        ref = refs[win]
        names = W.cmd_names()
        meta = dict(episode=np.int64(ep), split=np.array(split), room_seed=np.int64(look),
                    style=np.array("corner" if corner else "approach"), det_class=np.array(win),
                    det_classes_seen=np.int64(len(refs)), det_runs=np.int64(R.n_runs))
        for (b, ci), (o, nf) in sorted(bl[win].items()):
            T = states[b]
            com_T = labels({"com_pos": ref["com_pos"][T - 1:T], "head": ref["head"][T - 1:T]}, room)[2][0]
            save(os.path.join(out_dir, f"ep{ep:05d}_b{b}_c{ci:02d}.npz"), o, room,
                 dict(meta, kind=np.array("branch"), branch_state=np.int64(b), branch_frame=np.int64(T),
                      first_pair=np.int64(PREFIX), cf_command_index=np.int64(ci), cf_command_name=np.array(names[ci]),
                      start_clearance=np.float64(com_T), branch_seconds=np.float64(nf * 0.05),
                      segment=np.array([0] * PREFIX + [1] * nf, np.int8),
                      froude_height=np.float64(np.median(np.asarray(ref["com_pos"])[:, 2]))),
                 a_from=T - PREFIX, sl=slice(T - PREFIX, T + nf))
        seg = np.cumsum(np.r_[0, np.diff([e["cmd"] for e in elog]) != 0])
        crossed, n = save(os.path.join(out_dir, f"ep{ep:05d}.npz"), ref, room,
                          dict(meta, kind=np.array("episode"), log=np.array(json.dumps(elog)),
                               turn_cat=np.array(st.get("turn_cat", "")), escape_done=np.int64(st["done_at"] or -1)))
        log(f"ep{ep} ({'corner' if corner else 'approach'}): {n_ep} frames, {len(states)} branch states, classes "
            f"{len(refs)}, crossed {crossed}, escape done at {st['done_at']}, {time.time() - t0:.0f} s, runs {R.n_runs}")


def do_physics(a):
    import threading
    jobs = [(a.ep0 + i, a.look0 + i, a.corner) for i in range(a.n)]
    per = [jobs[k::len(a.ports)] for k in range(len(a.ports))]
    lock = threading.Lock()

    def log(m):
        with lock:
            print(m, flush=True)
    th = [threading.Thread(target=worker, args=(p, j, a.split, log)) for p, j in zip(a.ports, per) if j]
    [t.start() for t in th]; [t.join() for t in th]


def do_render(a):
    """ego frames in the 8 m room at the room origin: every stored file's link poses transformed into its room frame"""
    import glob
    import render_hex_replay as RH
    from render_shift_heldout import run_pool
    J = sorted(glob.glob(os.path.join(W.DATA, "_physics", a.split, "*.npz")))
    sel_f = os.path.join(W.DATA, "_physics", a.split, "selection.json")
    if os.path.exists(sel_f):
        keep = set(json.load(open(sel_f))["render"])
        J = [p for p in J if "_b" not in os.path.basename(p) or os.path.basename(p) in keep]
    J = [p for p in J if not os.path.exists(os.path.join(W.DATA, a.split, os.path.basename(p)))]
    print(f"{len(J)} files to render", flush=True)

    def fn(sim, p):
        d = dict(np.load(p, allow_pickle=True))
        phi, c = float(d["room_phi"]), np.asarray(d["room_c"], float)
        Rm = rot(-phi)
        lp = np.asarray(d["state_link_pose"], float).copy()
        lp[:, :, :2] = (Rm @ (lp[:, :, :2] - c).reshape(-1, 2).T).T.reshape(lp.shape[0], lp.shape[1], 2)
        qz = np.array([0.0, 0.0, np.sin(-phi / 2), np.cos(-phi / 2)])            # rotation about world z, (x, y, z, w)
        q = lp[:, :, 3:7]
        x1, y1, z1, w1 = qz; x2, y2, z2, w2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        lp[:, :, 3:7] = np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)
        head = np.asarray(d["head"], float).copy(); head[:, :2] = (Rm @ (head[:, :2] - c).T).T
        rec = dict(state_link_pose=lp, state_link_names=d["state_link_names"], head=head,
                   state_abdomen_pos=d["state_abdomen_pos"], state_abdomen_quat=d["state_abdomen_quat"],
                   state_joint_pos=d["state_joint_pos"], state_joint_names=d["state_joint_names"])
        fr, off, R_, cam = RH.render(sim, rec, 0, len(lp), int(d["room_seed"]), recentre=False, room=8.0, mode="links",
                                     return_cam=True, offset_xy=np.zeros(2))
        dst = os.path.join(W.DATA, a.split, os.path.basename(p))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **d, frames=fr, cam_pose=cam)
        os.replace(tmp, dst)
        return f"clear {d['clearance'].min():+.2f}"
    run_pool(J, a.ports, fn, "render")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("physics", "render", "select", "report", "stats", "sep"))
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ep0", type=int, default=9000)
    ap.add_argument("--look0", type=int, default=9000)
    ap.add_argument("--corner", action="store_true")
    ap.add_argument("--ports", type=int, nargs="+", default=[25780])
    ap.add_argument("--h", type=float, default=1.25)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    {"physics": do_physics, "render": do_render, "select": W.do_select, "report": W.do_report, "stats": W.do_stats,
     "sep": W.do_sep}[a.cmd](a)
