"""B1 wall data (DATA_PLAN 12, revised 2026-10-10) -- PILOT for review before any collection.

World: the ORIGINAL B1 room (17.65 m; walls at +-H around the world origin, NOT re-centred); MuJoCo has no walls ("virtual
boundary"); the footprint (1.13 x 0.47 m) clearance is computed per frame; nothing after a boundary crossing is kept (the camera would be
inside the wall).
A. Episodes: start 1.5-5 m from a wall heading toward it (+-60 deg); walk a random forward level; every 0.5 s the forward
   command is simulated 2 s ahead in MuJoCo (exact state); if it would come within 0.3 m of the boundary, evade: all 24
   commands 2 s ahead, drop those reaching < 0.15 m, one of the 3 leaving the most free space (front <= 4 m + clearance;
   backing up penalised -- it keeps the robot facing the wall), held 1 s, until > 3 m ahead is free. ~0.3 random segments
   per second (any command, held 0.5-1.5 s) so unsafe outcomes appear too. Ground-truth look-ahead steers data collection
   only. (Rule-based versions got stuck in corners / turned the 1 m body into the wall: results/check/wall_data_pilot/v1-v3.)
B. Branches: from states sampled along the episodes near a wall (clearance 0.5-4 m), the exact MuJoCo state is restored and
   EVERY command is run for 1.5 s (30 frames; 10 shared frames before) -- the counterfactual part.
Rendering: one random room look per episode (seed 5000 + episode; its branches share it); frames 0.05 s.

    .venv/bin/python3 scripts/dataset/collect_b1_wall_data.py pilot --port P     # -> results/check/wall_data_pilot/
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts/diagnostics/wall_pilot", "scripts/dataset", "sim/control", "sim/collect", "."):
    sys.path.insert(0, os.path.join(ROOT, p))
import b1_wall_feasibility as F  # noqa: E402
from build_b1_cf_branches import FIELDS  # noqa: E402

# B1 standing footprint (Unitree spec ~1.126 x 0.467 m; was 1.0 x 0.5 in the feasibility pilot), set before any clearance
F.FOOT_L, F.FOOT_W = 1.13, 0.47

CHK = os.path.join(ROOT, "results/check/wall_data_pilot")
# B1 wall library = the 24 tuned commands + 4 TURN-ON-THE-SPOT commands (2026-10-11, agreed with the user): B1's policy
# was trained on wz in [-0.6, 0.6] with any vx and spins cleanly (wz 0.5: 27.7 deg/s, drift 0.07 m/s, upright); the
# library's own turns walk forward at ~0.36 m/s and turn <= 14 deg/s, so a robot facing a wall could not turn away.
# Froude (forward / lateral / yaw): wz +0.30 -> +0.005 / -0.026 / +0.075, wz +0.45 -> +0.007 / -0.028 / +0.112 (and
# mirrored); the hexapod matches them with the turn recipe at a0 0.05, 6 cycles, spin -0.67 / -0.95 / +0.50 / +0.80
# (`scripts/diagnostics/wall_pilot/hex_spin_pilot.py`, results/check/hex_spin_pilot/summary.txt).
SPINS = (((0.0, 0.0, 0.30), "spin_left", "spin_left_w0.30"), ((0.0, 0.0, 0.45), "spin_left", "spin_left_w0.45"),
         ((0.0, 0.0, -0.30), "spin_right", "spin_right_w0.30"), ((0.0, 0.0, -0.45), "spin_right", "spin_right_w0.45"))


def commands28():
    """(commands, families) of the B1 wall library: 24 tuned + 4 spins (indices 24-27)"""
    cmds, fam = F.commands()
    return list(cmds) + [np.asarray(c, np.float64) for c, _, _ in SPINS], list(fam) + [f for _, f, _ in SPINS]


def cmd_names():
    _, fam = F.commands()
    return [f"{fam[i]}{i % 4}" for i in range(24)] + [n for _, _, n in SPINS]


OFFS_FRAME = lambda nf: [int(v) for v in np.round(2.5 * np.arange(1, nf + 1))]  # noqa: E731  frame j = policy step 2.5 j


def lookahead(r, cmd, secs=2.0):
    """run `cmd` for `secs` from the exact current state (restored after); returns (min clearance, end pose)"""
    snap = r.snapshot(); mn = 1e9
    for k in range(int(secs * 50)):
        r.step(*cmd)
        if k % 5 == 4:
            mn = min(mn, F.clearance(r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7]), None))
    pos, yw = r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7])
    r.restore(snap)
    return mn, pos, yw


def safe_pick(r, cmds, rng, secs=2.0, no_back=False):
    """all 24 commands 2 s ahead; drop those reaching < 0.15 m; one of the 3 with the most free space after
    (front distance capped 4 m + clearance). Ground-truth look-ahead: steers DATA COLLECTION only."""
    _, fam = commands28()
    sc = []
    for ci, c in enumerate(cmds):
        mn, pos, yw = lookahead(r, c, secs)
        v = -9.0 if mn < 0.15 else min(F.front_dist(pos, yw), 4.0) + F.clearance(pos, yw, None)
        sc.append(v - ((20.0 if no_back else 1.0) if fam[ci] == "bwd" else 0.0))     # backing up only when nothing else is safe: it
    return int(rng.choice(np.argsort(sc)[-3:]))                 # keeps the robot facing the same wall


def episode(seed, n_frames=400, style=None):
    """reactive episode (safety look-ahead controller); returns per-frame records (0.05 s) and per-frame snapshots.
    Walk the episode's forward goal; every 0.5 s check the current command 2 s ahead in physics; if it would come
    within 0.3 m of the boundary, pick a safe command (`safe_pick`) and hold it 1 s; ~0.3 random segments / s
    (held 0.5-1.5 s, any command, also unsafe ones -- the model must see bad outcomes too)."""
    cmds, fam = F.commands()
    rng = np.random.default_rng(seed)
    goal_i = int(rng.choice(F.FWD))
    # controller style per episode (2026-10-11, agreed with the user): 40 % early avoider (looks 3 s ahead, evades when the
    # forward walk would come within 0.8-1.5 m, never backs up unless nothing else is safe), 40 % late avoider (1 s ahead,
    # 0.2-0.5 m, may back up), 20 % no avoidance (walks into the wall; cut at contact) -- so the data is not one habit
    if style is None:
        style = str(rng.choice(["early", "late", "none"], p=[0.4, 0.4, 0.2]))
    secs, trig = (3.0, rng.uniform(0.8, 1.5)) if style == "early" else (1.0, rng.uniform(0.2, 0.5))
    # start 1.5-5 m from one wall (random side), heading toward it within +-60 deg, so walls are met within the episode
    wall = int(rng.integers(4)); dist = rng.uniform(1.5, 5.0); along = rng.uniform(-F.H + 2.0, F.H - 2.0)
    nrm = [(1, 0), (-1, 0), (0, 1), (0, -1)][wall]
    x, y = (np.array(nrm) * (F.H - dist - F.FOOT_L / 2) + np.array(nrm[::-1]) * along)
    yaw = np.arctan2(nrm[1], nrm[0]) + np.deg2rad(rng.uniform(-60, 60))
    r, snap = F.start_state(None, cmds, goal_i, x, y, yaw, int(rng.integers(0, 12)))
    r.restore(snap)
    rec, snaps, choice = [], [], goal_i
    step, hold_until, evading = 0, -1, False
    for j in range(n_frames):
        if j % 10 == 0 and j >= hold_until:                     # decision every 0.5 s unless holding
            if rng.random() < 0.15:                             # random segment (~0.3 / s), held 0.5-1.5 s
                choice = int(rng.integers(len(cmds))); hold_until = j + int(rng.integers(10, 31))
            elif style != "none" and (evading or lookahead(r, cmds[goal_i], secs)[0] < trig):  # evade until > 3 m free
                pos, yw = r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7])
                evading = F.front_dist(pos, yw) < 3.0
                choice = safe_pick(r, cmds, rng, secs, style == "early") if evading else goal_i
                hold_until = j + 20 if evading else hold_until
            else:
                choice = goal_i
        snaps.append(r.snapshot())
        tgt = int(round(2.5 * (j + 1)))
        while step < tgt:
            last = r.step(*cmds[choice]); step += 1
        pos, yw = r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7])
        c = F.clearance(pos, yw, None)
        rec.append(dict({f: np.asarray(last[f]).copy() for f in FIELDS}, clearance=c, front=F.front_dist(pos, yw),
                        cmd=choice))
        if c < 0:
            break
    return r, rec, snaps, style


def branch(r, snap, cmd, nf=30):
    from build_b1_cf_branches import run_branch
    br, fell = run_branch(r, snap, cmd, OFFS_FRAME(nf))
    out = []
    for k in range(nf):
        pos, q = br["base_pos"][k], br["base_quat"][k]
        yw = F.yaw_of(q)
        c = F.clearance(pos[:2], yw, None)
        out.append(dict(base_pos=pos, base_quat=q, joint_pos=br["joint_pos"][k], clearance=c))
        if c < 0:                                   # cut at the first boundary crossing
            break
    return out


def render(sim, look, recs):
    import render_shift_heldout as RS
    d = dict(base_pos=np.array([x["base_pos"] for x in recs], np.float32),
             base_quat=np.array([x["base_quat"] for x in recs], np.float32),
             joint_pos=np.array([x["joint_pos"] for x in recs], np.float32), room_seed=np.int64(look))
    fr, cp, R = RS.b1_render(sim, d, np.zeros(2))
    return fr


def topdown(recs_list, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((-F.H, -F.H), 2 * F.H, 2 * F.H, fill=False, lw=2))
    for recs, lab in zip(recs_list, labels):
        p = np.array([x["base_pos"][:2] for x in recs])
        ax.plot(p[:, 0], p[:, 1], lw=1.2, label=lab)
        ax.plot(p[0, 0], p[0, 1], "ko", ms=3)
    ax.set_aspect("equal"); ax.legend(fontsize=6, loc="upper right"); ax.set_title("top-down (black square = walls)")
    fig.savefig(path, dpi=110); plt.close(fig)


def do_pilot(a):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    os.makedirs(CHK, exist_ok=True)
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    cmds, fam = F.commands()
    names = [f"{fam[i]}{i % 4}" for i in range(24)]
    eps = []
    for e in range(3):
        r, rec, snaps, sty = episode(1000 + e, style=("early", "late", "none")[e])
        eps.append((r, rec, snaps))
        fr = render(sim, 5000 + e, rec)
        wr = imageio.get_writer(os.path.join(CHK, f"episode_{e}.mp4"), fps=20)
        for j, f in enumerate(fr):
            im = Image.fromarray(f).resize((384, 384)); dr = ImageDraw.Draw(im)
            dr.text((5, 5), f"ep{e} {sty} t={j * 0.05:.2f}s cmd {names[rec[j]['cmd']]} clear {rec[j]['clearance']:+.2f} m",
                    fill=(255, 255, 0))
            wr.append_data(np.asarray(im))
        wr.close()
        print(f"episode {e} ({sty}): {len(rec)} frames, min clearance {min(x['clearance'] for x in rec):+.2f}, "
              f"commands used {sorted(set(names[x['cmd']] for x in rec))}", flush=True)
    topdown([x[1] for x in eps], [f"episode {e}" for e in range(3)], os.path.join(CHK, "episodes_topdown.png"))
    # branches: the first frame of each episode whose clearance is within 0.6-2.0 m and that has >= 10 frames before
    show = [0, 4, 8, 11, 12, 15, 16, 18, 20, 22, 2, 3]       # fwd / bwd / turns / side-steps
    for e, (r, rec, snaps) in enumerate(eps[:2]):
        want = (0.8, 1.5)[e]                                  # one close, one mid-distance branch state
        j0 = 10 + int(np.argmin([abs(x["clearance"] - want) for x in rec[10:-1]]))
        rows, outs = [], []
        for ci in show:
            out = branch(r, snaps[j0], cmds[ci])
            outs.append(out)
            fr = render(sim, 5000 + e, rec[j0 - 10:j0 + 1] + out)
            tiles = []
            for k in (10, 20, 30, len(fr) - 1):
                k = min(k, len(fr) - 1)
                cl = (rec[j0 - 10:j0 + 1] + out)[k]["clearance"]
                im = Image.fromarray(fr[k]).resize((128, 128)); dr = ImageDraw.Draw(im)
                dr.text((3, 3), f"{names[ci]} {(k - 10) * 0.05:+.1f}s {cl:+.2f}", fill=(255, 255, 0))
                tiles.append(np.asarray(im))
            rows.append(np.hstack(tiles))
        half = len(rows) // 2
        sheet = np.hstack([np.vstack(rows[:half]), np.vstack(rows[half:])])
        Image.fromarray(sheet).save(os.path.join(CHK, f"branches_state{e}.png"))
        topdown([rec[j0 - 10:j0 + 1] + o for o in outs], [names[c] for c in show],
                os.path.join(CHK, f"branches_state{e}_topdown.png"))
        print(f"branch state {e}: episode frame {j0}, clearance {rec[j0]['clearance']:+.2f} m; futures: " +
              ", ".join(f"{names[c]} min {min(x['clearance'] for x in o):+.2f}{' CUT' if o[-1]['clearance'] < 0 else ''}"
                        for c, o in zip(show, outs)), flush=True)
    json.dump(dict(note="pilot only"), open(os.path.join(CHK, "pilot.json"), "w"))


def free_dist(pos, ang):
    """distance from the footprint centre along world direction `ang` to the boundary, minus the body half-length"""
    c, s_ = np.cos(ang), np.sin(ang); t = []
    for v, d in ((pos[0], c), (pos[1], s_)):
        if abs(d) > 1e-9:
            t.append(((F.H if d > 0 else -F.H) - v) / d)
    return min(t) - F.FOOT_L / 2


def gaps(pos, min_free=2.0, n=24):
    """GAPS (gap following): runs of directions (every 360/n deg) with > min_free m free; returns the world angle of
    each gap's centre (all gaps, not only the biggest)"""
    a = np.arange(n) * 2 * np.pi / n
    ok = np.array([free_dist(pos, x) > min_free for x in a])
    if ok.all():
        return list(a)
    out, k0 = [], int(np.argmin(ok))                             # start the scan at a blocked direction
    run = []
    for k in range(k0, k0 + n + 1):
        if ok[k % n]:
            run.append(k)
        elif run:
            out.append(a[run[len(run) // 2] % n]); run = []
    return out


def gap_command(r, cmds, rng, pos, yw):
    """escape toward a RANDOM gap: relative direction -> command family (behind: back up; left: side-step or turn left;
    right: side-step or turn right; front: turn toward it), level random; each candidate checked 1 s ahead in physics
    (> 0.15 m). Falls back to the safest back-up."""
    G = gaps(pos)
    rng.shuffle(G)
    for g in G:
        rel = float(np.angle(np.exp(1j * (g - yw))))
        if abs(rel) > np.deg2rad(135):
            fams = [[4, 5, 6, 7]]
        elif rel > np.deg2rad(45):
            fams = [[16, 17, 18, 19], [8, 9, 10, 11]]
        elif rel < -np.deg2rad(45):
            fams = [[20, 21, 22, 23], [12, 13, 14, 15]]
        else:
            fams = [[8, 9, 10, 11] if rel > 0 else [12, 13, 14, 15]]
        rng.shuffle(fams)
        for fm in fams:
            for ci in rng.permutation(fm):
                if lookahead(r, cmds[ci], 1.0)[0] > 0.15:
                    return int(ci)
    return int(max([4, 5, 6, 7], key=lambda c: lookahead(r, cmds[c], 1.0)[0]))


def wall_dist(pos, w):
    """distance of the base from wall w (0: +x, 1: -x, 2: +y, 3: -y)"""
    return (F.H - pos[0], F.H + pos[0], F.H - pos[1], F.H + pos[1])[w]


def episode_approach(seed, n_frames=600):
    """APPROACH-ESCAPE episodes (2026-10-11, agreed with the user): approach a wall, then a COMPLETE escape:
    back up / side-step to make room -> turn toward an open direction -> walk out into the room -> next wall.
    Phases:
    ORIENT   turn toward the target (a wall at 30-85 deg from its normal 75 % / 0-30 deg 25 %, or 30 % a corner region; never the wall just
             escaped; next wall preferably 2-6 m away); each turn command checked 1 s ahead (B1 turns walk forward) -- unsafe -> escape.
    APPROACH random forward level; branch states when clearance first < d0 ~ U(1.2, 2.0) (far band, 2026-10-11), < d1 ~
             U(0.8, 1.2) and < d2 ~ U(0.4, 0.8); at d2
             the escape starts. Episodes contain ONLY escapes (user 2026-10-11); crossings are in the branches.
    ESCAPE   target heading = current heading turned by a random amount (small 20-60 / medium 60-120 / sharp
             120-180 deg) toward a side with 4-10 m free and pointing along or away from the escaped wall (a nearby wall: after the walk-out the robot keeps walking into it, no
             crossing of the room; 2026-10-11: turns were always 70-160 deg and 8.5 s passed between escapes before).
      room     back up / side-step toward the target side (each checked 1 s ahead) until a SPIN toward the target
               is 1 s-safe (> 0.2 m);
      turnout  turn ON THE SPOT (spin commands 24-27, random rate) toward the target (checked every 0.5 s; unsafe -> room) until FACING OPEN SPACE = within
               10 deg of the target heading and > 4 m free ahead;
      walkout  forward for U(1, 2.5) s (checked every 0.5 s; unsafe -> room), then the next approach.
    Every escape is logged: start frame, crossed, frame facing open space, walk-out completed, moved away from the
    escaped wall (> 0.1 m further at the end of the walk-out than at its start), end frame; capped at 12 s."""
    cmds, fam = commands28()
    rng = np.random.default_rng(seed)
    w0 = int(rng.integers(4)); nrm0 = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)][w0])   # start 1.5-4 m from a wall
    pos0 = nrm0 * (F.H - rng.uniform(1.5, 4.0) - F.FOOT_L / 2) + nrm0[::-1] * rng.uniform(-F.H + 2.0, F.H - 2.0)
    yaw0 = np.arctan2(nrm0[1], nrm0[0]) + np.deg2rad(rng.uniform(-60, 60))   # facing that wall: first approach at once
    r, snap = F.start_state(None, cmds, int(rng.choice(F.FWD)), pos0[0], pos0[1], yaw0, int(rng.integers(0, 12)))
    r.restore(snap)
    rec, snaps, states, escapes = [], [], [], []
    st = dict(phase="orient", t=0, des=yaw0, d1=0.0, d2=0.0, marked=0, straight=False, fwd=0, n_app=0, last_wall=-1)
    choice, step = 0, 0
    BWD, SL, SR, TL, TR = [4, 5, 6, 7], [16, 17, 18, 19], [20, 21, 22, 23], [10, 11], [14, 15]
    SPL, SPR = [24, 25], [26, 27]                              # turn on the spot (escape turn-out)

    def new_approach(first=False):
        p_ = r.d.qpos[0:2]
        walls = [w for w in range(4) if w != st["last_wall"]]
        near = [w for w in walls if 2.0 <= wall_dist(p_, w) <= 6.0]      # a wall 2-6 m away: short approach
        w = int(rng.choice(near)) if near else min(walls, key=lambda w_: wall_dist(p_, w_))
        nrm = [(1, 0), (-1, 0), (0, 1), (0, -1)][w]
        if first:
            st["des"] = yaw0
        elif rng.random() < 0.3:                                 # a corner region next to that wall
            sx = nrm[0] if nrm[0] else rng.choice([-1.0, 1.0]); sy = nrm[1] if nrm[1] else rng.choice([-1.0, 1.0])
            tgt = np.array([sx, sy]) * (F.H - rng.uniform(0.5, 2.0))
            st["des"] = float(np.arctan2(tgt[1] - p_[1], tgt[0] - p_[0]))
        else:
            ang = rng.uniform(30, 85) if rng.random() < 0.75 else rng.uniform(0, 30)   # mostly angled approaches
            st["des"] = np.arctan2(nrm[1], nrm[0]) + np.deg2rad(ang) * rng.choice([-1, 1])
        st.update(d0=rng.uniform(1.2, 2.0), d1=rng.uniform(0.8, 1.2), d2=rng.uniform(0.4, 0.8), marked=-1,
                  straight=False, fwd=int(rng.choice(F.FWD)), n_app=st["n_app"] + 1)   # episodes = successful escapes only

    def away(w):
        """world angle pointing from wall w into the room (its inward normal)"""
        return (np.pi, 0.0, -np.pi / 2, np.pi / 2)[w]

    def facing_open(pos, yw, e):
        """escape criterion (2): within 10 deg of the escape's target heading AND > 4 m free ahead"""
        return abs(np.angle(np.exp(1j * (e["target"] - yw)))) < np.deg2rad(10) and F.front_dist(pos, yw) > 4.0

    def start_escape(j, pos, yw, cause):
        """target heading = current heading turned by a RANDOM amount: size drawn once per escape (small 20-60 /
        medium 60-120 / sharp 120-180 deg, uniform), 30 tries within that size, other sizes only if it is impossible;
        the target must have 4-10 m free (small turns: > 4 m, along a wall is long) and point along or away from the escaped wall (fallback: directly away)"""
        near = int(np.argmin([wall_dist(pos, w) for w in range(4)]))
        tgt, cat, amt = away(near), "fallback", np.nan
        R = {"small": (20, 60), "medium": (60, 120), "sharp": (120, 180)}
        first = str(rng.choice(list(R)))                         # size drawn ONCE; others only if it is impossible
        found = False
        for cat in [first] + [k for k in R if k != first]:
            for _ in range(30):
                amt = rng.uniform(*R[cat])
                ok = [sg for sg in (-1, 1) if 4.0 < free_dist(pos, yw + sg * np.deg2rad(amt)) < (99.0 if cat == "small" else 10.0)
                      and np.cos(yw + sg * np.deg2rad(amt) - away(near)) >= 0.0]   # along or away, never into the wall
                if ok:
                    tgt = float(yw + rng.choice(ok) * np.deg2rad(amt)); found = True; break
            if found:
                break
        if not found:
            cat, amt = "fallback", float(abs(np.degrees(np.angle(np.exp(1j * (tgt - yw))))))
        escapes.append(dict(start=j, cause=cause, wall=near, target=float(np.angle(np.exp(1j * tgt))), turn_cat=cat,
                            turn_amount=float(amt), crossed=False, open=-1, walk_start=-1,
                            walk_dur=float(rng.uniform(1.0, 2.5)), d_walk0=np.nan, moved=np.nan, completed=False, end=-1))
        st.update(phase="room", t=j)

    def safe(ci, thr=0.15):
        return lookahead(r, cmds[ci], 1.0)[0] > thr

    new_approach(first=True)                                     # first approach: the start heading, marks set
    for j in range(n_frames):
        pos, yw = r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7])
        c = F.clearance(pos, yw, None)
        ph = st["phase"]; e = escapes[-1] if escapes else None
        if ph in ("room", "turnout", "walkout") and j - e["start"] > 240:       # escape cap 12 s
            e["end"] = j; st["last_wall"] = e["wall"]; new_approach(); st.update(phase="orient", t=j); ph = "orient"
        if ph == "orient":
            err = float(np.angle(np.exp(1j * (st["des"] - yw))))
            if abs(err) < np.deg2rad(15) or j - st["t"] > 80:
                st.update(phase="approach", t=j); ph = "approach"
            elif j % 10 == 0:
                choice = int(rng.choice(TL if err > 0 else TR))
                if not safe(choice, 0.3):
                    start_escape(j, pos, yw, "orient"); ph = "room"
        if ph == "approach":
            choice = st["fwd"]
            if st["marked"] == -1 and c < st["d0"] and j >= PREFIX:
                states.append(j); st["marked"] = 0                 # FAR branch state 1.2-2.0 m (early decisions)
            if st["marked"] == 0 and c < st["d1"] and j >= PREFIX:
                if not states or states[-1] != j:
                    states.append(j)                               # branch state: the state BEFORE frame j's step
                st["marked"] = 1
            if st["marked"] == 1 and c < st["d2"] and j >= PREFIX and not st["straight"]:
                if states[-1] != j:
                    states.append(j)
                st["marked"] = 2
                start_escape(j, pos, yw, "approach"); ph = "room"
        if ph == "room":
            e = escapes[-1]
            rel = float(np.angle(np.exp(1j * (e["target"] - yw))))
            turn = SPL if rel > 0 else SPR
            if abs(rel) > np.deg2rad(20) and (j - st["t"]) % 10 == 0 and safe(int(turn[-1]), 0.2):
                st.update(phase="turnout", t=j); ph = "turnout"
            elif facing_open(pos, yw, e):
                st.update(phase="turnout", t=j); ph = "turnout"       # already facing the open direction
            elif (j - st["t"]) % 20 == 0:                           # every 1 s: back up / side-step toward the target
                fams = [BWD] if abs(rel) > np.deg2rad(100) else ([SL, BWD] if rel > 0 else [SR, BWD])
                cand = [int(ci) for fm in fams for ci in rng.permutation(fm)]
                choice = next((ci for ci in cand if safe(ci)),
                              max(BWD + SL + SR + SPL + SPR, key=lambda ci: lookahead(r, cmds[ci], 1.0)[0]))
        if ph == "turnout":
            e = escapes[-1]
            rel = float(np.angle(np.exp(1j * (e["target"] - yw))))
            if facing_open(pos, yw, e):
                e["open"] = j; e["walk_start"] = j; e["d_walk0"] = wall_dist(pos, e["wall"])
                st.update(phase="walkout", t=j, fwd=int(rng.choice(F.FWD))); ph = "walkout"
            elif (j - st["t"]) % 10 == 0:
                choice = int(rng.choice(SPL if rel > 0 else SPR))
                if not safe(choice, 0.2):
                    st.update(phase="room", t=j); ph = "room"
                    choice = next((int(ci) for ci in rng.permutation(BWD + SL + SR) if safe(int(ci))),
                                  max(BWD + SL + SR + SPL + SPR, key=lambda ci: lookahead(r, cmds[ci], 1.0)[0]))
        if ph == "walkout":
            e = escapes[-1]
            choice = st["fwd"]
            if e["walk_start"] == j:
                e["c_walk0"], e["closer"], e["into_wall"] = float(c), 0.0, -1.0
            # check (3), user 2026-10-11: during the walk-out the robot never faces back INTO the escaped wall (along
            # it or away is fine: cos(heading - away from the wall) >= -0.17, i.e. at most 10 deg into it) and never
            # gets closer to any wall (clearance never > 0.10 m below its walk-out start: a few deg of gait yaw sway moves the
            # corner of the 1.13 m footprint ~5 cm, measured 0.05-0.09 m on along-wall walk-outs)
            e["closer"] = max(e["closer"], e["c_walk0"] - float(c))
            e["into_wall"] = max(e["into_wall"], -float(np.cos(yw - away(e["wall"]))))
            if (j - st["t"]) % 10 == 0 and not safe(choice, 0.3):
                st.update(phase="room", t=j); e["walk_start"] = -1      # walk-out interrupted: make room again
            elif (j - e["walk_start"]) * 0.05 >= e["walk_dur"]:
                e["moved"] = wall_dist(pos, e["wall"]) - e["d_walk0"]
                e["completed"] = bool(e["into_wall"] <= 0.17 and e["closer"] <= 0.10); e["end"] = j
                st["last_wall"] = e["wall"]; new_approach(); st["des"] = yw     # keep walking into the wall ahead
                st.update(phase="orient", t=j)
        snaps.append(r.snapshot())
        tgt = int(round(2.5 * (j + 1)))
        while step < tgt:
            last = r.step(*cmds[choice]); step += 1
        pos, yw = r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7])
        c = F.clearance(pos, yw, None)
        rec.append(dict({f: np.asarray(last[f]).copy() for f in FIELDS}, clearance=c, front=F.front_dist(pos, yw),
                        cmd=choice, phase=st["phase"]))
        if c < 0:
            if escapes and escapes[-1]["end"] < 0 and st["phase"] in ("room", "turnout", "walkout"):
                escapes[-1].update(crossed=True, end=j)
            break
    if escapes and escapes[-1]["end"] < 0:
        escapes[-1]["end"] = len(rec)                            # still running when the episode ended
        escapes[-1]["unfinished"] = True
    return r, rec, snaps, states, escapes


DATA = os.path.join(ROOT, "data/wall_b1")
PREFIX, BRANCH_F = 10, 60          # branch files: 10 shared frames before the branch point + 3 s (60 frames) after
FAR_FROM, BRANCH_F_FAR = 1.2, 100  # far band (start clearance >= 1.2 m): 5 s (100 frames) -- PILOT ONLY (DATA_PLAN 13)
BINS = ((0.4, 0.9), (0.9, 1.4), (1.4, 2.0))   # one branch state per clearance bin (m), if the episode has one


def _arr(recs, f, dtype=np.float32):
    return np.asarray([x[f] for x in recs], dtype)


# Per-frame wall label (user + Codex 2026-10-11), fixed BEFORE any test result: margin 0.15 m = 0.13 B1 body lengths
# (footprint 1.13 m); for the hexapod the same 0.13 body lengths. Nothing physically touches in MuJoCo (virtual
# boundary): 0 SAFE (clearance >= margin), 1 NEAR = margin violation (0 <= clearance < margin; 0 = touching the
# boundary), 2 CROSSED (< 0, the terminal frame). Near and crossed are reported SEPARATELY (unsafe = either, for
# selection); every unsafe label states its time window (first_near_frame / cross_frame -> "within 3 s of the branch"
# vs "within the predicted 1.25 s" are computed separately, never interchanged).
MARGIN = 0.15
MARGIN_BODY_LENGTHS = 0.13
STATUS_NAMES = ("safe", "near", "crossed")


def frame_status(clear):
    """0 safe (>= MARGIN), 1 near = margin violation (0 <= c < MARGIN), 2 crossed (< 0)"""
    c = np.asarray(clear, np.float64)
    return np.where(c < 0, 2, np.where(c < MARGIN, 1, 0)).astype(np.int8)


def _terminal(clear):
    """boundary crossing as a terminal event: index of the first frame with clearance < 0 (kept; nothing after it is
    stored), or -1. Returns (crossed, cross_frame, clearance on the frame before)."""
    neg = np.flatnonzero(np.asarray(clear) < 0)
    if not len(neg):
        return False, -1, np.nan
    k = int(neg[0])
    return True, k, float(clear[k - 1]) if k > 0 else np.nan


def _save(path, recs, meta, segment):
    from wm.data.com import b1_com
    d = {f: _arr(recs, f) for f in FIELDS}
    clear = np.asarray([x["clearance"] for x in recs], np.float64)
    crossed, kx, cb = _terminal(clear)
    status = frame_status(clear)
    d["com_pos"] = np.asarray(b1_com(d["base_pos"].astype(np.float64), d["base_quat"].astype(np.float64),
                                     d["joint_pos"].astype(np.float64),
                                     os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat.xml")), np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **d, clearance=clear, front_dist=np.asarray([x["front"] for x in recs], np.float64),
                        cmd_index=np.asarray([x["cmd"] for x in recs], np.int64), segment=np.asarray(segment, np.int8),
                        crossed=np.bool_(crossed), cross_frame=np.int64(kx), clear_before_cross=np.float64(cb),
                        status=status, margin=np.float64(MARGIN), margin_body_lengths=np.float64(MARGIN_BODY_LENGTHS),
                        first_near_frame=np.int64(np.flatnonzero(status >= 1)[0] if (status >= 1).any() else -1),
                        dt=np.float64(0.05), fps=np.float64(20.0), embodiment=np.array("b1"),
                        foot_lw=np.array([F.FOOT_L, F.FOOT_W]), boundary_half=np.float64(F.H),
                        boundary=np.array("virtual: MuJoCo has no walls; crossing = footprint clearance < 0, stored as a "
                                          "terminal frame, nothing after it"),
                        model=np.array("sim/assets/b1_mujoco/b1_flat.xml"), **meta)
    os.replace(tmp, path)


def snap_clear(r, snap):
    r.restore(snap)
    return F.clearance(r.d.qpos[0:2].copy(), F.yaw_of(r.d.qpos[3:7]), None)


def physics_job(args):
    """one episode + its branches (no frames). Branch state j0: the episode frames before it are rec[j0-10:j0]
    (rec[j0-1] is exactly the snapshot snaps[j0] restored for the branch); branch frames = the command's 40 frames."""
    ep, split, look = args
    out = os.path.join(DATA, "_physics", split)
    if os.path.exists(os.path.join(out, f"ep{ep:05d}.npz")):
        return f"ep{ep} exists"
    cmds, fam = commands28(); names = cmd_names()
    r, rec, snaps, states, escapes = episode_approach(ep)
    style = "approach"
    seg = np.cumsum(np.r_[0, np.diff([x["cmd"] for x in rec]) != 0])
    meta = dict(episode=np.int64(ep), split=np.array(split), room_seed=np.int64(look), style=np.array(style),
                kind=np.array("episode"))
    ep_meta = dict(meta, escapes=np.array(json.dumps(escapes)), phase=np.array([x["phase"] for x in rec]))
    for b, j0 in enumerate(states):
        nf = BRANCH_F_FAR if snap_clear(r, snaps[j0]) >= FAR_FROM else BRANCH_F
        for ci, c in enumerate(cmds):
            from build_b1_cf_branches import run_branch
            br, fell = run_branch(r, snaps[j0], c, OFFS_FRAME(nf))
            fut = []
            for k in range(nf):
                pos, yw = br["base_pos"][k][:2], F.yaw_of(br["base_quat"][k])
                cl = F.clearance(pos, yw, None)
                fut.append(dict({f: br[f][k] for f in FIELDS}, clearance=cl, front=F.front_dist(pos, yw), cmd=ci))
                if cl < 0:                                          # terminal event: keep this frame, stop
                    break
            recs = rec[j0 - PREFIX:j0] + fut
            _save(os.path.join(out, f"ep{ep:05d}_b{b}_c{ci:02d}.npz"), recs,
                  dict(meta, kind=np.array("branch"), branch_state=np.int64(b), branch_frame=np.int64(j0),
                       first_pair=np.int64(PREFIX), cf_command_index=np.int64(ci), cf_command=np.asarray(c, np.float32),
                       cf_command_name=np.array(names[ci]), cf_fell=np.bool_(fell),
                       start_clearance=np.float64(snap_clear(r, snaps[j0])), branch_seconds=np.float64(nf * 0.05),
                       froude_height=np.float64(np.median(_arr(rec, "base_pos")[:, 2]))),
                  [0] * PREFIX + [1] * len(fut))
    _save(os.path.join(out, f"ep{ep:05d}.npz"), rec, ep_meta, seg)       # episode last = job complete marker
    cl = [x["clearance"] for x in rec]
    return (f"ep{ep} {style}: {len(rec)} frames, min clearance {min(cl):+.2f}, crossed {cl[-1] < 0}; branch states at "
            + ", ".join(f"{snap_clear(r, snaps[j]):.2f}" for j in states) + " m")


def do_physics(a):
    from multiprocessing import Pool
    J = [(a.ep0 + i, a.split, a.look0 + i) for i in range(a.n)]
    with Pool(a.workers) as p:
        for m in p.imap_unordered(physics_job, J):
            print(m, flush=True)


def do_render(a):
    import glob
    import render_shift_heldout as RS
    J = sorted(glob.glob(os.path.join(DATA, "_physics", a.split, "*.npz")))
    J = [p for p in J if not os.path.exists(os.path.join(DATA, a.split, os.path.basename(p)))]
    print(f"{len(J)} files to render", flush=True)

    def fn(sim, p):
        d = dict(np.load(p, allow_pickle=True))
        dd = dict(base_pos=d["base_pos"], base_quat=d["base_quat"], joint_pos=d["joint_pos"],
                  room_seed=np.int64(d["room_seed"]))
        fr, cp, R = RS.b1_render(sim, dd, np.zeros(2))          # original room at the world origin, not re-centred
        dst = os.path.join(DATA, a.split, os.path.basename(p))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **d, frames=fr, cam_pose=cp, room_size=np.float64(R["size"]))
        os.replace(tmp, dst)
        return f"{os.path.basename(p)} clear {d['clearance'].min():+.2f}"
    RS.SCENE_HOOK = None
    RS.run_pool(J, a.ports, fn, "render")


def do_review(a):
    """review material: per-episode videos + contact sheet; close-range (< 0.6 m) frames; safe vs crossing branches;
    label table."""
    import glob
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    D = os.path.join(DATA, a.split); out = os.path.join(CHK, f"review_{a.split}"); os.makedirs(out, exist_ok=True)
    gray = lambda f: np.repeat(np.asarray(Image.fromarray(f).convert("L"))[..., None], 3, -1)   # noqa: E731  as the model sees
    tile = lambda f, t, col=(255, 255, 0): _tile(f, t, col)  # noqa: E731
    eps = sorted(p for p in glob.glob(os.path.join(D, "ep*.npz")) if "_b" not in os.path.basename(p))
    NAMES = cmd_names()
    lines, sheet = [], []
    for p in eps:
        d = np.load(p, allow_pickle=True); n = len(d["clearance"]); e = int(d["episode"])
        wr = imageio.get_writer(os.path.join(out, f"episode_{e}_{d['style']}.mp4"), fps=20)
        for j in range(n):
            ph = str(d["phase"][j]) if "phase" in d.files else ""
            wr.append_data(np.hstack([tile(d["frames"][j], f"ep{e} t={j * .05:.2f}s {ph} {NAMES[int(d['cmd_index'][j])]} "
                                       f"clear {d['clearance'][j]:+.2f}"), tile(gray(d["frames"][j]), "gray")]))
        wr.close()
        ks = list(range(0, n, max(1, n // 7)))[:7] + [n - 1]
        sheet.append(np.hstack([tile(gray(d["frames"][k]), f"ep{e} {k * .05:.1f}s {d['clearance'][k]:+.2f}m") for k in ks]))
        lines.append(f"ep{e} {str(d['style']):5s} {n * .05:5.1f}s min clear {d['clearance'].min():+.2f} crossed "
                     f"{bool(d['crossed'])} (frame {int(d['cross_frame'])}, before {float(d['clear_before_cross']):+.2f})")
    Image.fromarray(np.vstack(sheet)).save(os.path.join(out, "episodes_sheet_gray.png"))
    # branches: per state, safe vs crossing outcomes + labels
    br = sorted(glob.glob(os.path.join(D, "ep*_b*_c*.npz")))
    st = {}
    for p in br:
        d = np.load(p, allow_pickle=True)
        st.setdefault(os.path.basename(p).rsplit("_c", 1)[0], []).append(
            (int(d["cf_command_index"]), str(d["cf_command_name"]), bool(d["crossed"]), int(d["cross_frame"]),
             float(d["clearance"][PREFIX:].min()), float(d["start_clearance"]), p))
    lines.append(f"\n{len(st)} branch states, {len(br)} branch files")
    nx = 0
    for k, v in sorted(st.items()):
        x = [u for u in v if u[2]]
        lines.append(f"{k}: start clearance {v[0][5]:.2f} m; crossing {len(x)}/{len(v)} "
                     f"({', '.join(u[1] for u in x)}); min clearance over 2 s safe ones {min((u[4] for u in v if not u[2]), default=np.nan):+.2f}")
        if x and nx < 4:                                        # one crossing vs the safest command, frames every 0.5 s
            safe = max(v, key=lambda u: u[4]); rows = []
            for u in (x[0], safe):
                d = np.load(u[6], allow_pickle=True); m = len(d["clearance"])
                ks = [min(PREFIX - 1 + q, m - 1) for q in (0, 10, 20, 30, 40)]
                rows.append(np.hstack([tile(gray(d["frames"][q]), f"{u[1]} {(q - PREFIX + 1) * .05:+.1f}s {d['clearance'][q]:+.2f}",
                                            (255, 80, 80) if u[2] else (80, 255, 80)) for q in ks]))
            Image.fromarray(np.vstack(rows)).save(os.path.join(out, f"branch_{k}_cross_vs_safe.png")); nx += 1
    # close range: every stored frame with clearance < 0.6 m, binned, as the model sees it (gray)
    rows = []
    for lo, hi in ((0.4, 0.6), (0.2, 0.4), (0.0, 0.2), (-1, 0.0)):
        fs = []
        for p in eps + br:
            d = np.load(p, allow_pickle=True)
            for j in np.flatnonzero((d["clearance"] >= lo) & (d["clearance"] < hi))[:1]:
                fs.append(tile(gray(d["frames"][j]), f"{os.path.basename(p)[:-4]} {d['clearance'][j]:+.2f}m"))
            if len(fs) >= 6:
                break
        if fs:
            rows.append(np.hstack(fs + [np.zeros_like(fs[0])] * (6 - len(fs))))
    Image.fromarray(np.vstack(rows)).save(os.path.join(out, "close_range_gray.png"))
    # approach history of the close states (start clearance < 0.8 m): frames 1.0 / 0.5 / 0.25 s before + the state,
    # as the model sees them (gray); approach angle = heading vs the nearest wall's normal at the state
    rows = []
    for k, v in sorted(st.items()):
        if v[0][5] >= 0.8:
            continue
        d = np.load(v[0][6], allow_pickle=True)
        bp, yw = d["base_pos"][PREFIX - 1], F.yaw_of(d["base_quat"][PREFIX - 1])
        nrm = [(1, 0), (-1, 0), (0, 1), (0, -1)][int(np.argmin([F.H - bp[0], F.H + bp[0], F.H - bp[1], F.H + bp[1]]))]
        ang = abs(np.degrees(np.angle(np.exp(1j * (yw - np.arctan2(nrm[1], nrm[0]))))))
        corner = sorted([F.H - abs(bp[0]), F.H - abs(bp[1])])[1] < 1.5
        kind = "corner" if corner else ("head-on" if ang < 30 else "oblique")
        rows.append(np.hstack([tile(gray(d["frames"][q]), f"{k} {kind} {ang:.0f}deg {(q - PREFIX + 1) * .05:+.2f}s "
                                    f"{d['clearance'][q]:+.2f}m") for q in (0, 5, 8, PREFIX - 1)]))
        if len(rows) >= 12:
            break
    if rows:
        Image.fromarray(np.vstack(rows)).save(os.path.join(out, "close_states_history_gray.png"))
    open(os.path.join(out, "labels.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("->", os.path.relpath(out, ROOT))


def do_stats(a):
    """physics-only check (no frames): per branch state, how many of the 24 commands cross the boundary within
    0.5 / 1 / 1.25 / 3 s and how many show NO CROSSING UNDER A HELD COMMAND for 3 s (a branch holds one command: it is
    not a complete escape -- those are measured on the episodes); states where all 24 cross are flagged. Top-down plot of the episodes with the branch states."""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = os.path.join(DATA, "_physics", a.split); out = os.path.join(CHK, f"stats_{a.split}"); os.makedirs(out, exist_ok=True)
    _, fam = commands28(); NC = len(fam)
    eps = sorted(p for p in glob.glob(os.path.join(D, "ep*.npz")) if "_b" not in os.path.basename(p))
    T = (0.5, 1.0, 1.25, 3.0)
    lines = ["state | start clearance m | crossing within " + " / ".join(f"{t:g}" for t in T) + " s | no crossing under the "
             "held command (3 s) | no-crossing families | crossing families"]
    tot = collections.Counter(); fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_patch(plt.Rectangle((-F.H, -F.H), 2 * F.H, 2 * F.H, fill=False, lw=2))
    E = []
    col = dict(orient="0.6", approach="tab:blue", room="tab:red", turnout="tab:orange", walkout="tab:green")
    for p in eps:
        d = np.load(p, allow_pickle=True); bp = d["base_pos"]
        ph = d["phase"] if "phase" in d.files else np.array(["approach"] * len(bp))
        for k in range(len(bp) - 1):
            ax.plot(bp[k:k + 2, 0], bp[k:k + 2, 1], lw=1.0, color=col.get(str(ph[k]), "k"))
        if "escapes" in d.files:
            E += [dict(x, episode=int(d["episode"])) for x in json.loads(str(d["escapes"]))]
        if bool(d["crossed"]):
            ax.plot(bp[-1, 0], bp[-1, 1], "rx", ms=8)
        e = int(d["episode"])
        for b in sorted({int(os.path.basename(x).split("_b")[1][0]) for x in glob.glob(p[:-4] + "_b*_c00.npz")}):
            fs = sorted(glob.glob(p[:-4] + f"_b{b}_c*.npz"))
            cr = []
            for f in fs:
                x = np.load(f, allow_pickle=True)
                cr.append((int(x["cross_frame"]) - PREFIX + 1) * 0.05 if bool(x["crossed"]) else np.inf)
            x0 = np.load(fs[0], allow_pickle=True); sc = float(x0["start_clearance"]); j0 = int(x0["branch_frame"])
            cr = np.array(cr); esc = int((cr == np.inf).sum())
            band = "0.4-0.8" if sc < 0.8 else "0.8-1.2" if sc < 1.2 else "1.2-2.0"
            tot[(band, "states")] += 1; tot[(band, "no escape")] += int(esc == 0); tot[(band, "escapes")] += esc
            tot[(band, "crossings 3s")] += len(cr) - esc
            fe = collections.Counter(fam[i] for i in range(len(cr)) if cr[i] == np.inf)
            fc = collections.Counter(fam[i] for i in range(len(cr)) if cr[i] < np.inf)
            lines.append(f"ep{e}_b{b} | {sc:.2f} | " + " / ".join(str(int((cr <= t + 1e-9).sum())) for t in T)
                         + f" | {esc}{'  << ALL CROSS' if esc == 0 else ''} | {dict(fe)} | {dict(fc)}")
            ax.plot(bp[j0 - 1, 0], bp[j0 - 1, 1], "o", ms=5, color="k" if esc else "r")
    sets = [l.split(" | ")[4] for l in lines[1:]]
    common = collections.Counter(sets).most_common(1)[0] if sets else ("", 0)
    lines.append(f"\nstate dependence: {len(set(sets))} distinct no-crossing-family sets over {len(sets)} states; the most "
                 f"common ({common[0]}) in {common[1]} -- a command-only rule is right on those")
    lines.append("per start-clearance band: " + "; ".join(
        f"{b}: {tot[(b, 'states')]} states, mean no-crossing {tot[(b, 'escapes')] / max(tot[(b, 'states')], 1):.1f}/{NC}, "
        f"mean crossings(3 s) {tot[(b, 'crossings 3s')] / max(tot[(b, 'states')], 1):.1f}/{NC}, all-cross states "
        f"{tot[(b, 'no escape')]}" for b in ("0.4-0.8", "0.8-1.2", "1.2-2.0")))
    if E:
        fin = [x for x in E if not x.get("unfinished")]
        ok3 = [x for x in fin if not x["crossed"] and x["open"] >= 0 and x["completed"]]
        lines.append(f"\nCOMPLETE ESCAPES (episodes): {len(E)} escapes started, {len(E) - len(fin)} still running at the "
                     f"episode end (excluded); of {len(fin)} finished: (1) no crossing {sum(not x['crossed'] for x in fin)}, "
                     f"(2) reached facing open space (within 10 deg of its target heading, > 4 m free ahead) "
                     f"{sum(x['open'] >= 0 for x in fin)}, (3) walked out 1-2.5 s never facing back into the escaped wall and never closer to any wall "
                     f"{sum(bool(x['completed']) for x in fin)}; ALL THREE {len(ok3)}/{len(fin)} = "
                     f"{len(ok3) / max(len(fin), 1):.2f}")
        if ok3:
            t_open = np.array([(x["open"] - x["start"]) * 0.05 for x in ok3])
            t_all = np.array([(x["end"] - x["start"]) * 0.05 for x in ok3])
            cats = collections.Counter(x.get("turn_cat", "-") for x in ok3)
            amts = np.array([x.get("turn_amount", np.nan) for x in ok3], float)
            gaps_ = [(E[i + 1]["start"] - x["end"]) * 0.05 for i, x in enumerate(E[:-1])
                     if x in ok3 and E[i + 1]["episode"] == x["episode"]]
            lines.append(f"turn per escape: {dict(cats)}; amount median {np.nanmedian(amts):.0f} deg, 10-90% "
                         f"{np.nanpercentile(amts, 10):.0f}-{np.nanpercentile(amts, 90):.0f}, min {np.nanmin(amts):.0f} max "
                         f"{np.nanmax(amts):.0f}; end of walk-out -> next escape: median "
                         f"{np.median(gaps_) if gaps_ else np.nan:.1f} s")
            lines.append(f"time from escape start to facing open space: median {np.median(t_open):.1f} s (range "
                         f"{t_open.min():.1f}-{t_open.max():.1f}); to the end of the walk-out: median {np.median(t_all):.1f} s")
        bad = [x for x in fin if x not in ok3]
        lines.append("not complete: " + ", ".join(f"ep{x['episode']}@{x['start'] * 0.05:.1f}s("
                     + ("crossed" if x["crossed"] else "no open view" if x["open"] < 0 else
                        f"into wall {x.get('into_wall', np.nan):+.2f}, closer {x.get('closer', np.nan):.2f} m")
                     + ")" for x in bad))
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color=v) for v in col.values()], list(col), fontsize=7, loc="upper right")
    ax.set_aspect("equal"); ax.set_title("episodes by phase; dots = branch states (red = all cross), x = crossing")
    fig.savefig(os.path.join(out, "topdown.png"), dpi=110); plt.close(fig)
    open(os.path.join(out, "stats.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("->", os.path.relpath(out, ROOT))


def do_sep(a):
    """Is a crossing within 3 s decidable from the state at the rollout horizon h (1.25 s)? Per branch state and
    command: clearance at h, closing rate at h (clearance change over the last 0.25 s, m/s), front distance at h; label =
    crosses within 3 s (a crossing before h counts as known). Reports (1) within-state AUC of each feature (mean over
    states with both outcomes; ties averaged), (2) leave-one-episode-out logistic rule on [clearance_h, rate_h] -> share of
    states where picking the command with the highest predicted safety is truly safe for 3 s, (3) best FIXED command
    (from the data): share of states where it escapes; best fixed ranking = commands ordered by their escape rate over
    the other episodes (leave-one-episode-out), top-1 safe share."""
    import glob
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    D = os.path.join(DATA, "_physics", a.split); _, fam = commands28()
    H = a.h; kh = PREFIX - 1 + int(round(H / 0.05)); k25 = kh - 5
    rows = []                                                  # (episode, state, cmd, clear_h, rate_h, front_h, crosses)
    for f in sorted(glob.glob(os.path.join(D, "ep*_b*_c*.npz"))):
        x = np.load(f, allow_pickle=True); cl, fr = x["clearance"], x["front_dist"]
        crossed = bool(x["crossed"]); k = min(kh, len(cl) - 1)
        before = crossed and int(x["cross_frame"]) <= kh
        ch = -0.01 if before else cl[k]; rate = (cl[k] - cl[max(k - 5, 0)]) / 0.25
        b = os.path.basename(f)
        unsafe = bool((frame_status(cl[PREFIX:]) >= 1).any())                      # near (< margin) or crossed
        rows.append((int(b[2:7]), b.rsplit("_c", 1)[0], int(x["cf_command_index"]), ch, rate, fr[k], crossed,
                     float(x["start_clearance"]), unsafe,
                     float(x["branch_seconds"]) if "branch_seconds" in x.files else 3.0))
    ep = np.array([r[0] for r in rows]); st = np.array([r[1] for r in rows]); ci = np.array([r[2] for r in rows])
    X = np.array([[r[3], r[4], r[5]] for r in rows]); y = np.array([r[6] for r in rows])
    sc0 = np.array([r[7] for r in rows])
    L = [f"split {a.split}: {len(set(st))} states, {len(rows)} branches, horizon {H} s, label = crosses within the branch (3 s; far band 5 s) "
         f"(base rate {y.mean():.2f})",
         "NOTE: features are the TRUE state at the horizon (physics) -- an upper bound on what an imagined future could "
         "carry, NOT evidence that the FTM predicts it"]
    for lo, hi in ((0.4, 0.8), (0.8, 1.2), (1.2, 2.0)):
        m_ = (sc0 >= lo) & (sc0 < hi); S_ = sorted(set(st[m_]))
        if not S_:
            continue
        n_cross = [int(y[st == s_].sum()) for s_ in S_]
        un = np.array([r[8] for r in rows]); bs = np.array([r[9] for r in rows])
        au = [roc_auc_score(~y[st == s_], X[st == s_, 0]) for s_ in S_ if 0 < y[st == s_].sum() < (st == s_).sum()]
        au_u = [roc_auc_score(~un[st == s_], X[st == s_, 0]) for s_ in S_ if 0 < un[st == s_].sum() < (st == s_).sum()]
        L.append(f"  [{lo}-{hi} m] branch length {np.unique(bs[m_]).tolist()} s; UNSAFE (near < margin or crossed) per state "
                 f"mean {np.mean([un[st == s_].sum() for s_ in S_]):.1f}; AUC of clearance at h for unsafe: "
                 + (f"{np.mean(au_u):.3f} (min {np.min(au_u):.2f}, {len(au_u)} states)" if au_u else "n/a"))
        L.append(f"band {lo}-{hi} m: {len(S_)} states; crossings within 3 s per state mean {np.mean(n_cross):.1f}/"
                 f"{(st == S_[0]).sum()} (states with none: {sum(v == 0 for v in n_cross)}); within-state AUC of clearance "
                 f"at h for crossing within 3 s: " + (f"{np.mean(au):.3f} (min {np.min(au):.2f}, {len(au)} states with "
                 f"both outcomes)" if au else "n/a (no state with both outcomes)"))
    for j, nm in enumerate(("clearance at h", "closing rate at h", "front distance at h")):
        au = [roc_auc_score(~y[st == s], X[st == s, j]) for s in set(st) if 0 < y[st == s].sum() < (st == s).sum()]
        L.append(f"within-state AUC, {nm:20s}: {np.mean(au):.3f} (min {np.min(au):.2f}, {len(au)} states)")
    safe_pick, fixed_pick, top_rank = [], [], []
    for e in sorted(set(ep)):
        tr, te = ep != e, ep == e
        lr = LogisticRegression(max_iter=1000).fit(X[tr][:, :2], y[tr])
        esc_rate = np.array([1 - y[tr & (ci == c)].mean() for c in range(len(fam))])
        for s in sorted(set(st[te])):
            m = st == s; p = lr.predict_proba(X[m][:, :2])[:, 1]
            safe_pick.append(not y[m][np.argmin(p)])
            fixed_pick.append(not y[m][ci[m] == int(np.argmax(esc_rate))][0])
    esc_all = np.array([1 - y[ci == c].mean() for c in range(len(fam))])
    bc = int(np.argmax(esc_all))
    L.append(f"pick the command with the safest state at h (LOEO logistic on clearance, rate): truly safe for 3 s in "
             f"{np.mean(safe_pick):.2f} of states")
    L.append(f"best fixed command (LOEO, by escape rate on other episodes): safe in {np.mean(fixed_pick):.2f} of states; "
             f"over all data the best is {fam[bc]}{bc % 4} (escapes in {esc_all[bc]:.2f} of states)")
    L.append("escape rate per command over all states: " + ", ".join(f"{fam[c]}{c % 4} {esc_all[c]:.2f}" for c in
                                                                      np.argsort(-esc_all)))
    os.makedirs(os.path.join(CHK, f"stats_{a.split}"), exist_ok=True)
    open(os.path.join(CHK, f"stats_{a.split}", f"separability_h{H}.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def do_branchgrid(a):
    """ONE start state, SIX commands side by side over the 3 s branch (grayscale = model input), for 3 states: the
    closest head-on, the most angled, and a corner state. Shows that different commands from the same view lead to
    different outcomes (crossing frames in red)."""
    import glob
    import imageio.v2 as imageio
    from PIL import Image
    D = os.path.join(DATA, a.split); out = os.path.join(CHK, f"review_{a.split}"); os.makedirs(out, exist_ok=True)
    names = cmd_names(); show = (2, 6, 10, 18, 22, 25)          # fwd2 bwd2 turn_left2 side_L2 side_R2 spin_left_w0.45
    gray = lambda f: np.repeat(np.asarray(Image.fromarray(f).convert("L"))[..., None], 3, -1)   # noqa: E731
    info = {}
    for f in sorted(glob.glob(os.path.join(D, "ep*_b*_c00.npz"))):
        x = np.load(f, allow_pickle=True); bp = x["base_pos"][PREFIX - 1]; yw = F.yaw_of(x["base_quat"][PREFIX - 1])
        dw = [F.H - bp[0], F.H + bp[0], F.H - bp[1], F.H + bp[1]]; w = int(np.argmin(dw))
        ang = abs(np.degrees(np.angle(np.exp(1j * (yw - (0, np.pi, np.pi / 2, -np.pi / 2)[w])))))
        info[f[:-8]] = (float(x["start_clearance"]), ang, sorted(dw)[1] < 1.5 + F.FOOT_L / 2)
    if not info:
        print("no branch files"); return
    pick = {"head-on, close": min((k for k in info if info[k][1] < 30), key=lambda k: info[k][0], default=None),
            "back to wall": max(info, key=lambda k: info[k][1]),
            "corner": min((k for k in info if info[k][2]), key=lambda k: info[k][0], default=None)}
    for lab, k in pick.items():
        if k is None:
            continue
        D_ = [np.load(f"{k}_c{c:02d}.npz", allow_pickle=True) for c in show]
        n = max(len(d["frames"]) for d in D_)
        wr = imageio.get_writer(os.path.join(out, f"branchgrid_{lab.replace(', ', '_').replace(' ', '_')}.mp4"), fps=10)
        for t in range(0, n):
            tiles = []
            for c, d in zip(show, D_):
                q = min(t, len(d["frames"]) - 1); cl = float(d["clearance"][q]); stt = int(frame_status([cl])[0])
                col = ((80, 255, 80), (255, 230, 0), (255, 60, 60))[stt]
                txt = f"{names[c]} {(q - PREFIX + 1) * .05:+.2f}s {cl:+.2f}m {STATUS_NAMES[stt].upper()}"
                if t > q:
                    txt += " (stopped)"
                tiles.append(_tile(gray(d["frames"][q]), txt, col))
            wr.append_data(np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
        wr.close()
        print(f"{lab}: {os.path.basename(k)}, start clearance {info[k][0]:.2f} m, angle {info[k][1]:.0f} deg; "
              + ", ".join(f"{names[c]} {STATUS_NAMES[int(frame_status(d['clearance']).max())]}" for c, d in zip(show, D_)))


def _map(paths, size=384, view=None, title=""):
    """ALLOCENTRIC top-down map (PIL): virtual boundary square, each path = (xy (T,2), yaw (T,), colours per point,
    draw_footprint_at_last). view = (cx, cy, half) metres to zoom on; default the whole room."""
    from PIL import Image, ImageDraw
    cx, cy, half = view if view else (0.0, 0.0, F.H + 0.5)
    S = size / (2 * half)
    P_ = lambda x, y: ((x - cx + half) * S, (half - (y - cy)) * S)   # noqa: E731  world -> pixel, +y up
    im = Image.new("RGB", (size, size), (25, 25, 30)); dr = ImageDraw.Draw(im)
    dr.rectangle([P_(-F.H, F.H), P_(F.H, -F.H)], outline=(230, 230, 230), width=3)
    for item in paths:
        xy, yw, cols, foot = item[:4]
        if len(item) > 4 and len(xy):                            # optional label at the path's current end
            dr.text((P_(*xy[-1])[0] + 6, P_(*xy[-1])[1] - 6), item[4], fill=(255, 255, 255))
        for k in range(1, len(xy)):
            dr.line([P_(*xy[k - 1]), P_(*xy[k])], fill=cols[k], width=2)
        if foot and len(xy):
            fp = F.footprint(xy[-1], yw[-1])                     # corners in order: front-L, front-R, back-L, back-R
            dr.polygon([P_(*fp[0]), P_(*fp[1]), P_(*fp[3]), P_(*fp[2])], outline=cols[-1], width=2)
            c, s_ = np.cos(yw[-1]), np.sin(yw[-1])               # heading arrow = camera direction
            dr.line([P_(*xy[-1]), P_(xy[-1][0] + 0.8 * c, xy[-1][1] + 0.8 * s_)], fill=(255, 255, 255), width=2)
    if title:
        dr.text((4, 4), title, fill=(255, 255, 0))
    return np.asarray(im)


PHASE_COL = dict(orient=(150, 150, 150), approach=(80, 140, 255), room=(255, 80, 80), turnout=(255, 160, 40),
                 walkout=(80, 220, 80))
STATUS_COL = ((80, 255, 80), (255, 230, 0), (255, 60, 60))


def do_allo(a):
    """ALLOCENTRIC cross-check videos (no re-rendering: drawn from the stored physics). Per episode: camera (gray, as
    the model sees it) | whole-room top-down map (path by phase, footprint + heading now) | zoomed map. Per branch-grid
    state: the 6 commands' camera views | one map with all 6 paths from the same start, coloured per frame by
    safe / near / crossed."""
    import glob
    import imageio.v2 as imageio
    from PIL import Image
    D = os.path.join(DATA, a.split); out = os.path.join(CHK, f"review_{a.split}", "allocentric"); os.makedirs(out, exist_ok=True)
    names = cmd_names()
    gray = lambda f: np.repeat(np.asarray(Image.fromarray(f).convert("L"))[..., None], 3, -1)   # noqa: E731
    for p in sorted(x for x in glob.glob(os.path.join(D, "ep*.npz")) if "_b" not in os.path.basename(x))[:a.n]:
        d = np.load(p, allow_pickle=True); e = int(d["episode"]); xy = d["base_pos"][:, :2].astype(float)
        yw = np.array([F.yaw_of(q) for q in d["base_quat"]]); ph = [str(x) for x in d["phase"]]
        st = frame_status(d["clearance"])
        wr = imageio.get_writer(os.path.join(out, f"episode_{e}_allo.mp4"), fps=20)
        for j in range(len(xy)):
            cols = [PHASE_COL.get(x, (200, 200, 200)) for x in ph[:j + 1]]
            lab = f"t={j * .05:.2f}s {ph[j]} {names[int(d['cmd_index'][j])]} clear {d['clearance'][j]:+.2f} {STATUS_NAMES[st[j]].upper()}"
            big = _map([(xy[:j + 1], yw[:j + 1], cols, True)], 384, title=f"ep{e} room (top-down)")
            zoom = _map([(xy[:j + 1], yw[:j + 1], cols, True)], 384, view=(xy[j][0], xy[j][1], 2.5), title="zoom 5 m")
            cam = _tile(gray(d["frames"][j]), lab, STATUS_COL[st[j]])
            wr.append_data(np.hstack([np.asarray(Image.fromarray(cam).resize((384, 384))), big, zoom]))
        wr.close(); print("->", os.path.relpath(os.path.join(out, f"episode_{e}_allo.mp4"), ROOT), flush=True)
    # branch grids: same three states as `branchgrid`
    show = (2, 6, 10, 18, 22, 25)
    info = {}
    for f in sorted(glob.glob(os.path.join(D, "ep*_b*_c00.npz"))):
        x = np.load(f, allow_pickle=True); bp = x["base_pos"][PREFIX - 1]; y0 = F.yaw_of(x["base_quat"][PREFIX - 1])
        dw = [F.H - bp[0], F.H + bp[0], F.H - bp[1], F.H + bp[1]]; w = int(np.argmin(dw))
        ang = abs(np.degrees(np.angle(np.exp(1j * (y0 - (0, np.pi, np.pi / 2, -np.pi / 2)[w])))))
        info[f[:-8]] = (float(x["start_clearance"]), ang, sorted(dw)[1] < 1.5 + F.FOOT_L / 2)
    pick = {"head-on_close": min((k for k in info if info[k][1] < 30), key=lambda k: info[k][0], default=None),
            "back_to_wall": max(info, key=lambda k: info[k][1]) if info else None,
            "corner": min((k for k in info if info[k][2]), key=lambda k: info[k][0], default=None)}
    for lab, k in pick.items():
        if k is None:
            continue
        D_ = [np.load(f"{k}_c{c:02d}.npz", allow_pickle=True) for c in show]
        n = max(len(d["frames"]) for d in D_); x0 = D_[0]["base_pos"][PREFIX - 1][:2]
        wr = imageio.get_writer(os.path.join(out, f"branchgrid_{lab}_allo.mp4"), fps=10)
        for t in range(n):
            tiles, paths = [], []
            for ci, (c, d) in enumerate(zip(show, D_)):
                q = min(t, len(d["frames"]) - 1); s_ = frame_status(d["clearance"])
                xy = d["base_pos"][PREFIX - 1:q + 1, :2].astype(float); yw = np.array([F.yaw_of(z) for z in d["base_quat"][PREFIX - 1:q + 1]])
                if len(xy):
                    paths.append((xy, yw, [STATUS_COL[v] for v in s_[PREFIX - 1:q + 1]], True, names[c].replace("_w0.45", "")))
                txt = f"{names[c]} {(q - PREFIX + 1) * .05:+.2f}s {d['clearance'][q]:+.2f}m {STATUS_NAMES[s_[q]].upper()}" + (" (stopped)" if t > q else "")
                tiles.append(_tile(gray(d["frames"][q]), txt, STATUS_COL[s_[q]]))
            m = _map(paths, 384, view=(x0[0], x0[1], 2.5), title=f"{lab}: 6 commands from one state (top-down, 5 m)")
            leg = np.zeros((384, 192, 3), np.uint8); im = Image.fromarray(leg); from PIL import ImageDraw
            dr = ImageDraw.Draw(im)
            for ci, c in enumerate(show):
                dr.text((4, 10 + 22 * ci), names[c], fill=(255, 255, 255))
            dr.text((4, 160), "path colour per frame:", fill=(255, 255, 255))
            for v, nm in enumerate(STATUS_NAMES):
                dr.text((4, 182 + 20 * v), nm.upper(), fill=STATUS_COL[v])
            grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
            wr.append_data(np.hstack([grid, m, np.asarray(im)]))
        wr.close(); print("->", os.path.relpath(os.path.join(out, f"branchgrid_{lab}_allo.mp4"), ROOT), flush=True)


def _tile(f, t, col):
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(f)).resize((192, 192)); ImageDraw.Draw(im).text((3, 3), t, fill=col)
    return np.asarray(im)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("pilot", "physics", "render", "review", "stats", "sep", "branchgrid", "allo"))
    ap.add_argument("--port", type=int, default=25750)
    ap.add_argument("--ports", type=int, nargs="+", default=[25780])
    ap.add_argument("--split", default="pilot10")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ep0", type=int, default=7000)
    ap.add_argument("--look0", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--h", type=float, default=1.25)
    a = ap.parse_args()
    {"pilot": do_pilot, "physics": do_physics, "render": do_render, "review": do_review, "stats": do_stats, "sep": do_sep, "branchgrid": do_branchgrid, "allo": do_allo}[a.cmd](a)
