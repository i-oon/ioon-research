"""B1 virtual-boundary feasibility pilot (CPU only, MuJoCo, no rendering, no model), 2026-10-09.

A FILTER, not evidence for the world model: from paired start states near a wall or a corner, can any controller that
chooses among B1's 24 tuned commands avoid the boundary while walking forward, does the choice matter, and is there a gap
between a true-future oracle and a well-tuned privileged time-to-collision (TTC) rule -- per scenario?

World: the ORIGINAL B1 room (17.65 m, walls at +-H around the origin, H = 8.8 m); MuJoCo has no walls -> "virtual boundary".
Footprint: B1 trunk rectangle FOOT_L x FOOT_W centred on the base; clearance = min over its corners of the distance to the
nearest wall plane (negative = crossed).
Execution rules, IDENTICAL for every controller: the same 24 commands (`build_branches.b1_cmds`, as the walks ran them), the
same heading-PI runtime (`build_b1_cf_branches.Roll`, gains (2.5, 1.0)), a decision every DEC policy steps (0.1 s = 2 frames,
the planners' cadence), the chosen command held until the next decision, episode EP_STEPS (10 s), stop at a crossing.
Controllers:
  direct    always the goal command (no view, no sensor) -- lower baseline
  ttc(tau)  PRIVILEGED distance: front distance d (ray from the footprint front along the heading to the wall) and current
            forward speed v; if d / v < tau (or d < 0.5 m) -> avoid mode: strongest turn toward the side with more room,
            until d / v > tau + 1 s; else the goal command. tau tuned on the VAL starts (grid TAUS), reported on TEST.
  oracle(H) TRUE future: at each decision every command is simulated H steps from the exact current state (restore); a
            command is safe if its footprint clearance stays >= MARGIN; among safe ones the one whose mean forward speed is
            closest to the goal speed; if none is safe, the one with the largest minimum clearance. H in HORIZONS.
Scenarios (paired: the same start state for every controller): wall (front wall at distance d, heading theta from the wall
normal) and corner (also a side wall 1.5 m to the left); d in DISTS, theta in THETAS, goal speed in SPEEDS (forward commands
of B1's library); val / test differ by a start jitter (position +-0.3 m, extra gait steps).
Metrics per (scenario, controller): crossed (yes/no), minimum clearance, progress = mean forward speed / goal speed, and
whether any command avoids the boundary for the whole episode when held (avoidable) / how many of the 24 do (choice matters).

    .venv/bin/python3 scripts/diagnostics/wall_pilot/b1_wall_feasibility.py --workers 6
    -> results/check/wall_feasibility/{episodes.json, summary.txt}
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("scripts/dataset", "sim/control", "sim/collect", "."):
    sys.path.insert(0, os.path.join(ROOT, p))

OUT = os.path.join(ROOT, "results/check/wall_feasibility_v2")
# v2 (2026-10-09, after v1 in results/check/wall_feasibility/): TTC uses the signed clearance of the WHOLE footprint and its
# rate of change (kinematic extrapolation over 0.1 s from the current base velocity and yaw rate), not the distance along
# the heading (v1 crossed only on oblique approaches); oracle uses the goal command whenever its true future is safe over H,
# else the safe command with the highest mean forward speed, else the largest minimum clearance.
H = 8.8                      # half room, original B1 room 17.65 m (inner wall face ~8.80)
FOOT_L, FOOT_W = 1.0, 0.5    # B1 trunk footprint (m)
MARGIN = 0.15                # oracle safety margin (m)
DEC = 5                      # policy steps per decision (5 x 20 ms = 0.1 s = 2 frames)
EP_STEPS = 500               # 10 s
HORIZONS = (25, 50, 100)     # oracle look-ahead: 0.5 / 1 / 2 s (no longer: the FTM is not shown to predict further)
TAUS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BACKS = (0.0, 0.5, 1.0)
BACKS2 = (0.0, 0.1, 0.2, 0.3)  # v2: back up when the footprint clearance < this (m)      # ttc: back up (weakest backward command) when the front distance < this (m); tuned on val
TURN_L, TURN_R, BACK = 11, 15, 4   # strongest left / right turn, weakest backward (B1 library order)
DISTS = (1.0, 1.5, 2.0, 3.0)
THETAS = (0.0, 30.0, 60.0)
FWD = (0, 1, 2, 3)           # command indices of the 4 forward speed levels (B1 library order)
SPEEDS = (1, 3)              # goal = forward level 1 / 3


def runtime():
    from build_b1_cf_branches import Roll
    import torch
    torch.set_num_threads(1)
    return Roll("gait3", (2.5, 1.0), os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat.xml"))


def commands():
    from build_branches import b1_cmds, FAMILY
    return b1_cmds(), list(FAMILY)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def footprint(pos, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for a in (FOOT_L / 2, -FOOT_L / 2):
        for b in (FOOT_W / 2, -FOOT_W / 2):
            pts.append((pos[0] + c * a - s * b, pos[1] + s * a + c * b))
    return np.asarray(pts)


def clearance(pos, yaw, corner):
    p = footprint(pos, yaw)
    d = np.minimum(H - np.abs(p[:, 0]), H - np.abs(p[:, 1]))
    return float(d.min())


def front_dist(pos, yaw):
    """distance along the heading from the footprint front-centre to the first wall plane"""
    c, s = np.cos(yaw), np.sin(yaw)
    fx, fy = pos[0] + c * FOOT_L / 2, pos[1] + s * FOOT_L / 2
    ts = []
    for v, w in ((c, fx), (s, fy)):
        if v > 1e-6:
            ts.append((H - w) / v)
        elif v < -1e-6:
            ts.append((-H - w) / v)
    return float(min(ts)) if ts else 1e9


def side_room(pos, yaw):
    """free distance to the left / right (perpendicular rays) from the base"""
    out = []
    for sgn in (1, -1):
        a = yaw + sgn * np.pi / 2
        out.append(front_dist(pos - np.array([np.cos(yaw), np.sin(yaw)]) * FOOT_L / 2, a))
    return out


def fwd_speed(r):
    q = r.d.qpos[3:7]
    yaw = yaw_of(q)
    v = r.d.qvel[0:2]
    return float(np.cos(yaw) * v[0] + np.sin(yaw) * v[1])


def start_state(r, cmds, goal_i, x, y, yaw, extra):
    """walk the goal command at the origin until steady, then place the robot at (x, y, yaw); returns a snapshot."""
    import mujoco
    from build_b1_cf_branches import POLICY_WARMUP
    r2 = runtime()
    for _ in range(POLICY_WARMUP + 100 + extra):
        r2.step(*cmds[goal_i])
    d = r2.d
    yaw0 = yaw_of(d.qpos[3:7])
    dpsi = yaw - yaw0
    cz, sz = np.cos(dpsi / 2), np.sin(dpsi / 2)
    w, qx, qy, qz = d.qpos[3:7]
    d.qpos[3:7] = [cz * w - sz * qz, cz * qx - sz * qy, cz * qy + sz * qx, cz * qz + sz * w]
    Rz = np.array([[np.cos(dpsi), -np.sin(dpsi), 0], [np.sin(dpsi), np.cos(dpsi), 0], [0, 0, 1]])
    d.qvel[0:3] = Rz @ d.qvel[0:3]
    d.qpos[0:2] = [x, y]
    mujoco.mj_forward(r2.m, d)
    r2.heading_target = (r2.heading_target or yaw0) + dpsi
    return r2, r2.snapshot()


def run(r, snap, policy, max_steps=EP_STEPS, log=False):
    """policy(r, k) -> command INDEX. With log: every decision (k, index, front distance, TTC, clearance, x, y, yaw)."""
    cmds, _ = commands()
    r.restore(snap)
    minc, vs, crossed, n = 1e9, [], False, 0
    cmd, dec = None, []
    for k in range(max_steps):
        if k % DEC == 0:
            i = policy(r, k)
            cmd = cmds[i]
            if log:
                pos, yw = r.d.qpos[0:2].copy(), yaw_of(r.d.qpos[3:7])
                df, v = front_dist(pos, yw), fwd_speed(r)
                dec.append([k, i, round(df, 3), round(df / max(v, 1e-3), 3), round(clearance(pos, yw, None), 3),
                            round(float(pos[0]), 3), round(float(pos[1]), 3), round(yw, 4)])
        r.step(*cmd)
        pos, yaw = r.d.qpos[0:2].copy(), yaw_of(r.d.qpos[3:7])
        c = clearance(pos, yaw, None)
        minc = min(minc, c); vs.append(fwd_speed(r)); n += 1
        if c < 0:
            crossed = True
            break
    out = dict(crossed=crossed, min_clear=minc, mean_v=float(np.mean(vs)), steps=n)
    if log:
        out["decisions"] = dec
    return out


def scenario_job(job):
    kind, d, th, sp, split = job
    cmds, fam = commands()
    goal_i = FWD[sp]
    import zlib
    rng = np.random.default_rng(zlib.crc32(repr((kind, d, th, sp, split)).encode()))
    jit = rng.uniform(-0.3, 0.3, 2) if split == "test" else np.zeros(2)
    extra = int(rng.integers(0, 12)) if split == "test" else 0
    # front wall = +x; base placed so the footprint front-centre is d from it along +x for theta = 0
    x = H - d - FOOT_L / 2 + jit[0]
    y = (H - 1.5 - FOOT_W / 2 + jit[1]) if kind == "corner" else jit[1]
    yaw = np.deg2rad(th)                      # turning toward +y (the corner side) for positive theta
    r, snap = start_state(None, cmds, goal_i, x, y, yaw, extra)
    t0 = time.time()
    # goal speed: steady forward speed of the goal command (measured on the warm-up state)
    r.restore(snap); vg = fwd_speed(r)
    res = {"job": job, "goal_v": vg}
    # avoidability: each of the 24 commands held for the whole episode
    held = [run(r, snap, lambda rr, k, i=i: i) for i in range(len(cmds))]
    res["held_safe"] = [not h["crossed"] for h in held]
    res["direct"] = held[goal_i]

    def ttc_policy(tau, back):
        state = {"avoid": False, "turn": None}

        def pol(rr, k):
            pos, yw = rr.d.qpos[0:2], yaw_of(rr.d.qpos[3:7])
            df, v = front_dist(pos, yw), max(fwd_speed(rr), 1e-3)
            ttc = df / v
            if df < back:
                return BACK
            if state["avoid"] and ttc > tau + 1.0:
                state["avoid"] = False
            if not state["avoid"] and ttc < tau:
                left, right = side_room(np.asarray(pos), yw)
                state["avoid"], state["turn"] = True, (TURN_L if left >= right else TURN_R)
            return state["turn"] if state["avoid"] else goal_i
        return pol
    def ttc2_policy(tau, back):
        state = {"avoid": False, "turn": None}

        def pol(rr, k):
            pos, yw = rr.d.qpos[0:2].copy(), yaw_of(rr.d.qpos[3:7])
            c = clearance(pos, yw, None)
            dt = 0.1
            c_next = clearance(pos + rr.d.qvel[0:2] * dt, yw + float(rr.d.qvel[5]) * dt, None)
            rate = (c_next - c) / dt
            ttb = c / -rate if rate < -1e-4 else 1e9          # time to the boundary at the current closing rate
            if c < back:
                return BACK
            if state["avoid"] and ttb > tau + 1.0:
                state["avoid"] = False
            if not state["avoid"] and ttb < tau:
                left, right = side_room(pos, yw)
                state["avoid"], state["turn"] = True, (TURN_L if left >= right else TURN_R)
            return state["turn"] if state["avoid"] else goal_i
        return pol
    res["ttc"] = {f"{tau}|{b}": run(r, snap, ttc2_policy(tau, b), log=True) for tau in TAUS for b in BACKS2}

    def oracle_policy(h):
        def pol(rr, k):
            now = rr.snapshot()
            best, best_key = None, None
            for i, c in enumerate(cmds):
                rr.restore(now)
                mc, vs = 1e9, []
                for _ in range(h):
                    rr.step(*c)
                    mc = min(mc, clearance(rr.d.qpos[0:2], yaw_of(rr.d.qpos[3:7]), None)); vs.append(fwd_speed(rr))
                safe = mc >= MARGIN
                # v2: the goal command first when safe; else the safe command with the highest forward speed
                key = ((0, 0.0) if i == goal_i else (1, -np.mean(vs))) if safe else (2, -mc)
                if best_key is None or key < best_key:
                    best, best_key = i, key
            rr.restore(now)
            return best
        return pol
    res["oracle"] = {str(h): run(r, snap, oracle_policy(h), log=True) for h in HORIZONS}
    res["secs"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cmds, fam = commands()
    assert all(fam[i] == "fwd" for i in FWD), fam
    assert fam[TURN_L] == "turn_left" and fam[TURN_R] == "turn_right" and fam[BACK] == "bwd", fam
    jobs = [(k, d, th, sp, s) for s in ("val", "test") for k in ("wall", "corner") for d in DISTS for th in THETAS
            for sp in range(len(SPEEDS))]
    if a.limit:
        jobs = jobs[:a.limit]
    os.makedirs(OUT, exist_ok=True)
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(scenario_job, jobs)):
            res.append(r)
            print(f"{i + 1}/{len(jobs)} {r['job']} {r['secs']:.0f}s held-safe {sum(r['held_safe'])}/24 "
                  f"direct crossed {r['direct']['crossed']}", flush=True)
    json.dump(res, open(os.path.join(OUT, "episodes.json"), "w"), indent=1, default=float)
    summarise(res)


def summarise(res):
    val = [r for r in res if r["job"][4] == "val"]
    test = [r for r in res if r["job"][4] == "test"]

    def score(x, vg):            # crossing = fail; else progress (mean forward speed / goal)
        return 0.0 if x["crossed"] else max(0.0, x["mean_v"] / max(vg, 1e-3))
    best_tau = max(val[0]["ttc"], key=lambda t: np.mean([score(r["ttc"][t], r["goal_v"]) for r in val]))
    lines = [f"TTC (tau s | back-up m) tuned on {len(val)} val starts: {best_tau}; report on {len(test)} test starts",
             "per scenario (test): kind d theta goal | held-safe /24 | crossed: direct ttc oracle0.5/1/2 | progress same | "
             "min clearance same"]
    for r in sorted(test, key=lambda r: r["job"]):
        k, d, th, sp, _ = r["job"]
        rows = [r["direct"], r["ttc"][best_tau]] + [r["oracle"][str(h)] for h in HORIZONS]
        lines.append(f"{k:6s} d={d:.1f} th={th:4.0f} v{SPEEDS[sp]} | {sum(r['held_safe']):2d} | "
                     + " ".join("X" if x["crossed"] else "." for x in rows) + " | "
                     + " ".join(f"{score(x, r['goal_v']):.2f}" for x in rows) + " | "
                     + " ".join(f"{x['min_clear']:+.2f}" for x in rows))
    for name, get in (("direct", lambda r: r["direct"]), (f"ttc({best_tau})", lambda r: r["ttc"][best_tau])) + tuple(
            (f"oracle({h * 0.02:.1f}s)", (lambda r, h=h: r["oracle"][str(h)])) for h in HORIZONS):
        xs = [get(r) for r in test]
        lines.append(f"TOTAL {name:14s} crossed {np.mean([x['crossed'] for x in xs]):.2f}  score "
                     f"{np.mean([score(x, r['goal_v']) for x, r in zip(xs, test)]):.2f}  min clearance "
                     f"{np.median([x['min_clear'] for x in xs]):+.2f}")
    lines.append("")
    lines.append("oracle v2 = ONE-ACTION receding-horizon oracle: each decision simulates every command HELD for H from the exact")
    lines.append("state, safe = footprint min clearance >= %.2f m over H; the goal command if safe, else the safe command with the" % MARGIN)
    lines.append("highest mean forward speed, else the largest min clearance; not the optimal 10 s sequence. TTC v2 = time to the")
    lines.append("boundary from the footprint's signed clearance and its closing rate. Score: crossed = 0, else progress =")
    lines.append("mean forward speed / goal, floored at 0 (backing away is safe but scores ~0).")
    lines.append("")
    lines.append("TTC vs oracle(1 s), test, per scenario: first decision where they choose differently (time, front distance")
    lines.append("in each run), share of differing decisions over the shorter run, outcome (crossed, score) TTC -> oracle")
    for r in sorted(test, key=lambda r: r["job"]):
        k, d, th, sp, _ = r["job"]
        a_, b_ = r["ttc"][best_tau], r["oracle"]["50"]
        A, B = a_["decisions"], b_["decisions"]
        n = min(len(A), len(B))
        diff = [j for j in range(n) if A[j][1] != B[j][1]]
        first = (f"t={A[diff[0]][0] * 0.02:4.1f}s dfront ttc {A[diff[0]][2]:.2f} / oracle {B[diff[0]][2]:.2f} m"
                 if diff else "never")
        lines.append(f"{k:6s} d={d:.1f} th={th:4.0f} v{SPEEDS[sp]} | first differ {first} | differ "
                     f"{len(diff) / max(n, 1):.2f} | crossed {int(a_['crossed'])}->{int(b_['crossed'])} score "
                     f"{score(a_, r['goal_v']):.2f}->{score(b_, r['goal_v']):.2f}")
    V1_CROSSED = {("wall", 1.0, 30.0, 0), ("wall", 1.0, 60.0, 0), ("wall", 1.0, 60.0, 1), ("wall", 1.5, 30.0, 0),
                  ("wall", 1.5, 30.0, 1), ("wall", 1.5, 60.0, 1)}       # the 6 test starts TTC v1 crossed
    lines.append("")
    lines.append("paired cases (the 6 test starts TTC v1 crossed): controller | turn starts (first non-goal command) | "
                 "commands used (index: share) | min clearance | progress | crossed")
    import collections
    for r in sorted(test, key=lambda r: r["job"]):
        k, d, th, sp, _ = r["job"]
        if (k, d, th, sp) not in V1_CROSSED:
            continue
        lines.append(f"{k} d={d} th={th} v{SPEEDS[sp]} (goal command {FWD[sp]})")
        for name, x in (("ttc v2", r["ttc"][best_tau]), ("oracle 0.5s", r["oracle"]["25"]), ("oracle 1s", r["oracle"]["50"])):
            dec = x["decisions"]
            g = FWD[sp]
            first = next((dd[0] * 0.02 for dd in dec if dd[1] != g), None)
            cnt = collections.Counter(dd[1] for dd in dec)
            used = ", ".join(f"{i}: {c / len(dec):.2f}" for i, c in cnt.most_common(4))
            lines.append(f"   {name:11s} | turn {('%.1f s' % first) if first is not None else 'never'} | {used} | "
                         f"{x['min_clear']:+.2f} | {score(x, r['goal_v']):.2f} | {int(x['crossed'])}")
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
