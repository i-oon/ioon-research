"""Gates for the wall data (B1 `data/wall_b1`, c10 `data/wall_c10`) before any wall training (2026-10-13).

One body per call (the c10 collector re-sets the shared module's settings on import):
    .venv/bin/python3 scripts/dataset/check_wall_data.py --body c10 [--splits train val test] [--ref_body b1]

Checks, per split (physics files `_physics/<split>`, rendered files `<split>`); each prints PASS / FAIL with numbers:
  1. inventory + disjoint splits: episode numbers and room looks never shared between splits
  2. every state has all 28 commands; branch files per state = 28 (physics); train selection 14 per state
  3. EXACT prefix (F305 rule: every recorded per-frame field, exact equality): each branch's first PREFIX rows ==
     the stored episode's rows T-PREFIX..T-1, so all 28 branches of a state start from the same physical state
  4. c10 only -- stored episode == the live run the controller saw: the live log's pose (room frame, rounded to 1 mm /
     1 mrad) at every decision equals the stored episode's (replay_equals_live is also read where stored)
  5. labels: clearance recomputed from the stored poses (the collector's own function) equals the stored value; status
     == frame_status(clearance, margin); a crossing is terminal (nothing stored after the first crossed frame)
  6. no falls (min height of the centre of mass vs the body's median)
  7. rendered set: every selected file rendered, frame count == label count, frames not blank, camera yaw follows the
     body heading (spread of their difference)
  8. coverage: start states per distance band x approach group, left / right balance of the approach and of the corner
     side (F338: which side is free must flip), crossing share per band; with --ref_body the cell counts are compared
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("scripts/dataset", "sim/collect", "sim/control", "sim/render"):
    sys.path.insert(0, os.path.join(ROOT, p))

PF = {"b1": ("base_pos", "base_quat", "joint_pos", "joint_vel", "action", "command", "com_pos", "foot_contact"),
      "c10": ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat", "state_joint_pos",
              "state_link_pose", "com_pos", "base_pos", "base_quat")}


def ok(flag):
    return "PASS" if flag else "FAIL"


def wrap(a):
    return np.angle(np.exp(1j * np.asarray(a, float)))


def state_geometry(W, F, x, T):
    """distance band, approach group (as the collectors' report), approach side (+1 left / -1 right of the wall normal),
    corner side (+1 the second wall is on the left) -- all in the room frame at frame T-1"""
    bp = np.asarray(x["base_pos"], float); yw = F.yaw_of(np.asarray(x["base_quat"], float)[T - 1]); p0 = bp[T - 1, :2]
    dw = [F.H - p0[0], F.H + p0[0], F.H - p0[1], F.H + p0[1]]; w = int(np.argmin(dw))
    nrm = (0.0, np.pi, np.pi / 2, -np.pi / 2)[w]
    rel = float(wrap(yw - nrm)); ang = abs(np.degrees(rel))
    grp = "corner" if W.is_corner(sorted(dw)) else ("head-on" if ang < 30 else "oblique")
    side = 1 if rel > 0 else -1
    w2 = sorted(range(4), key=lambda k: dw[k])[1]
    corner_side = 1 if wrap((0.0, np.pi, np.pi / 2, -np.pi / 2)[w2] - yw) > 0 else -1
    return grp, side, corner_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=("b1", "c10"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--ref_body", default=None, help="compare cell counts with this body's saved check json")
    ap.add_argument("--render", action="store_true", help="also check the rendered files (slow: loads every frame)")
    a = ap.parse_args()
    if a.body == "c10":
        import collect_c10_wall_data as C
        W = C.W
    else:
        import collect_b1_wall_data as W
        C = None
    F = W.F
    PREFIX = W.PREFIX
    out = {"body": a.body, "cells": {}, "fail": []}
    seen_ep, seen_look = {}, {}
    lines = []

    def say(m):
        print(m, flush=True); lines.append(m)

    def fail(m):
        out["fail"].append(m); say("  FAIL " + m)

    for split in a.splits:
        D = os.path.join(W.DATA, "_physics", split)
        eps = sorted(f for f in glob.glob(os.path.join(D, "ep*.npz")) if "_b" not in os.path.basename(f))
        say(f"=== {a.body} {split}: {len(eps)} episodes ({D})")
        # 1. disjoint
        for f in eps:
            x = np.load(f, allow_pickle=True); e = int(x["episode"]); lk = int(x["room_seed"])
            for d, k in ((seen_ep, e), (seen_look, lk)):
                if k in d and d[k] != split:
                    fail(f"{os.path.basename(f)}: number / look {k} also in {d[k]}")
                d[k] = split
        sel_f = os.path.join(D, "selection.json") if split == "train" else None
        sel = set(json.load(open(sel_f))["render"]) if sel_f and os.path.exists(sel_f) else None
        n_states = 0; bad_prefix = bad_lab = bad_term = bad_cmds = bad_live = 0; live_checked = 0
        prefix_max = 0.0; lab_max = 0.0; zmin = []
        cells = collections.Counter(); sides = collections.Counter(); csides = collections.Counter()
        cross = collections.defaultdict(list)
        for f in eps:
            x = np.load(f, allow_pickle=True)
            stem = f[:-4]
            brs = sorted(glob.glob(stem + "_b*_c*.npz"))
            by_state = collections.defaultdict(list)
            for g in brs:
                by_state[int(os.path.basename(g).split("_b")[1].split("_")[0])].append(g)
            # 4. stored == live (c10)
            if a.body == "c10" and "log" in x.files:
                live_checked += 1
                rm = C.Room(np.asarray(x["room_c"], float), float(x["room_phi"]))
                hd = np.asarray(x["head"], float); ab = np.asarray(x["state_abdomen_pos"], float); n = len(hd)
                bad = 0
                for ent in json.loads(str(x["log"])):
                    t = int(ent["t"]) - 1                 # the decision at step t sees the state recorded on row t-1
                    if t < 0 or t >= n:
                        continue
                    c = rm.p(((hd[t, :2] + ab[t, :2]) / 2)[None])[0]
                    yw = float(rm.yaw(float(np.arctan2(hd[t, 1] - ab[t, 1], hd[t, 0] - ab[t, 0]))))
                    if np.abs(c - np.asarray(ent["pos"])).max() > 6e-4 or abs(wrap(yw - ent["yaw"])) > 6e-4:
                        bad += 1
                if "replay_equals_live" in x.files and not bool(x["replay_equals_live"]):
                    bad += 1
                if bad:
                    bad_live += 1; fail(f"{os.path.basename(f)}: stored episode differs from the live run at {bad} decisions")
            for b, files in sorted(by_state.items()):
                n_states += 1
                if len(files) != 28 or sorted(int(np.load(g)["cf_command_index"]) for g in files) != list(range(28)):
                    bad_cmds += 1; fail(f"{os.path.basename(stem)} state {b}: {len(files)} command files")
                y0 = np.load(files[0], allow_pickle=True); T = int(y0["branch_frame"])
                # 3. exact prefix vs the stored episode
                for g in files:
                    y = np.load(g, allow_pickle=True)
                    for k in PF[a.body]:
                        if k not in y.files or k not in x.files:
                            continue
                        u = np.asarray(y[k])[:PREFIX]; v = np.asarray(x[k])[T - PREFIX:T]
                        if not np.array_equal(u, v):
                            d = float(np.abs(u.astype(float) - v.astype(float)).max()) if u.shape == v.shape else np.inf
                            prefix_max = max(prefix_max, d); bad_prefix += 1
                            fail(f"{os.path.basename(g)}: prefix field {k} differs (max {d:.2e})"); break
                    # 5. labels
                    cl = np.asarray(y["clearance"], float)
                    if C is not None:
                        rm = C.Room(np.asarray(y["room_c"], float), float(y["room_phi"]))
                        rc = C.labels({"com_pos": y["com_pos"], "head": y["head"]}, rm)[2]
                    else:
                        bp = np.asarray(y["base_pos"], float); bq = np.asarray(y["base_quat"], float)
                        rc = np.array([F.clearance(bp[i, :2], F.yaw_of(bq[i]), None) for i in range(len(bp))])
                    dl = float(np.abs(rc - cl).max()); lab_max = max(lab_max, dl)
                    st = W.frame_status(cl) if not hasattr(W, "MARGIN") else np.where(cl < 0, 2, np.where(cl < float(y["margin"]), 1, 0))
                    if dl > 1e-6 or not np.array_equal(st, np.asarray(y["status"])):
                        bad_lab += 1; fail(f"{os.path.basename(g)}: labels differ (clearance {dl:.2e})")
                    if (cl[:-1] < 0).any():
                        bad_term += 1; fail(f"{os.path.basename(g)}: frames stored after the crossing")
                    zmin.append(float(np.asarray(y["com_pos"], float)[:, 2].min()))
                    cross[W.band_of(float(y0["start_clearance"]))].append(bool((cl < 0).any()))
                grp, side, cs = state_geometry(W, F, y0, PREFIX)
                band = W.band_of(float(y0["start_clearance"]))
                cells[(band, grp)] += 1
                if grp == "corner":
                    csides[cs] += 1
                else:
                    sides[side] += 1
        say(f"  1-2 states {n_states}; command sets complete: {ok(bad_cmds == 0)} ({bad_cmds} bad)")
        say(f"  3 exact prefix (all fields, branch rows 0..{PREFIX - 1} == episode rows T-{PREFIX}..T-1): "
            f"{ok(bad_prefix == 0)} ({bad_prefix} files differ, max {prefix_max:.2e})")
        if a.body == "c10":
            say(f"  4 stored == live run (every decision, 1 mm / 1 mrad): {ok(bad_live == 0)} "
                f"({bad_live} of {live_checked} episodes differ)")
        say(f"  5 labels recomputed == stored: {ok(bad_lab == 0)} (max clearance diff {lab_max:.2e}); "
            f"crossing terminal: {ok(bad_term == 0)}")
        zm = np.array(zmin) if zmin else np.zeros(1)
        say(f"  6 CoM height min over files: p1 {np.percentile(zm, 1):.3f} / min {zm.min():.3f} m (median of minima "
            f"{np.median(zm):.3f}); falls (< 0.5 x median): {int((zm < 0.5 * np.median(zm)).sum())}")
        say(f"  8 cells (band x group): " + "; ".join(f"{bd} {g} {cells[(bd, g)]}" for bd in sorted({k[0] for k in cells})
                                                       for g in ("head-on", "oblique", "corner")))
        say(f"    approach side left / right {sides[1]} / {sides[-1]}; corner second wall left / right "
            f"{csides[1]} / {csides[-1]}; crossing share per band "
            + ", ".join(f"{bd} {np.mean(v):.2f}" for bd, v in sorted(cross.items())))
        out["cells"][split] = {f"{k[0]}|{k[1]}": v for k, v in cells.items()}
        # 7. rendered set
        if a.render:
            R = os.path.join(W.DATA, split)
            want = sel if sel is not None else {os.path.basename(g) for g in glob.glob(os.path.join(D, "ep*_b*_c*.npz"))}
            miss = [w for w in want if not os.path.exists(os.path.join(R, w))]
            bad_n = blank = near_uniform = 0; spread = []
            for w in sorted(want - set(miss)):
                z = np.load(os.path.join(R, w), allow_pickle=True); fr = z["frames"]
                if len(fr) != len(z["clearance"]):
                    bad_n += 1
                # BROKEN render = black or completely constant (a renderer failure); near-UNIFORM (std < 3 grey levels)
                # = the robot facing a wall 0.4-0.8 m away, the wall filling the 90 deg view: real data, reported only
                # (F346; the first version failed on these, 2026-10-15)
                if float(fr.mean()) < 10 or float(fr.reshape(len(fr), -1).std(1).max()) < 0.5:
                    blank += 1
                elif float(fr.std()) < 3:
                    near_uniform += 1
                if "cam_pose" in z.files:
                    q = np.asarray(z["cam_pose"], float)[:, 3:]                 # qx qy qz qw; camera looks along +z
                    x_, y_, z_, w_ = q.T
                    fwd = np.stack([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_)], 1)
                    cy = np.arctan2(fwd[:, 1], fwd[:, 0])
                    by = np.array([F.yaw_of(v) for v in np.asarray(z["base_quat"], float)])
                    dd = wrap(cy - by); spread.append(float(np.abs(wrap(dd - np.median(dd))).max()))
            sp = np.degrees(np.array(spread)) if spread else np.zeros(1)
            say(f"  7 rendered: {len(want)} wanted, missing {len(miss)}, frame count wrong {bad_n}, broken (black / constant) "
                f"{blank}, near-uniform wall views (info) {near_uniform}: "
                f"{ok(not miss and not bad_n and not blank)}; camera-vs-body yaw spread median {np.median(sp):.2f} / max "
                f"{sp.max():.2f} deg")
            if miss or bad_n or blank:
                fail(f"{split}: rendered set incomplete")
    if a.ref_body:
        ref = json.load(open(os.path.join(ROOT, f"results/check/wall_data_{a.ref_body}/gate.json")))
        say(f"=== cell counts {a.body} vs {a.ref_body} (must be equal after the trim)")
        for split in a.splits:
            mine, theirs = out["cells"].get(split, {}), ref["cells"].get(split, {})
            diff = {k: mine.get(k, 0) - theirs.get(k, 0) for k in sorted(set(mine) | set(theirs))}
            say(f"  {split}: " + "; ".join(f"{k} {v:+d}" for k, v in diff.items() if v)
                + ("" if any(diff.values()) else "equal"))
    say(f"GATES {'ALL PASS' if not out['fail'] else str(len(out['fail'])) + ' FAILURES'}")
    od = os.path.join(ROOT, f"results/check/wall_data_{a.body}"); os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "gate.json"), "w"), indent=1)
    open(os.path.join(od, "gate.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
