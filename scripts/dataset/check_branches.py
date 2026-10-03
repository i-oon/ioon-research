"""DATA_PLAN v2 stage 3 gates for data/counterfactual_walks/{c10,b1}_branches_{train,val,heldout}
(hexapod: collect_c10_walks_and_branches.py; B1: build_branches.py).

    .venv/bin/python3 scripts/dataset/check_branches.py [--workers 12]
        (hexapod = c10_clips_* / c10_branches_*, scene reuse, F305: prefix and own-command branch must be
        bit-identical, hard gate; B1 = b1_clips_* / b1_branches_*)
    .venv/bin/python3 scripts/dataset/check_branches.py --c08
        (c08f09t09 test set, collect_c08_test_set.py: c08_clips_heldout / c08_branches_heldout; same as
        --hex_prefix c08 --splits heldout --bodies hex)
Gates: 1 coverage, 2 rooms, 3 B1 integrity, 4 hexapod integrity / noise floor / switching effect, 5 branch gait
phases, 6 falls, 7 loader + label independence of the prefix. Prints tables; exit 1 if a hard gate fails.
"""
import argparse
import glob
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset"):
    sys.path.insert(0, os.path.join(ROOT, p))
import wm.data.embodiment as E  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
SPLITS = ("train", "val", "heldout")
FAM = ["fwd"] * 4 + ["bwd"] * 4 + ["turn_left"] * 4 + ["turn_right"] * 4 + ["side_L"] * 4 + ["side_R"] * 4
PRE = 10
NB = 31
HEXP = "c10"          # hexapod dir prefix: c10 (data/counterfactual_walks/c10_{clips,branches}_*) or c08


def cf_dir(body, split):
    return os.path.join(CW, f"{HEXP if body == 'hex' else body}_branches_{split}")


def main_sub(body):
    return "b1_clips" if body == "b1" else f"{HEXP}_clips"


def tilt_deg(q_xyzw):
    from scipy.spatial.transform import Rotation as Rt
    up = Rt.from_quat(q_xyzw).inv().apply([0, 0, 1.0])
    return np.degrees(np.arccos(np.clip(up @ up[0], -1, 1)))


def body_disp(P, yaw0):
    """xy displacement from frame PRE, in the body frame of frame PRE (removes global drift)."""
    d = P[:, :2] - P[PRE, :2]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def scan(args):
    body, split, path = args
    spec = E.HEXAPOD if body == "hex" else E.B1
    with np.load(path, allow_pickle=True) as d:
        m = {k: (d[k].item() if d[k].ndim == 0 else None) for k in d.files if d[k].ndim == 0}
        clip = E.load(path, spec)
        r = dict(path=path, body=body, split=split, **{k: m[k] for k in (
            "cf_source", "cf_source_cond_index", "cf_command_index", "cf_t", "cf_gait_phase", "room_seed", "first_pair",
            "froude_height", "cf_own") if k in m})
        r["n"] = len(d["frames"])
        r["seg_ok"] = bool(np.array_equal(d["segment"], (np.arange(NB) >= PRE).astype(np.int8)))
        r["label"] = clip["body_motion"].astype(np.float64)
        r["finite"] = bool(np.isfinite(clip["body_motion"]).all() and np.isfinite(clip["actions"]).all())
        r["frames"] = d["frames"]
        r["cam"] = d["cam_pose"]
        r["com"] = d["com_pos"].astype(np.float64)
        if body == "hex":
            r["joint"] = d["state_joint_pos"].astype(np.float64)
            r["head"] = d["head"].astype(np.float64)
            r["act"] = d["actions"]
            r["prefix_ok"] = bool(d["cf_prefix_cmd_identical"])
            r["yaw"] = E.heading(d["body_quat"].astype(np.float64), "hexapod")
            r["tilt"] = float(tilt_deg(d["state_abdomen_quat"]).max())
            r["zmin"] = float(d["com_pos"][:, 2].min())
        else:
            r["joint"] = d["joint_pos"].astype(np.float64)
            r["head"] = d["base_pos"].astype(np.float64)
            r["yaw"] = E.heading(d["base_quat"].astype(np.float64), "b1")
            q = d["base_quat"].astype(np.float64)
            r["tilt"] = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))).max())
            r["zmin"] = float(d["base_pos"][:, 2].min())
            r["fell"] = bool(d["cf_fell"])
            r["mae_t"] = float(d["cf_branch_frame_rerender_mae"])
            r["state"] = {k: d[k] for k in ("base_pos", "base_quat", "joint_pos", "action", "command")}
    return r


def source_slice(body, r):
    """The source main clip's window frames t-10 .. t+20 as a clip with the branch's segment / froude_height,
    labelled by the loader (so labels are comparable frame by frame)."""
    src = os.path.join(CW, f"{main_sub(body)}_{r['split']}", r["cf_source"])
    t = int(r["cf_t"]); a = t - PRE
    with np.load(src, allow_pickle=True) as d:
        out = {k: (d[k][a:a + NB] if (d[k].ndim > 0 and d[k].shape[0] == len(d["frames"])) else d[k]) for k in d.files}
    out["segment"] = (np.arange(NB) >= PRE).astype(np.int8)
    out["froude_height"] = np.float64(r["froude_height"])
    tmp = tempfile.mktemp(suffix=".npz")
    np.savez(tmp, **out)
    lab = E.load(tmp, E.HEXAPOD if body == "hex" else E.B1)["body_motion"].astype(np.float64)
    os.remove(tmp)
    return out, lab


def pct(x, qs=(50, 90, 100)):
    x = np.asarray(x, float)
    return " / ".join(f"{np.percentile(x, q):.3g}" for q in qs)


def group_scan(args):
    """All 24 branches of one (split, source, t): light per-file records + group metrics (frames never leave)."""
    body, split, src, t, paths = args
    rs = [scan((body, split, p)) for p in paths]
    own = [r for r in rs if r["cf_own"]]
    g = dict(body=body, split=split, src=src, t=t, n_own=len(own))
    if len(own) != 1:
        for r in rs:
            for k in ("frames", "state", "act"):
                r.pop(k, None)
        return rs, g
    own = own[0]
    sl, lab = source_slice(body, own)
    pre, post = slice(0, PRE + 1), slice(PRE + 1, NB)
    if body == "b1":
        g["pre_bad"] = sum(int(not (np.array_equal(r["frames"][pre], own["frames"][pre]) and
                                    all(np.array_equal(r["state"][k][pre], own["state"][k][pre]) for k in r["state"])))
                           for r in rs)
        fo, fs = own["frames"][post].astype(float), sl["frames"][post].astype(float)
        g.update(joint=np.abs(own["joint"] - sl["joint_pos"]).max(), pos=np.abs(own["head"] - sl["base_pos"]).max(),
                 quat=(np.abs(own["state"]["base_quat"].astype(np.float64) - sl["base_quat"]).max()
                       if body == "b1" else 0.0),
                 st_eq=(all(np.array_equal(own["state"][k], sl[k]) for k in ("joint_pos", "base_quat", "action", "command"))
                        if body == "b1" else True),
                 froude=np.abs(own["label"] - lab)[PRE:].max(), mae=np.abs(fo - fs).mean(),
                 corr=min(np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(fo, fs)), mae_t=own["mae_t"],
                 cam=np.abs(own["cam"][:, :3] - sl["cam_pose"][:, :3]).max(),
                 pre_src_mae=np.abs(own["frames"][pre].astype(float) - sl["frames"][pre].astype(float)).mean())
    else:
        g["pre_ok"] = all(bool(np.array_equal(r["act"][pre], sl["actions"][pre])) and r["prefix_ok"] for r in rs)
        g["rows_src"] = [dict(com=1000 * np.linalg.norm(r["com"][pre, :2] - sl["com_pos"][pre, :2], axis=1).max(),
                              head=1000 * np.linalg.norm(r["head"][pre] - sl["head"][pre], axis=1).max(),
                              joint=np.abs(r["joint"][pre] - sl["state_joint_pos"][pre]).max(),
                              pix=np.abs(r["frames"][pre].astype(float) - sl["frames"][pre].astype(float)).mean())
                         for r in rs]
        g["rows_btw"] = [dict(com=1000 * np.linalg.norm(r["com"][pre, :2] - own["com"][pre, :2], axis=1).max(),
                              head=1000 * np.linalg.norm(r["head"][pre] - own["head"][pre], axis=1).max(),
                              joint=np.abs(r["joint"][pre] - own["joint"][pre]).max(),
                              pix=np.abs(r["frames"][pre].astype(float) - own["frames"][pre].astype(float)).mean())
                         for r in rs if r is not own]
        ys = E.heading(sl["body_quat"].astype(np.float64), "hexapod")
        ds = body_disp(sl["com_pos"].astype(np.float64), ys[PRE])
        do = body_disp(own["com"], own["yaw"][PRE])
        g["noise"] = 1000 * np.linalg.norm(do - ds, axis=1)
        g["lab_noise"] = np.abs(own["label"] - lab).max(1)
        g["own"] = dict(com=1000 * np.linalg.norm(do[post] - ds[post], axis=1).max(),
                        joint=np.abs(own["joint"][post] - sl["state_joint_pos"][post]).max(),
                        pix=np.abs(own["frames"][post].astype(float) - sl["frames"][post].astype(float)).mean(),
                        froude=np.abs(own["label"][PRE:] - lab[PRE:]).max(0))
        g["eff"] = [(FAM[int(r["cf_command_index"])], 1000 * np.linalg.norm(body_disp(r["com"], r["yaw"][PRE]) - do, axis=1),
                     np.abs(r["label"] - own["label"]).max(1)) for r in rs if r is not own]
    for r in rs:
        for k in ("frames", "state", "act"):
            r.pop(k, None)
    return rs, g


def main():
    global HEXP, SPLITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bodies", nargs="+", default=["hex", "b1"])
    ap.add_argument("--hex_prefix", default="c10", help="hexapod dir prefix: c10 (default) or c08")
    ap.add_argument("--c08", action="store_true", help="the c08f09t09 test set: --hex_prefix c08 --splits heldout "
                    "--bodies hex")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    a = ap.parse_args()
    if a.c08:
        a.hex_prefix, a.splits, a.bodies = "c08", ["heldout"], ["hex"]
    HEXP = a.hex_prefix
    SPLITS = tuple(a.splits)
    print(f"hexapod dirs: {HEXP}_clips_* / {HEXP}_branches_*")
    gj = defaultdict(list)
    for b in a.bodies:
        for s in SPLITS:
            for p in sorted(glob.glob(os.path.join(cf_dir(b, s), "*.npz"))):
                if ".tmp" in p:
                    continue
                with np.load(p, allow_pickle=True) as d:
                    gj[(b, s, str(d["cf_source"]), int(d["cf_t"]))].append(p)
    fails = []
    R = defaultdict(list)
    G = defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex:
        for rs, g in ex.map(group_scan, [(*k, v) for k, v in sorted(gj.items())], chunksize=1):
            R[(g["body"], g["split"])].extend(rs)
            G[g["body"]].append(g)
    bad_own = [(g["split"], g["src"], g["t"], g["n_own"]) for b in G for g in G[b] if g["n_own"] != 1]
    if bad_own:
        print("groups without exactly one own-command branch:", bad_own[:10])
        fails.append("own branch")

    # ---------------- 1 coverage
    print("== 1 coverage: files and (source behaviour -> target behaviour) pairs")
    want = {"train": 3456, "val": 1728, "heldout": 1728}
    cnt = {}
    for b in a.bodies:
        for s in SPLITS:
            rs = R[(b, s)]
            pc = defaultdict(int)
            for r in rs:
                pc[(int(r["cf_source_cond_index"]), int(r["cf_command_index"]))] += 1
            vals = [pc.get((i, j), 0) for i in range(24) for j in range(24)]
            cnt[(b, s)] = vals
            ok = len(rs) == want[s] and min(vals) == max(vals) and min(vals) > 0
            print(f"  {b:<4} {s:<8} files {len(rs):5d} (want {want[s]}), pairs present {sum(v > 0 for v in vals)}/576, "
                  f"per pair {min(vals)}..{max(vals)} {'OK' if ok else 'FAIL'}")
            if not ok:
                fails.append(f"coverage {b} {s}")
    if len(a.bodies) == 2:
        same = all(cnt[("hex", s)] == cnt[("b1", s)] for s in SPLITS)
        print(f"  pair counts identical across bodies in every split: {same}")
        if not same:
            fails.append("coverage bodies")

    # ---------------- 2 rooms
    print("== 2 rooms")
    seeds = {}
    for b in a.bodies:
        bad = 0
        for s in SPLITS:
            src_seed = {}
            for p in glob.glob(os.path.join(CW, f"{main_sub(b)}_{s}", "*.npz")):
                with np.load(p, allow_pickle=True) as d:
                    src_seed[os.path.basename(p)] = int(d["room_seed"])
            for r in R[(b, s)]:
                bad += int(r["room_seed"]) != src_seed[r["cf_source"]]
            seeds[(b, s)] = {int(r["room_seed"]) for r in R[(b, s)]}
        x = [(s1, s2) for i, s1 in enumerate(SPLITS) for s2 in SPLITS[i + 1:] if seeds[(b, s1)] & seeds[(b, s2)]]
        print(f"  {b}: branches not in their source's room: {bad}; seeds per split "
              f"{[len(seeds[(b, s)]) for s in SPLITS]}; splits sharing a seed: {x or 'none'}")
        if bad or x:
            fails.append(f"rooms {b}")
    if len(a.bodies) == 2:
        same = all(seeds[("hex", s)] == seeds[("b1", s)] for s in SPLITS)
        print(f"  identical seed sets for both bodies per split: {same}")
        if not same:
            fails.append("rooms bodies")

    groups = {b: defaultdict(list) for b in a.bodies}
    for (b, s_), rs in R.items():
        for r in rs:
            groups[b][(s_, r["cf_source"], int(r["cf_t"]))].append(r)

    # ---------------- 3 B1
    if "b1" in a.bodies:
        print("== 3 B1 integrity")
        gs = [g for g in G["b1"] if g["n_own"] == 1]
        o = {k: np.array([g[k] for g in gs]) for k in ("pre_bad", "joint", "pos", "quat", "st_eq", "froude", "mae", "corr",
                                                      "mae_t", "cam", "pre_src_mae")}
        print(f"  groups {len(gs)}; branches whose prefix (11 frames: frames + state) differs from the group's own-command "
              f"branch: {int(o['pre_bad'].sum())}; prefix frames vs source: pixel MAE max {o['pre_src_mae'].max():.3f}")
        print(f"  own-command branch vs source window ({len(gs)} groups), max: joint {o['joint'].max():.2e} rad, "
              f"base pos {o['pos'].max():.2e} m, base quat {o['quat'].max():.2e}, groups with joint_pos / base_quat / "
              f"action / command not bit-identical {int((~o['st_eq']).sum())}, Froude (from the branch frame) {o['froude'].max():.2e}, cam pos "
              f"{o['cam'].max():.2e} m, pixel MAE {o['mae'].max():.3f} (min corr {o['corr'].min():.5f}); branch frame "
              f"re-rendered from the reproduced state vs stored: pixel MAE max {o['mae_t'].max():.3f}")
        # Froude tolerance = float32 storage bound of the label, not an empirical margin. base_pos / base_quat
        # are stored float32 (com_pos is float64 but computed from them). Each stored xy is rounded twice:
        # walk coords (|x| <= 10.95 m, ulp/2 = 4.77e-7) then window coords after the offset (|x| <= 1.76 m,
        # ulp/2 = 5.96e-8) -> <= 5.4e-7 m per file, <= 1.07e-6 m between branch and source. The label is a
        # one-sided (segment edge) gradient, <= 2 * 1.07e-6 / dt(0.05) per axis, sqrt(2) in-plane = 6.07e-5 m/s,
        # smoothing averages (no gain), / sqrt(g * h_min), h_min = 0.5235 m -> 2.68e-5. Rounded up: 3e-5.
        # **Strict own-command gate (2026-10-02).** The B1 restore is bit-exact once the branch runs the command as the
        # walk ran it (6-decimal CLI value, build_branches.b1_cmd_as_run), so the own-command branch must equal
        # the walk to float32 storage: joint_pos / base_quat / action / command are float32(the same float64) ->
        # bit-identical; base_pos only differs by the offset re-rounding, <= 1.07e-6 m (bound above), Froude <= 3e-5.
        # Before the fix: base 2.6e-5 m, quat 3.4e-6, joint 3.4e-7 (all failing this gate).
        if (o["pre_bad"].sum() or (~o["st_eq"]).sum() or o["joint"].max() > 0 or o["quat"].max() > 0
                or o["pos"].max() > 1.07e-6 or o["froude"].max() > 3e-5 or o["corr"].min() < 0.999):
            fails.append("B1 integrity")

    # ---------------- 4 hexapod
    if "hex" in a.bodies:
        print("== 4 hexapod integrity (replay noise floor)")
        gs = [g for g in G["hex"] if g["n_own"] == 1]
        pre_ok = all(g["pre_ok"] for g in gs)
        n_cmp = sum(len(g["rows_src"]) for g in gs)
        rows_src = [x for g in gs for x in g["rows_src"]]
        rows_btw = [x for g in gs for x in g["rows_btw"]]
        own_rows = [g["own"] for g in gs]
        noise = {k: [g["noise"][PRE + k] for g in gs] for k in range(1, NB - PRE)}
        lab_noise = {k: [g["lab_noise"][PRE + k] for g in gs] for k in range(1, NB - PRE)}
        eff = defaultdict(lambda: defaultdict(list))
        lab_eff = defaultdict(lambda: defaultdict(list))
        for g in gs:
            for fam, e, le in g["eff"]:
                for k in range(1, NB - PRE):
                    eff[fam][k].append(e[PRE + k]); lab_eff[fam][k].append(le[PRE + k])
        print(f"  prefix commands bit-identical to the source plan: {pre_ok} ({n_cmp} branches)")
        if not pre_ok:
            fails.append("hex prefix commands")
        for nm, rows in (("vs source window", rows_src), ("vs own-command branch of the group", rows_btw)):
            q = {k: [x[k] for x in rows] for k in rows[0]}
            print(f"  prefix (11 frames) {nm}, median / p90 / max over {len(rows)} branches: CoM {pct(q['com'])} mm, "
                  f"head {pct(q['head'])} mm, joint {pct(q['joint'])} rad, pixel MAE {pct(q['pix'])}")
        q = {k: [x[k] for x in own_rows] for k in own_rows[0]}
        fr = np.array(q["froude"])
        print(f"  own-command branch vs source, 20 frames after the branch ({len(own_rows)} groups), median / p90 / max: "
              f"CoM displacement from the branch frame (body frame) {pct(q['com'])} mm, joint {pct(q['joint'])} rad, "
              f"pixel MAE {pct(q['pix'])}, Froude |d| fwd {pct(fr[:, 0])} lat {pct(fr[:, 1])} yaw {pct(fr[:, 2])}")
        if True:   # scene-reuse hexapod data (F305): exact by construction
            exact = (max(max(x.values()) for x in rows_src) == 0 and max(max(x.values()) for x in rows_btw) == 0
                     and max(q["com"]) == 0 and max(q["joint"]) == 0 and max(q["pix"]) == 0 and fr.max() == 0)
            print(f"  [exact] prefix bit-identical to the source window and across every group, own-command branch == "
                  f"source on all 31 frames (state, labels, pixels): {exact}")
            if not exact:
                fails.append("hex exactness")
        print("  switching effect vs noise floor, CoM displacement from the branch frame (mm, body frame): noise = own "
              "branch vs source p90; effect = median over switched branches of |branch - own branch|; first frame k "
              "after the branch where effect > noise p90 (and Froude label |d| > its noise p90)")
        npk = {k: np.percentile(noise[k], 90) for k in noise}
        lpk = {k: np.percentile(lab_noise[k], 90) for k in lab_noise}
        print(f"    noise p90 at k=1,3,5,10,20: " + ", ".join(f"{npk[k]:.1f}" for k in (1, 3, 5, 10, 20)) +
              "  | label noise p90: " + ", ".join(f"{lpk[k]:.4f}" for k in (1, 3, 5, 10, 20)))
        for fam in ("fwd", "bwd", "turn_left", "turn_right", "side_L", "side_R"):
            em = {k: np.median(eff[fam][k]) for k in eff[fam]}
            lm = {k: np.median(lab_eff[fam][k]) for k in lab_eff[fam]}
            k1 = next((k for k in range(1, 21) if em[k] > npk[k]), None)
            k2 = next((k for k in range(1, 21) if lm[k] > lpk[k]), None)
            print(f"    target {fam:<10} effect at k=1,3,5,10,20: " + ", ".join(f"{em[k]:.1f}" for k in (1, 3, 5, 10, 20))
                  + f"  -> exceeds noise from k={k1} (CoM), k={k2} (Froude label)")

    # ---------------- 5 phases
    print("== 5 branch-point gait phases (10 bins over the cycle; thirds)")
    for b in a.bodies:
        ph = np.array([rs[0]["cf_gait_phase"] for rs in groups[b].values()])
        print(f"  {b}: {len(ph)} points, hist {np.histogram(ph, 10, (0, 1))[0].tolist()}, thirds "
              f"{np.histogram(ph, 3, (0, 1))[0].tolist()}")

    # ---------------- 6 falls
    print("== 6 falls")
    for b in a.bodies:
        rs = [r for v in R.values() for r in v if r["body"] == b]
        if b == "hex":
            zf = min(r["zmin"] for r in rs)
            print(f"  hexapod: min CoM z {zf:.3f} m (source median CoM z {np.median([r['froude_height'] for r in rs]):.3f}), "
                  f"max tilt from the first frame {max(r['tilt'] for r in rs):.1f} deg")
            if zf < 0.5 * min(r["froude_height"] for r in rs) or max(r["tilt"] for r in rs) > 45:
                fails.append("hex falls")
        else:
            nf = sum(r["fell"] for r in rs)
            print(f"  B1: fell flags {nf}; min base z {min(r['zmin'] for r in rs):.3f} m, max tilt {max(r['tilt'] for r in rs):.1f} deg")
            if nf:
                fails.append("b1 falls")

    # ---------------- 7 loader / labels
    print("== 7 loader + labels")
    for b in a.bodies:
        rs = [r for v in R.values() for r in v if r["body"] == b]
        print(f"  {b}: loader OK on {len(rs)} files, finite labels {all(r['finite'] for r in rs)}, segment field correct "
              f"{all(r['seg_ok'] for r in rs)}, first_pair=10 {all(int(r['first_pair']) == PRE for r in rs)}, "
              f"31 frames {all(r['n'] == NB for r in rs)}")
        if not all(r["finite"] and r["seg_ok"] and int(r["first_pair"]) == PRE and r["n"] == NB for r in rs):
            fails.append(f"loader {b}")
        # label independence: scramble the reference positions / orientation of frames 0..PRE-2 (frame PRE-1
        # enters the velocity difference at the switch frame, F296) -> labels from the branch frame unchanged
        worst = 0.0
        for r in rs[:: max(1, len(rs) // 40)]:
            with np.load(r["path"], allow_pickle=True) as d:
                dd = {k: d[k] for k in d.files}
            rng = np.random.default_rng(0)
            for k in ("com_pos", "head", "base_pos"):
                if k in dd:
                    dd[k] = dd[k].copy(); dd[k][:PRE - 1] += rng.normal(0, 0.3, dd[k][:PRE - 1].shape)
            qk = "body_quat" if b == "hex" else "base_quat"
            dd[qk] = dd[qk].copy(); dd[qk][:PRE - 1] = rng.normal(size=dd[qk][:PRE - 1].shape)
            dd[qk][:PRE - 1] /= np.linalg.norm(dd[qk][:PRE - 1], axis=1, keepdims=True)
            tmp = tempfile.mktemp(suffix=".npz"); np.savez(tmp, **dd)
            lab = E.load(tmp, E.HEXAPOD if b == "hex" else E.B1)["body_motion"]
            os.remove(tmp)
            worst = max(worst, float(np.abs(lab[PRE:] - r["label"][PRE:]).max()))
        print(f"  {b}: labels from the branch frame with frames 0..{PRE - 2} scrambled: max |diff| {worst:.2e}")
        if worst > 1e-6:
            fails.append(f"label independence {b}")
    print("\nGATES", "PASS" if not fails else f"FAIL: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
