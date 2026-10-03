"""Hexapod v4 main clips + counterfactual branches re-collected with deterministic scene reuse (FINDINGS F305)
-> data/counterfactual_walks/c10_walks, c10_clips_{train,val,heldout}, c10_branches_{train,val,heldout}.

Why: the v4 hexapod branches were replayed in a freshly loaded scene each, and Bullet does not repeat a run
across loads (prefix CoM 63 / 239 / 2540 mm median / p90 / max from the source, F304). With the scene loaded
once and only stop/startSimulation between runs, runs become bit-identical after ~8-10 warm-up runs
(`sim/collect/scene_reuse.py`). The "class" of runs differs between instances and changes with the number of
runs since the load (measured: at runs 1, 8, 85, 261, 361, and a class can come back), so a whole condition --
its long walk AND all 288 branches -- runs on ONE instance session, and every branch is assigned to a class.

Per condition (24), one worker process per CoppeliaSim instance:
  1. commands of the walk (stored plan of data/counterfactual_walks/_superseded/c10_replay_noise/hex_main_walks/<c>.npz, same length N) and of all
     4 windows x 3 branch points x 24 commands (branch_points.json, same causal 4-frame cross-fade as
     build_branches.hex_plan), computed before the scene is loaded;
  2. load once; 90 short warm-up runs (past the switch at run 85), then full walk runs until 3 consecutive
     ones agree bit for bit (the first class reference);
  3. every branch run from frame 0 to T + 20 (T = window start + t); it belongs to the class whose walk run it
     equals bit for bit on frames T-10 .. T (a prefix matching no known walk -> one walk run, which identifies the
     current class or adds a new one); results are kept per class until one class has all 288 branches, and
     that class's walk run is saved as the walk; the own-command branch must equal that walk on all 31 frames;
  4. render (render_hex_replay links mode): 4 main windows (rooms / seeds / re-centring as collect_c10_replay_superseded)
     and the 288 branches (source room, source window offset).

    .venv/bin/python3 scripts/dataset/collect_c10_walks_and_branches.py drive --ports 24600 24610 ...   (launch + supervise)
    .venv/bin/python3 scripts/dataset/collect_c10_walks_and_branches.py worker --port P --conds c1 c2 ...
    .venv/bin/python3 scripts/dataset/collect_c10_walks_and_branches.py targets   (B1 clips vs the new hexapod targets)
    .venv/bin/python3 scripts/dataset/collect_c10_walks_and_branches.py video     (results/check/hex_det_branches_samples.mp4)
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render", "sim/scene"):
    sys.path.insert(0, os.path.join(ROOT, p))
from collect_c10_replay_superseded import ORDER, FAMILY, ROLES, EP, seed_of, split_of, schedule  # noqa: E402
from collect_switch_hex import COND, KEYS, BASE, CENTRE  # noqa: E402
from build_branches import hex_plan, code_of, PREFIX, BRANCH, NB, FADE, POINTS, CENTRE_NPZ, \
    PER_FRAME_HEX  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
OLD = "_superseded/c10_replay_noise"   # stage-1 replay walks / clips (plans + branch-point keys)
OLD_WALKS = os.path.join(CW, OLD, "hex_main_walks")
WALKS = os.path.join(CW, "c10_walks")
PHYS = os.path.join(WALKS, "_physics")
SCENE = "medauroidea_c10f10t10.ttt"
SPLITS = ("train", "val", "heldout")
PY = os.path.join(ROOT, ".venv/bin/python3")
# Warm-up (measured 2026-10-02, 5 instances): after a load the run "class" switches at run 1, run 8 and run 85
# (counted in runs, the same for 30 / 60 / 150-frame runs), then held for >= 165 runs. So: 90 short runs, then
# full walks until 3 consecutive identical ones. A switch later on is handled in the SAME loaded scene (re-establish
# the walk, redo the condition's branches) -- reloading would reset the count and meet the same switch again.
SHORT_WARM = 90
WARM_MIN = 3
COPPELIA = os.path.expanduser("~/CoppeliaSim")
DRIVE_KW = dict(travel=0.0, warmup=20, cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), ego=True, ego_euler=None,
                ego_offset=None, ego_box=40.0, ego_seed=0, cam_fov=90.0, cmd_noise=0.0, capture_frames=False)
STATE = ("state_abdomen_pos", "state_abdomen_quat", "state_joint_pos", "state_link_pose", "state_sim_time")
CMP = ("actions", "forces", "head", "body_quat") + STATE + ("com_pos",)
RENDER_MAIN = "render_hex_replay links mode, re-centred, fov 90, room 8 m"
RENDER_CF = "render_hex_replay links mode, source room + source window offset, fov 90, room 8 m"


def main_path(split, ep):
    return os.path.join(CW, f"c10_clips_{split}", f"hexapod_ep{ep}.npz")


def cf_path(split, ep, t, ci):
    return os.path.join(CW, f"c10_branches_{split}", f"hexapod_ep{code_of(ep, t, ci)}.npz")


def save_atomic(dst, **kw):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, dst)


def windows(i, c, W):
    """[(w, start, role, split, seed, ep, old main path, points t, phases)] -- same rule as collect_c10_replay_superseded."""
    P = json.load(open(POINTS))
    out = []
    for w, st in enumerate(W["window_starts"]):
        role = ROLES[(w + i) % 4]
        split, seed, ep = split_of(role), seed_of(i, role), 40000 + 10 * i + w
        old = f"data/counterfactual_walks/{OLD}/hex_main_{split}/hexapod_ep{ep}.npz"
        v = P[old]
        assert v["ep"] == ep and v["cond_index"] == i
        out.append((w, int(st), role, split, seed, ep, old, list(v["t"]), list(v["phase"])))
    return out


def cpg_phase(pace, N):
    cyc = BASE * N / EP * (np.cumsum(pace) - pace) / N
    return np.mod(cyc, 1.0), cyc


# ------------------------------------------------------------------------------------------------ physics
def physics(sim, port, i, c, log):
    """Walk + 288 branches of condition c on this instance (one loaded scene). Runs fall into "classes" (all
    runs of a class are bit-identical for identical commands; the class changes with the number of runs since
    the load: runs 1, 8, 85, 261, 361 measured). Every full walk run defines / identifies a class; a branch run
    belongs to the class whose walk it equals bit for bit on frames T-10 .. T (its prefix). Results are kept per
    class until one class has all 288 branches; that class's walk is saved. A prefix matching no known class
    triggers a walk run (a new class appears) and the branch is classified again."""
    from scene_reuse import SceneReuse, state_hash
    W = dict(np.load(os.path.join(OLD_WALKS, f"{c}.npz"), allow_pickle=True))
    N = len(W["actions"])
    g, st_, ph_, N2 = schedule(c)
    assert N2 == N and list(st_) == [int(x) for x in W["window_starts"]]
    plan = {k: np.asarray(W[f"plan_{k}"], float) for k in KEYS}
    for k in KEYS:
        assert np.all(plan[k] == float(COND[c][1][k])), (c, k)
    wins = windows(i, c, W)
    jobs = [(w, t, ph, ci) for (w, st, *_r, ts, phs) in wins for t, ph in zip(ts, phs) for ci in range(24)]
    starts = {w: st for (w, st, *_r) in wins}
    centre = np.load(CENTRE_NPZ, allow_pickle=True)["actions"].astype(np.float64).mean(0)
    cpg_kw = dict(cycles=BASE * N / EP, mirror_joints=(0, 1, 2), spin_amp=0.25, symmetric=False, legtune=None)
    plans = [plan] + [hex_plan(W, starts[w] + t, ci)[0] for (w, t, ph, ci) in jobs]
    R = SceneReuse(sim, SCENE, DRIVE_KW)
    t0 = time.time()
    cmds = R.commands(plans, N, centre, cpg_kw)
    assert np.array_equal(cmds[0], W["actions"]), "walk commands differ from the stored walk"
    for (w, t, ph, ci), cm in zip(jobs, cmds[1:]):
        T = starts[w] + t
        assert np.array_equal(cm[:T + 1], cmds[0][:T + 1]), "branch prefix commands differ"
        if ci == i:
            assert np.array_equal(cm, cmds[0]), "own-command cross-fade not bit-identical to the walk"
    log(f"{c}: {len(cmds)} command sets in {time.time() - t0:.0f} s; warm-up")
    ref0, nwarm, hashes = R.warm_up(cmds[0], min_runs=WARM_MIN, max_runs=40, n_same=3, short_runs=SHORT_WARM, log=log)
    refs = {state_hash(ref0): ref0}          # class -> walk run
    first_seen = {state_hash(ref0): R.n_runs}
    walk_runs = [(R.n_runs, state_hash(ref0))]
    res = {}                                  # class -> {job index: (run, tries)}
    n_unmatched, n_runs_branch = 0, 0

    def classify(out, a, T):
        for h, ref in refs.items():
            if R.same_as(out, ref, range(a, T + 1), a_from=a, keys=CMP):
                return h
        return None

    def new_walk():
        out = R.run(cmds[0])
        for h, ref in refs.items():
            if R.same_as(out, ref, range(N), keys=CMP):
                walk_runs.append((R.n_runs, h))
                return h, False
        h = state_hash(out)
        refs[h] = out; first_seen[h] = R.n_runs
        walk_runs.append((R.n_runs, h))
        log(f"{c}: new class {h} at run {R.n_runs} ({len(refs)} classes)")
        return h, True

    while True:
        best = max(res, key=lambda h: len(res[h])) if res else next(iter(refs))
        todo = [k for k in range(len(jobs)) if k not in res.get(best, {})]
        if not todo:
            break
        if R.n_runs > 2500:
            raise RuntimeError(f"{c}: no class completed in 2500 runs: {[(h, len(v)) for h, v in res.items()]}")
        for k in todo:
            w, t, ph, ci = jobs[k]
            T = starts[w] + t
            a = T - PREFIX
            for tr in range(3):
                out = R.run(cmds[1 + k][:T + BRANCH + 1], state_from=a)
                n_runs_branch += 1
                out["state_from"] = a
                h = classify(out, a, T)
                if h is None:
                    n_unmatched += 1
                    new_walk()
                    h = classify(out, a, T)
                if h is not None:
                    break
            if h is None:
                log(f"   {c} w{w} t{t} c{ci}: prefix matches no class walk after 3 tries")
                continue
            if ci == i and not R.same_as(out, refs[h], range(a, T + BRANCH + 1), a_from=a, keys=CMP):
                raise RuntimeError(f"{c}: own-command branch != walk of its class {h}")
            res.setdefault(h, {})[k] = (out, tr + 1)
            nb = len(res[h])
            if nb % 24 == 0 or nb == len(jobs):
                log(f"{c}: class {h}: {nb}/{len(jobs)} branches; {R.n_runs} runs, {time.time() - t0:.0f} s; classes "
                    f"{ {x: len(v) for x, v in res.items()} }")
            if nb == len(jobs):
                break
            # a new current class (more of the remaining work now lands elsewhere): re-plan
            if h != best and len(res[h]) > len(res.get(best, {})):
                break
    cls = max(res, key=lambda h: len(res[h]))
    ref = refs[cls]
    got = res[cls]
    res_arr = {key: [] for key in CMP}
    meta = dict(w=[], t=[], T=[], ci=[], phase=[], tries=[], own=[], own_equal=[])
    for k, (w, t, ph, ci) in enumerate(jobs):
        out, tries = got[k]
        T = starts[w] + t
        a = T - PREFIX
        for key in CMP:
            v = np.asarray(out[key])
            res_arr[key].append(v[a:a + NB] if not (key.startswith("state_") or key == "com_pos") else v[:NB])
        own_eq = bool(ci == i and R.same_as(out, ref, range(a, T + BRANCH + 1), a_from=a, keys=CMP))
        for kk, vv in zip(meta, (w, t, T, ci, ph, tries, ci == i, own_eq)):
            meta[kk].append(vv)
    det = dict(det_class=cls, det_instance=f"port{port}", det_instance_pid=_instance_pid(port),
               det_class_first_run=first_seen[cls], det_warmup_runs=nwarm, det_classes_seen=len(refs),
               det_runs=R.n_runs, det_branch_runs=n_runs_branch, det_unmatched_prefix=n_unmatched)
    save_walk(W, ref, i, c, N, det)
    save_atomic(os.path.join(PHYS, f"{c}.npz"), **{k: np.asarray(v) for k, v in res_arr.items()},
                **{f"meta_{k}": np.asarray(v) for k, v in meta.items()}, **det, warm_hashes=np.array(hashes),
                walk_runs=np.array([f"{n}:{h}" for n, h in walk_runs]), class_counts=np.array(json.dumps({h: len(v) for h, v in res.items()})))
    log(f"{c}: physics done, class {cls} (first seen at run {first_seen[cls]}), {len(refs)} classes seen, "
        f"{R.n_runs} runs ({n_runs_branch} branch runs for {len(jobs)} branches), unmatched prefixes {n_unmatched}, "
        f"{time.time() - t0:.0f} s")


def _instance_pid(port):
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if f"rpcPort={port} " in ln + " " and "coppeliaSim -h" in ln:
            return int(ln.split()[0])
    return -1


def save_walk(W, ref, i, c, N, det):
    rec = {k: W[k] for k in ("foot_order", "morph", "expert_episode", "repeat", "scale", "behavior", "schedule",
                             "gait", "state_joint_names", "state_link_names", "rec_ego", "rec_ego_box", "rec_cam_fov",
                             "rec_spawn", "rec_warmup", "dt", "condition", "cond_index", "family", "family_level",
                             "walk_gap", "window_starts", "centre_from")}
    rec.update({f"plan_{k}": W[f"plan_{k}"] for k in KEYS})
    for k in CMP:
        rec[k] = np.asarray(ref[k])
    rec["head"] = rec["head"].astype(np.float32)
    rec["frames"] = np.zeros((0,), np.uint8)
    rec["step_idx"] = np.zeros((0,), np.int64)
    rec["state_link_mass"] = ref["state_link_mass"]
    ph, cyc = cpg_phase(W["plan_pace"], N)
    assert np.array_equal(ph, W["cpg_phase"]), "cpg_phase differs"
    rec["cpg_phase"], rec["cpg_cycles_total"] = ph, cyc
    rec.update(det, scene_reuse=True)
    assert np.array_equal(rec["actions"], W["actions"])
    save_atomic(os.path.join(WALKS, f"{c}.npz"), **rec)


# ------------------------------------------------------------------------------------------------ render
_tags = {}


def old_main(path):
    with np.load(os.path.join(ROOT, path), allow_pickle=True) as d:
        return {k: d[k] for k in d.files if k != "frames"}


def render_condition(sim, i, c, log):
    from render_hex_replay import render, ROOM, CAM_POSE_CONVENTION
    rec = dict(np.load(os.path.join(WALKS, f"{c}.npz"), allow_pickle=True))
    B = dict(np.load(os.path.join(PHYS, f"{c}.npz"), allow_pickle=True))
    det = {k: rec[k] for k in rec if k.startswith("det_")}
    wins = windows(i, c, rec)
    per_frame = ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat",
                 "state_joint_pos", "state_link_pose", "state_sim_time", "cpg_phase", "cpg_cycles_total", "com_pos") + \
        tuple(f"plan_{k}" for k in KEYS)
    t0 = time.time()
    mains = {}
    for (w, st, role, split, seed, ep, oldp, ts, phs) in wins:
        dst = main_path(split, ep)
        O = old_main(oldp)
        if not os.path.exists(dst):
            frames, off, Rm, cam = render(sim, rec, st, EP, seed, recentre=True, room=ROOM, return_cam=True)
            out = {k: np.array(rec[k][st:st + EP]) for k in per_frame}
            out["head"] = out["head"].astype(np.float64); out["head"][:, :2] += off
            out["head"] = out["head"].astype(np.float32)
            out["state_abdomen_pos"][:, :2] += off
            out["state_link_pose"][:, :, :2] += off
            out["com_pos"][:, :2] += off
            out.update(frames=frames, foot_order=rec["foot_order"], morph=rec["morph"], expert_episode=ep, repeat=w,
                       scale=rec["scale"], behavior="walk", schedule="", gait="cpg", step_idx=np.zeros((0,), np.int64),
                       state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"],
                       condition=c, behaviour=O["behaviour"], level=O["level"], family=FAMILY[i], family_level=i % 4,
                       cond_index=i, embodiment="hexapod", room_seed=seed, ego_seed=seed, room_size=float(Rm["size"]),
                       window_start=st, window_index=w, copy=role, split=split,
                       source_walk=os.path.relpath(os.path.join(WALKS, f"{c}.npz"), ROOT), offset_xy=off, dt=0.05,
                       centre_from=CENTRE, render=RENDER_MAIN, cam_pose=cam,
                       cam_pose_convention=np.array(CAM_POSE_CONVENTION), cam_pose_local=O["cam_pose_local"],
                       cam_pose_parent=O["cam_pose_parent"], **det)
            assert O["room_seed"] == seed and O["window_start"] == st and O["split"] == split
            save_atomic(dst, **out)
        with np.load(dst, allow_pickle=True) as d:
            mains[w] = {k: d[k] for k in ("offset_xy", "com_pos", "behaviour", "condition")}
            mains[w]["path"] = os.path.relpath(dst, ROOT)
    log(f"{c}: 4 main clips rendered ({time.time() - t0:.0f} s)")
    beh = {}
    for s in SPLITS:
        for p in glob.glob(os.path.join(CW, OLD, f"hex_main_{s}", "*.npz")):
            with np.load(p, allow_pickle=True) as f:
                beh[int(f["cond_index"])] = str(f["behaviour"])
    winfo = {w: (st, split, seed, ep) for (w, st, role, split, seed, ep, *_r) in wins}
    N = len(rec["actions"])
    n = 0
    for k in range(len(B["meta_w"])):
        w, t, T, ci = (int(B[f"meta_{x}"][k]) for x in ("w", "t", "T", "ci"))
        ph = float(B["meta_phase"][k])
        st, split, seed, ep = winfo[w]
        dst = cf_path(split, ep, t, ci)
        if os.path.exists(dst):
            continue
        a = T - PREFIX
        plan, wgt = hex_plan(rec, T, ci)
        br = {key: np.array(B[key][k]) for key in CMP}
        br.update(state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"])
        off_src = mains[w]["offset_xy"]
        frames, off, Rr, cam = render(sim, br, 0, NB, seed, room=8.0, return_cam=True, offset_xy=off_src)
        out = {key: br[key] for key in CMP}
        cph, ccyc = cpg_phase(plan["pace"], N)
        out["cpg_phase"], out["cpg_cycles_total"] = cph[a:a + NB], ccyc[a:a + NB]
        out["head"] = out["head"].astype(np.float64); out["head"][:, :2] += off; out["head"] = out["head"].astype(np.float32)
        out["state_abdomen_pos"][:, :2] += off
        out["state_link_pose"][:, :, :2] += off
        out["com_pos"][:, :2] += off
        for kk in KEYS:
            out[f"plan_{kk}"] = plan[kk][a:a + NB]
            if kk != "pace":
                out[f"xf_{kk}"] = plan[f"xf_{kk}"][a:a + NB]
        out["xf_w"] = wgt[a:a + NB]
        froude_h = float(np.median(mains[w]["com_pos"][:, 2]))
        name = ORDER[ci]
        src = mains[w]["path"]
        out.update(frames=frames, cam_pose=cam, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
                   foot_order=rec["foot_order"], morph=rec["morph"], scale=rec["scale"], gait="cpg",
                   state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"],
                   condition=np.array(name), cond_index=np.array(ci), family=np.array(FAMILY[ci]),
                   family_level=np.array(ci % 4), behaviour=np.array(beh[ci]), level=np.array(ci % 4),
                   embodiment=np.array("hexapod"), expert_episode=np.array(code_of(ep, t, ci)), dt=np.float64(0.05),
                   room_seed=np.int64(seed), ego_seed=np.int64(seed), room_size=np.float64(Rr["size"]),
                   split=np.array(split), offset_xy=off, centre_from=np.array(CENTRE),
                   segment=(np.arange(NB) >= PREFIX).astype(np.int8), first_pair=np.int64(PREFIX),
                   froude_height=np.float64(froude_h),
                   cf_source=np.array(os.path.basename(src)), cf_source_path=np.array(src), cf_source_episode=np.array(ep),
                   cf_source_condition=np.array(c), cf_source_cond_index=np.array(i), cf_source_family=np.array(FAMILY[i]),
                   cf_source_behaviour=mains[w]["behaviour"], cf_source_walk=np.array(os.path.relpath(os.path.join(WALKS, f"{c}.npz"), ROOT)),
                   cf_t=np.array(t), cf_walk_frame=np.array(T), cf_window_start=np.array(st),
                   cf_branch_index=np.array(PREFIX), cf_gait_phase=np.float64(ph), cf_gait_phase_kind=np.array("cpg clock"),
                   cf_command_index=np.array(ci), cf_command_name=np.array(name), cf_target_condition=np.array(name),
                   cf_own=np.array(ci == i), cf_fade_frames=np.array(FADE),
                   cf_prefix_cmd_identical=np.array(bool(np.array_equal(br["actions"][:PREFIX + 1], rec["actions"][a:T + 1]))),
                   cf_prefix_state_identical=np.array(True), cf_prefix_tries=np.int64(B["meta_tries"][k]),
                   cf_own_equals_walk=np.array(bool(B["meta_own_equal"][k])) if ci == i else np.array(False),
                   cf_settings=np.array(json.dumps(dict(scene_reuse=True, drive=dict(DRIVE_KW, spawn=[0.0, 0.0]),
                                                        centre_from=os.path.relpath(CENTRE_NPZ, ROOT), walk_frames=N,
                                                        stop_after=T + BRANCH + 1,
                                                        fade="cmd=(1-w)cmd_old+w cmd_new, w=clip((frame-T)/4,0,1), pace blended, one clock"))),
                   render=np.array(RENDER_CF), **det)
        save_atomic(dst, **out)
        n += 1
        if n % 48 == 0:
            log(f"{c}: {n} branches rendered ({time.time() - t0:.0f} s)")
    log(f"{c}: render done ({n} new branch files, {time.time() - t0:.0f} s)")


def condition_done(i, c):
    if not os.path.exists(os.path.join(PHYS, f"{c}.npz")):
        return False
    W = np.load(os.path.join(OLD_WALKS, f"{c}.npz"), allow_pickle=True)
    wins = windows(i, c, W)
    return all(os.path.exists(main_path(s, ep)) and all(os.path.exists(cf_path(s, ep, t, ci)) for t in ts for ci in range(24))
               for (w, st, role, s, seed, ep, oldp, ts, phs) in wins)


def do_worker(a):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    logf = open(a.log, "a")

    def log(msg):
        logf.write(f"{time.strftime('%H:%M:%S')} {msg}\n"); logf.flush()
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    for c in a.conds:
        i = ORDER.index(c)
        if condition_done(i, c):
            continue
        if not os.path.exists(os.path.join(PHYS, f"{c}.npz")):
            physics(sim, a.port, i, c, log)
        render_condition(sim, i, c, log)
    log("WORKER DONE")


# ------------------------------------------------------------------------------------------------ driver
def start_instance(port, log_dir):
    subprocess.Popen(f"tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort={port} "
                     f"-GzmqRemoteApi.cntPort={port + 1}", shell=True, cwd=COPPELIA, start_new_session=True,
                     stdout=open(os.path.join(log_dir, f"csim_{port}.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        r = subprocess.run([PY, "-c", f"from coppeliasim_zmqremoteapi_client import RemoteAPIClient as C; "
                                      f"print(C('localhost', port={port}).require('sim').getSimulationState())"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return
    raise RuntimeError(f"CoppeliaSim on {port} did not start")


def kill_instance(port):
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if f"rpcPort={port} " in ln + " " and "grep" not in ln:
            try:
                os.kill(int(ln.split()[0]), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    time.sleep(2)


def do_drive(a):
    log_dir = os.path.join(WALKS, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    conds = a.conds or ORDER
    lens = {c: len(np.load(os.path.join(OLD_WALKS, f"{c}.npz"))["actions"]) for c in conds}
    shares = [[] for _ in a.ports]
    for k, c in enumerate(sorted(conds, key=lambda c: -lens[c])):
        r = k % (2 * len(a.ports))
        shares[r if r < len(a.ports) else 2 * len(a.ports) - 1 - r].append(c)
    procs = {}

    def launch(slot):
        port = a.ports[slot]
        lg = os.path.join(log_dir, f"worker_{port}.log")
        procs[slot] = (subprocess.Popen([PY, os.path.abspath(__file__), "worker", "--port", str(port), "--log", lg,
                                         "--conds"] + shares[slot], cwd=ROOT, start_new_session=True,
                                        stdout=open(lg + ".out", "a"), stderr=subprocess.STDOUT), lg)
    for s in range(len(a.ports)):
        print(f"port {a.ports[s]}: {shares[s]}", flush=True)
        launch(s)
    restarts = 0
    while procs:
        time.sleep(30)
        for s, (p, lg) in list(procs.items()):
            rc = p.poll()
            stale = os.path.exists(lg) and time.time() - os.path.getmtime(lg) > a.stall
            if rc == 0:
                del procs[s]
                continue
            if rc is not None or stale:
                port = a.ports[s]
                tail = open(lg + ".out").read()[-800:] if os.path.exists(lg + ".out") else ""
                print(f"worker {port} {'exited ' + str(rc) if rc is not None else 'stalled'}; restarting instance + worker"
                      f"\n{tail}", flush=True)
                if rc is None:
                    os.killpg(p.pid, signal.SIGKILL)
                restarts += 1
                if restarts > a.max_restarts:
                    raise RuntimeError("too many restarts")
                for f in glob.glob(os.path.join(CW, "c10_*", "*.tmp*.npz")) + glob.glob(os.path.join(PHYS, "*.tmp*.npz")):
                    os.remove(f)
                kill_instance(port)
                start_instance(port, log_dir)
                with open(lg, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} RESTART instance {port}\n")
                launch(s)
    print(f"all workers done; restarts {restarts}", flush=True)


# ------------------------------------------------------------------------------------------------ B1 targets
def do_targets(a):
    """Per-condition hexapod targets (mean of the 4 clip-mean CoM Froude labels) from c10_clips vs the targets
    the B1 was tuned to (_superseded/c10_replay_noise/hex_main, b1_walks/targets.npy), and the B1 main clips (b1_clips) checked against the
    new targets with the stage-2 tolerance max(10 %, 0.005) per channel (collect_b1_walks.ok)."""
    import wm.data.embodiment as E
    from collect_b1_walks import ok

    def means(pattern, spec):
        T = {}
        for s_ in SPLITS:
            for p in glob.glob(os.path.join(CW, pattern.format(s_), "*.npz")):
                if ".tmp" in p:
                    continue
                with np.load(p, allow_pickle=True) as d:
                    i = int(d["cond_index"])
                T.setdefault(i, []).append(E.load(p, spec)["body_motion"].astype(np.float64).mean(0))
        assert sorted(T) == list(range(24)) and all(len(v) == 4 for v in T.values()), {k: len(v) for k, v in T.items()}
        return np.array([np.mean(T[i], 0) for i in range(24)]), np.array([np.std(T[i], 0) for i in range(24)])
    new, new_sd = means("c10_clips_{}", E.HEXAPOD)
    old, _ = means(OLD + "/hex_main_{}", E.HEXAPOD)
    tuned = np.load(os.path.join(CW, "b1_walks", "targets.npy"))
    b1, _ = means("b1_clips_{}", E.B1)
    np.save(os.path.join(WALKS, "targets.npy"), new)
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)  # noqa: E731
    print(f"old hex targets == b1_walks/targets.npy: max |d| {np.abs(old - tuned).max():.2e}")
    print("cond | new hex target (fwd lat yaw) | old target | new - old | B1 clips mean | B1 - new | tol | within")
    nbad = 0
    for i, c in enumerate(ORDER):
        okc = ok(b1[i], new[i])
        nbad += int(not okc.all())
        tol = np.maximum(0.1 * np.abs(new[i]), 0.005)
        print(f"{i:2d} {c:<16} {f(new[i])} | {f(old[i])} | {f(new[i] - old[i])} | {f(b1[i])} | {f(b1[i] - new[i])} | "
              f"{f(tol)} | {'OK' if okc.all() else 'OUT ' + str(okc.tolist())}")
    print(f"max |new - old| per channel {np.round(np.abs(new - old).max(0), 4)}; B1 conditions out of tolerance: {nbad}/24")
    print("TARGETS", "PASS" if nbad == 0 else "FAIL")


# ------------------------------------------------------------------------------------------------ video
def do_video(a):
    """results/check/hex_det_branches_samples.mp4: per block one branch point x 4 commands side by side (ego view),
    with CoM Froude traces (fwd red, lat green, yaw blue; yellow = branch frame)."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import wm.data.embodiment as E
    W_, H_ = 256, 110
    cols = [(255, 80, 80), (80, 200, 80), (90, 140, 255)]
    ix = ORDER.index
    blocks = [("speed_c7.1", "train0", ["speed_c7.1", "turn_s0.56", "side_R_lvl2", "speed_c7.1_bwd"]),
              ("turn_s0.29_neg", "val", ["turn_s0.29_neg", "speed_c8.8", "side_L_lvl3", "turn_s0.29"]),
              ("side_L_lvl2", "heldout", ["side_L_lvl2", "speed_c5.8", "turn_s0.15_neg", "side_R_lvl3"])]

    def trace(bm, t, w):
        im = Image.new("RGB", (w, H_), (20, 20, 20)); dr = ImageDraw.Draw(im)
        lo, hi = min(bm.min(), -0.05), max(bm.max(), 0.05)
        y = lambda v: H_ - 4 - (v - lo) / (hi - lo) * (H_ - 22)  # noqa: E731
        x = lambda k: 4 + k * (w - 8) / (NB - 1)  # noqa: E731
        dr.line([(0, y(0)), (w, y(0))], fill=(80, 80, 80))
        dr.line([(x(PREFIX), 16), (x(PREFIX), H_)], fill=(200, 200, 0))
        for ch in range(3):
            dr.line([(x(k), y(bm[k, ch])) for k in range(NB)], fill=cols[ch], width=2)
        dr.line([(x(t), 16), (x(t), H_)], fill=(255, 255, 255))
        dr.text((3, 2), f"Froude fwd {bm[t, 0]:+.3f} lat {bm[t, 1]:+.3f} yaw {bm[t, 2]:+.3f}", fill=(230, 230, 230))
        return np.asarray(im)

    dst = os.path.join(ROOT, "results/check/hex_det_branches_samples.mp4")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    wr = imageio.get_writer(dst, fps=6)
    for c, role, cmds in blocks:
        i = ix(c)
        W = np.load(os.path.join(WALKS, f"{c}.npz"), allow_pickle=True)
        w_, st, role_, split, seed, ep, oldp, ts, phs = [x for x in windows(i, c, W) if x[2] == role][0]
        t_b = ts[0]
        data = []
        for cn in cmds:
            p = cf_path(split, ep, t_b, ix(cn))
            with np.load(p, allow_pickle=True) as f:
                data.append((f["frames"], cn, bool(f["cf_own"]), float(f["cf_gait_phase"]), E.load(p, E.HEXAPOD)["body_motion"]))
        same = all(np.array_equal(d[0][:PREFIX + 1], data[0][0][:PREFIX + 1]) for d in data)
        title = (f"hexapod (scene reuse), source {c} ({split}, room {seed}), branch at window frame {t_b}; "
                 f"prefix frames identical across the 4: {same}")
        for t in list(range(NB)) + [NB - 1] * 6:
            tiles = []
            for fr, name, own, ph, bm in data:
                im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, W_ - 1, 38], fill=(0, 0, 0))
                dr.text((3, 2), f"{c} ->", fill=(255, 255, 255))
                dr.text((3, 14), f"{name}{' (own = no switch)' if own else ''}", fill=(255, 255, 0) if own else (255, 255, 255))
                phase = "prefix (before switch)" if t < PREFIX else ("BRANCH FRAME" if t == PREFIX else "new command")
                dr.text((3, 26), f"t{t - PREFIX:+d} {phase} ph {ph:.2f}", fill=(200, 200, 200))
                tiles.append(np.concatenate([np.asarray(im), trace(bm, t, W_)], 0))
            img = np.concatenate(tiles, 1)
            head = Image.new("RGB", (img.shape[1], 18), (40, 40, 40)); ImageDraw.Draw(head).text((4, 3), title, fill=(255, 255, 255))
            wr.append_data(np.concatenate([np.asarray(head), img], 0))
    wr.close()
    print("->", os.path.relpath(dst, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("drive", "worker", "targets", "video"))
    ap.add_argument("--ports", type=int, nargs="+")
    ap.add_argument("--port", type=int)
    ap.add_argument("--conds", nargs="+", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--stall", type=float, default=600.0, help="s without a worker log line -> restart")
    ap.add_argument("--max_restarts", type=int, default=12)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(PHYS, exist_ok=True)
    if a.step == "drive":
        do_drive(a)
    elif a.step == "targets":
        do_targets(a)
    elif a.step == "video":
        do_video(a)
    else:
        try:
            do_worker(a)
        except Exception:
            with open(a.log, "a") as f:
                f.write(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
