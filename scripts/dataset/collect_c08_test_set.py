"""c08f09t09 (held-out hexapod morphology, TEST ONLY, never trained on) v4-style test set, built exactly like the c10
deterministic heldout data (collect_c10_walks_and_branches.py, FINDINGS F305)
-> data/counterfactual_walks/c08_walks (24 walks), c08_clips_heldout (24 clips), c08_branches_heldout (1728 branches).

Same as c10, per condition i (beh24_conditions.ORDER):
  - commands: the c10 walk's plan (c10_walks/<c>.npz plan_*, same length N, same CPG parameters:
    cycles 8.8 * N / 66, spin_amp 0.25, mirror (0, 1, 2)); the CPG centre pose is c08's OWN, fitted (bias +
    sinusoids, residual 1.2e-7 rad) from the c08 speed_c7.1 raw-pose clip data/egocentric/beh12_c08f09t09_ego_flat/
    hexapod_ep100.npz, as c10's centre is fitted from its own speed_c7.1 clip (collect_switch_hex.centre_pose);
    joint commands from collect_ik.cpg_commands on medauroidea_c08f09t09.ttt;
  - one window: the c10 heldout copy's window (w with ROLES[(w + i) % 4] == "heldout", same start frame), room seed
    200 + i, re-centred (render_hex_replay links mode, room 8 m, fov 90);
  - branches: the c10 heldout clip's 3 branch points (branch_points.json; the CPG clock is identical because the plan is,
    asserted) x 24 commands, causal 4-frame cross-fade (build_branches.hex_plan), 31 frames, segment /
    first_pair = 10 / froude_height = median CoM z of the c08 main clip;
  - determinism: scene loaded once per condition, 90 short warm-up runs, walk runs until 3 in a row identical,
    every branch assigned to the run class whose walk it matches bit for bit on frames T-10 .. T; the walk of the
    first class with all 72 branches is saved; own-command branch == walk on all 31 frames (hard).
CoM: com_pos from the c08 scene's own masses (collect_ik records state_link_mass live; checked against
wm.data.com.hex_com(..., morph="c08f09t09") = sim/env/medauroidea_c08f09t09_masses.json).
Episodes: main 60000 + 10 i + w (c10: 40000 + ...), branches code_of(ep, t, ci).

    .venv/bin/python3 scripts/dataset/collect_c08_test_set.py drive --ports 24800 24810 24820 24830 24840 24850
    .venv/bin/python3 scripts/dataset/collect_c08_test_set.py worker --port P --conds c1 c2 ... --log L
    .venv/bin/python3 scripts/dataset/collect_c08_test_set.py check    (Froude vs c10, falls, CoM, rooms, phases)
    .venv/bin/python3 scripts/dataset/collect_c08_test_set.py video    (results/check/c08_det_samples.mp4)
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
from beh24_conditions import ORDER, FAMILY, ROLES, EP  # noqa: E402
from collect_switch_hex import COND, KEYS, BASE  # noqa: E402
from build_branches import hex_plan, code_of, PREFIX, BRANCH, NB, FADE, POINTS  # noqa: E402
from collect_c10_walks_and_branches import (DRIVE_KW, CMP, SHORT_WARM, WARM_MIN, save_atomic, cpg_phase, _instance_pid,  # noqa: E402
                                start_instance, kill_instance, RENDER_MAIN, RENDER_CF)

MORPH = "c08f09t09"
SCENE = f"medauroidea_{MORPH}.ttt"
CW = os.path.join(ROOT, "data/counterfactual_walks")
C10_WALKS = os.path.join(CW, "c10_walks")
WALKS = os.path.join(CW, "c08_walks")
PHYS = os.path.join(WALKS, "_physics")
MAIN = os.path.join(CW, "c08_clips_heldout")
CF = os.path.join(CW, "c08_branches_heldout")
CENTRE_SRC = "data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz"      # c08 speed_c7.1, raw pose
CENTRE_NPZ = os.path.join(WALKS, "_work", "centre_pose.npz")
PY = os.path.join(ROOT, ".venv/bin/python3")
EP0 = 60000


def centre_pose():
    """bias + sinusoids fit of the c08 speed_c7.1 clip's commands (as collect_switch_hex.centre_pose for c10)."""
    with np.load(os.path.join(ROOT, CENTRE_SRC), allow_pickle=True) as d:
        a = d["actions"].astype(np.float64)
        assert str(d["condition"]) == "speed_c7.1" and str(d["morph"]) == MORPH
    t = np.arange(len(a))
    ph = 2 * np.pi * 7.1 * t / EP
    M = np.stack([np.ones(len(a)), np.sin(ph), np.cos(ph)], 1)
    coef = np.linalg.lstsq(M, a, rcond=None)[0]
    assert np.abs(M @ coef - a).max() < 1e-4, "c08 centre clip is not a 7.1-cycle CPG clip"
    os.makedirs(os.path.dirname(CENTRE_NPZ), exist_ok=True)
    tmp = CENTRE_NPZ[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, actions=coef[0][None, :], source=CENTRE_SRC)
    os.replace(tmp, CENTRE_NPZ)
    return coef[0]


def heldout_window(i, c):
    """(w, start, c10 heldout ep, c08 ep, seed, t list, phase list) -- the c10 heldout copy of condition i."""
    W = np.load(os.path.join(C10_WALKS, f"{c}.npz"), allow_pickle=True)
    w = [k for k in range(4) if ROLES[(k + i) % 4] == "heldout"][0]
    st = int(W["window_starts"][w])
    ep10 = 40000 + 10 * i + w
    v = json.load(open(POINTS))[f"data/counterfactual_walks/c10_clips_heldout/hexapod_ep{ep10}.npz"]
    assert v["ep"] == ep10 and v["cond_index"] == i
    return w, st, ep10, EP0 + 10 * i + w, 200 + i, [int(x) for x in v["t"]], [float(x) for x in v["phase"]]


def main_path(ep):
    return os.path.join(MAIN, f"hexapod_ep{ep}.npz")


def cf_path(ep, t, ci):
    return os.path.join(CF, f"hexapod_ep{code_of(ep, t, ci)}.npz")


# ------------------------------------------------------------------------------------------------ physics
def physics(sim, port, i, c, log):
    """Walk + 72 branches of condition c on this instance (one loaded scene); class logic of
    collect_c10_walks_and_branches.physics."""
    from scene_reuse import SceneReuse, state_hash
    W = dict(np.load(os.path.join(C10_WALKS, f"{c}.npz"), allow_pickle=True))
    N = len(W["actions"])
    plan = {k: np.asarray(W[f"plan_{k}"], float) for k in KEYS}
    for k in KEYS:
        assert np.all(plan[k] == float(COND[c][1][k])), (c, k)
    w, st, ep10, ep, seed, ts, phs = heldout_window(i, c)
    for t, ph in zip(ts, phs):
        assert W["cpg_phase"][st + t] == ph, "c10 branch-point phase mismatch"
    jobs = [(t, ph, ci) for t, ph in zip(ts, phs) for ci in range(24)]
    centre = np.load(CENTRE_NPZ, allow_pickle=True)["actions"].astype(np.float64).mean(0)
    cpg_kw = dict(cycles=BASE * N / EP, mirror_joints=(0, 1, 2), spin_amp=0.25, symmetric=False, legtune=None)
    plans = [plan] + [hex_plan(W, st + t, ci)[0] for (t, ph, ci) in jobs]
    R = SceneReuse(sim, SCENE, DRIVE_KW)
    t0 = time.time()
    cmds = R.commands(plans, N, centre, cpg_kw)
    for (t, ph, ci), cm in zip(jobs, cmds[1:]):
        T = st + t
        assert np.array_equal(cm[:T + 1], cmds[0][:T + 1]), "branch prefix commands differ"
        if ci == i:
            assert np.array_equal(cm, cmds[0]), "own-command cross-fade not bit-identical to the walk"
    log(f"{c}: {len(cmds)} command sets in {time.time() - t0:.0f} s; warm-up")
    ref0, nwarm, hashes = R.warm_up(cmds[0], min_runs=WARM_MIN, max_runs=40, n_same=3, short_runs=SHORT_WARM, log=log)
    refs = {state_hash(ref0): ref0}
    first_seen = {state_hash(ref0): R.n_runs}
    walk_runs = [(R.n_runs, state_hash(ref0))]
    res = {}
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
                return
        h = state_hash(out)
        refs[h] = out; first_seen[h] = R.n_runs
        walk_runs.append((R.n_runs, h))
        log(f"{c}: new class {h} at run {R.n_runs} ({len(refs)} classes)")

    while True:
        best = max(res, key=lambda h: len(res[h])) if res else next(iter(refs))
        todo = [k for k in range(len(jobs)) if k not in res.get(best, {})]
        if not todo:
            break
        if R.n_runs > 1500:
            raise RuntimeError(f"{c}: no class completed in 1500 runs: {[(h, len(v)) for h, v in res.items()]}")
        for k in todo:
            t, ph, ci = jobs[k]
            T = st + t
            a = T - PREFIX
            h = None
            for tr in range(3):
                out = R.run(cmds[1 + k][:T + BRANCH + 1], state_from=a)
                n_runs_branch += 1
                h = classify(out, a, T)
                if h is None:
                    n_unmatched += 1
                    new_walk()
                    h = classify(out, a, T)
                if h is not None:
                    break
            if h is None:
                log(f"   {c} t{t} c{ci}: prefix matches no class walk after 3 tries")
                continue
            if ci == i and not R.same_as(out, refs[h], range(a, T + BRANCH + 1), a_from=a, keys=CMP):
                raise RuntimeError(f"{c}: own-command branch != walk of its class {h}")
            res.setdefault(h, {})[k] = (out, tr + 1)
            nb = len(res[h])
            if nb % 24 == 0:
                log(f"{c}: class {h}: {nb}/{len(jobs)} branches; {R.n_runs} runs, {time.time() - t0:.0f} s")
            if nb == len(jobs):
                break
            if h != best and len(res[h]) > len(res.get(best, {})):
                break
    cls = max(res, key=lambda h: len(res[h]))
    ref, got = refs[cls], res[cls]
    res_arr = {key: [] for key in CMP}
    meta = dict(t=[], T=[], ci=[], phase=[], tries=[], own_equal=[])
    for k, (t, ph, ci) in enumerate(jobs):
        out, tries = got[k]
        T = st + t
        a = T - PREFIX
        for key in CMP:
            v = np.asarray(out[key])
            res_arr[key].append(v[a:a + NB] if not (key.startswith("state_") or key == "com_pos") else v[:NB])
        own_eq = bool(ci == i and R.same_as(out, ref, range(a, T + BRANCH + 1), a_from=a, keys=CMP))
        for kk, vv in zip(meta, (t, T, ci, ph, tries, own_eq)):
            meta[kk].append(vv)
    det = dict(det_class=cls, det_instance=f"port{port}", det_instance_pid=_instance_pid(port),
               det_class_first_run=first_seen[cls], det_warmup_runs=nwarm, det_classes_seen=len(refs),
               det_runs=R.n_runs, det_branch_runs=n_runs_branch, det_unmatched_prefix=n_unmatched)
    save_walk(W, ref, i, c, N, det)
    save_atomic(os.path.join(PHYS, f"{c}.npz"), **{k: np.asarray(v) for k, v in res_arr.items()},
                **{f"meta_{k}": np.asarray(v) for k, v in meta.items()}, **det, warm_hashes=np.array(hashes),
                walk_runs=np.array([f"{n}:{h}" for n, h in walk_runs]),
                class_counts=np.array(json.dumps({h: len(v) for h, v in res.items()})))
    log(f"{c}: physics done, class {cls} (first seen at run {first_seen[cls]}), {len(refs)} classes seen, "
        f"{R.n_runs} runs ({n_runs_branch} branch runs for {len(jobs)} branches), unmatched prefixes {n_unmatched}, "
        f"{time.time() - t0:.0f} s")


def save_walk(W, ref, i, c, N, det):
    rec = {k: W[k] for k in ("foot_order", "scale", "behavior", "schedule", "gait", "state_joint_names",
                             "state_link_names", "rec_ego", "rec_ego_box", "rec_cam_fov", "rec_spawn", "rec_warmup",
                             "dt", "condition", "cond_index", "family", "family_level", "walk_gap", "window_starts")}
    rec.update({f"plan_{k}": W[f"plan_{k}"] for k in KEYS})
    for k in CMP:
        rec[k] = np.asarray(ref[k])
    rec["head"] = rec["head"].astype(np.float32)
    rec["frames"] = np.zeros((0,), np.uint8)
    rec["step_idx"] = np.zeros((0,), np.int64)
    rec["state_link_mass"] = ref["state_link_mass"]
    for k in ("state_joint_names", "state_link_names"):
        if k in ref:
            assert [str(x) for x in ref[k]] == [str(x) for x in W[k]], k
    ph, cyc = cpg_phase(W["plan_pace"], N)
    assert np.array_equal(ph, W["cpg_phase"]), "cpg_phase differs"
    rec["cpg_phase"], rec["cpg_cycles_total"] = ph, cyc
    rec.update(det, scene_reuse=True, morph=MORPH, scene=SCENE, centre_from=CENTRE_SRC, expert_episode=-1, repeat=0,
               plan_from=os.path.relpath(os.path.join(C10_WALKS, f"{c}.npz"), ROOT))
    save_atomic(os.path.join(WALKS, f"{c}.npz"), **rec)


# ------------------------------------------------------------------------------------------------ render
def c10_heldout(ep10):
    with np.load(os.path.join(CW, "c10_clips_heldout", f"hexapod_ep{ep10}.npz"), allow_pickle=True) as d:
        return {k: d[k] for k in ("behaviour", "level", "room_seed", "window_start", "condition")}


def render_condition(sim, i, c, log):
    from render_hex_replay import render, ROOM, CAM_POSE_CONVENTION
    rec = dict(np.load(os.path.join(WALKS, f"{c}.npz"), allow_pickle=True))
    B = dict(np.load(os.path.join(PHYS, f"{c}.npz"), allow_pickle=True))
    det = {k: rec[k] for k in rec if k.startswith("det_")}
    w, st, ep10, ep, seed, ts, phs = heldout_window(i, c)
    O = c10_heldout(ep10)
    assert int(O["room_seed"]) == seed and int(O["window_start"]) == st and str(O["condition"]) == c
    per_frame = ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat",
                 "state_joint_pos", "state_link_pose", "state_sim_time", "cpg_phase", "cpg_cycles_total", "com_pos") + \
        tuple(f"plan_{k}" for k in KEYS)
    t0 = time.time()
    dst = main_path(ep)
    walk_rel = os.path.relpath(os.path.join(WALKS, f"{c}.npz"), ROOT)
    if not os.path.exists(dst):
        frames, off, Rm, cam = render(sim, rec, st, EP, seed, recentre=True, room=ROOM, return_cam=True, scene=SCENE)
        cam_local = np.asarray(sim.getObjectPose(sim.getObject("/vjepa_cam"), sim.getObject("/head")), np.float64)
        out = {k: np.array(rec[k][st:st + EP]) for k in per_frame}
        out["head"] = out["head"].astype(np.float64); out["head"][:, :2] += off
        out["head"] = out["head"].astype(np.float32)
        out["state_abdomen_pos"][:, :2] += off
        out["state_link_pose"][:, :, :2] += off
        out["com_pos"][:, :2] += off
        out.update(frames=frames, foot_order=rec["foot_order"], morph=MORPH, expert_episode=ep, repeat=w,
                   scale=rec["scale"], behavior="walk", schedule="", gait="cpg", step_idx=np.zeros((0,), np.int64),
                   state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"],
                   condition=c, behaviour=O["behaviour"], level=O["level"], family=FAMILY[i], family_level=i % 4,
                   cond_index=i, embodiment="hexapod", room_seed=seed, ego_seed=seed, room_size=float(Rm["size"]),
                   window_start=st, window_index=w, copy="heldout", split="heldout", test_only=True,
                   source_walk=walk_rel, offset_xy=off, dt=0.05, centre_from=CENTRE_SRC, render=RENDER_MAIN,
                   cam_pose=cam, cam_pose_convention=np.array(CAM_POSE_CONVENTION), cam_pose_local=cam_local,
                   cam_pose_parent="/head", c10_counterpart=f"data/counterfactual_walks/c10_clips_heldout/hexapod_ep{ep10}.npz",
                   **det)
        save_atomic(dst, **out)
    with np.load(dst, allow_pickle=True) as d:
        main = {k: d[k] for k in ("offset_xy", "com_pos", "behaviour", "condition")}
    log(f"{c}: main clip rendered ({time.time() - t0:.0f} s)")
    beh = {}
    for p in glob.glob(os.path.join(CW, "c10_clips_heldout", "*.npz")):
        with np.load(p, allow_pickle=True) as f:
            beh[int(f["cond_index"])] = str(f["behaviour"])
    N = len(rec["actions"])
    src = os.path.relpath(dst, ROOT)
    froude_h = float(np.median(main["com_pos"][:, 2]))
    n = 0
    for k in range(len(B["meta_t"])):
        t, T, ci = (int(B[f"meta_{x}"][k]) for x in ("t", "T", "ci"))
        ph = float(B["meta_phase"][k])
        out_p = cf_path(ep, t, ci)
        if os.path.exists(out_p):
            continue
        a = T - PREFIX
        plan, wgt = hex_plan(rec, T, ci)
        br = {key: np.array(B[key][k]) for key in CMP}
        br.update(state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"])
        frames, off, Rr, cam = render(sim, br, 0, NB, seed, room=8.0, return_cam=True, offset_xy=main["offset_xy"],
                                      scene=SCENE)
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
        name = ORDER[ci]
        out.update(frames=frames, cam_pose=cam, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
                   foot_order=rec["foot_order"], morph=np.array(MORPH), scale=rec["scale"], gait="cpg",
                   state_joint_names=rec["state_joint_names"], state_link_names=rec["state_link_names"],
                   condition=np.array(name), cond_index=np.array(ci), family=np.array(FAMILY[ci]),
                   family_level=np.array(ci % 4), behaviour=np.array(beh[ci]), level=np.array(ci % 4),
                   embodiment=np.array("hexapod"), expert_episode=np.array(code_of(ep, t, ci)), dt=np.float64(0.05),
                   room_seed=np.int64(seed), ego_seed=np.int64(seed), room_size=np.float64(Rr["size"]),
                   split=np.array("heldout"), test_only=np.array(True), offset_xy=off, centre_from=np.array(CENTRE_SRC),
                   segment=(np.arange(NB) >= PREFIX).astype(np.int8), first_pair=np.int64(PREFIX),
                   froude_height=np.float64(froude_h),
                   cf_source=np.array(os.path.basename(src)), cf_source_path=np.array(src), cf_source_episode=np.array(ep),
                   cf_source_condition=np.array(c), cf_source_cond_index=np.array(i), cf_source_family=np.array(FAMILY[i]),
                   cf_source_behaviour=main["behaviour"], cf_source_walk=np.array(walk_rel),
                   cf_t=np.array(t), cf_walk_frame=np.array(T), cf_window_start=np.array(st),
                   cf_branch_index=np.array(PREFIX), cf_gait_phase=np.float64(ph), cf_gait_phase_kind=np.array("cpg clock"),
                   cf_command_index=np.array(ci), cf_command_name=np.array(name), cf_target_condition=np.array(name),
                   cf_own=np.array(ci == i), cf_fade_frames=np.array(FADE),
                   cf_prefix_cmd_identical=np.array(bool(np.array_equal(br["actions"][:PREFIX + 1], rec["actions"][a:T + 1]))),
                   cf_prefix_state_identical=np.array(True), cf_prefix_tries=np.int64(B["meta_tries"][k]),
                   cf_own_equals_walk=np.array(bool(B["meta_own_equal"][k])) if ci == i else np.array(False),
                   cf_settings=np.array(json.dumps(dict(scene_reuse=True, scene=SCENE, drive=dict(DRIVE_KW, spawn=[0.0, 0.0]),
                                                        centre_from=CENTRE_SRC, walk_frames=N,
                                                        stop_after=T + BRANCH + 1,
                                                        fade="cmd=(1-w)cmd_old+w cmd_new, w=clip((frame-T)/4,0,1), pace blended, one clock"))),
                   render=np.array(RENDER_CF), **det)
        save_atomic(out_p, **out)
        n += 1
    log(f"{c}: render done ({n} new branch files, {time.time() - t0:.0f} s)")


def condition_done(i, c):
    w, st, ep10, ep, seed, ts, phs = heldout_window(i, c)
    return os.path.exists(os.path.join(PHYS, f"{c}.npz")) and os.path.exists(main_path(ep)) and \
        all(os.path.exists(cf_path(ep, t, ci)) for t in ts for ci in range(24))


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


def do_drive(a):
    log_dir = os.path.join(WALKS, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(MAIN, exist_ok=True); os.makedirs(CF, exist_ok=True)
    centre_pose()
    conds = a.conds or ORDER
    lens = {c: len(np.load(os.path.join(C10_WALKS, f"{c}.npz"))["actions"]) for c in conds}
    shares = [[] for _ in a.ports]
    for k, c in enumerate(sorted(conds, key=lambda c: -lens[c])):
        r = k % (2 * len(a.ports))
        shares[r if r < len(a.ports) else 2 * len(a.ports) - 1 - r].append(c)
    for port in a.ports:
        if _instance_pid(port) < 0:
            start_instance(port, log_dir)
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
                kill_instance(port)
                for f in glob.glob(os.path.join(CW, "c08_*", "*.tmp*.npz")) + glob.glob(os.path.join(PHYS, "*.tmp*.npz")):
                    os.remove(f)
                start_instance(port, log_dir)
                with open(lg, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} RESTART instance {port}\n")
                launch(s)
    print(f"all workers done; restarts {restarts}", flush=True)
    if a.stop:
        for port in a.ports:
            kill_instance(port)
        print("instances stopped", flush=True)


# ------------------------------------------------------------------------------------------------ check
def do_check(a):
    """Main-clip gates: loader, rooms 200-223, window / phases = c10 heldout, CoM from the c08 masses, falls,
    achieved CoM Froude per condition vs c10 (same window of the c10 det walk, and the c10 4-window mean)."""
    import wm.data.embodiment as E
    from wm.data.com import hex_com
    from beh24_conditions import tilt_deg
    sign_ch = {"fwd": (0, 1), "bwd": (0, -1), "turn_left": (2, 1), "turn_right": (2, -1), "side_L": (1, 1), "side_R": (1, -1)}
    fails = []
    files = sorted(glob.glob(os.path.join(MAIN, "*.npz")))
    files = [p for p in files if ".tmp" not in p]
    print(f"c08 main heldout files: {len(files)} (want 24); branch files: "
          f"{len([p for p in glob.glob(os.path.join(CF, '*.npz')) if '.tmp' not in p])} (want 1728)")
    if len(files) != 24:
        fails.append("count")
    c10_all = {}
    for s in ("train", "val", "heldout"):
        for p in glob.glob(os.path.join(CW, f"c10_clips_{s}", "*.npz")):
            with np.load(p, allow_pickle=True) as d:
                c10_all.setdefault(int(d["cond_index"]), []).append(E.load(p, E.HEXAPOD)["body_motion"].mean(0))
    rows, com_err, com_err_walk, seeds = [], 0.0, 0.0, []
    zmin_rel, tilt_max = [], []
    print("\ncond | c08 clip-mean CoM Froude (fwd lat yaw) | c10 same window | c10 4-window mean | c08/c10 on the "
          "dominant channel | sign | min CoM z / median | max tilt")
    for p in files:
        clip = E.load(p, E.HEXAPOD)
        with np.load(p, allow_pickle=True) as d:
            i, c, seed = int(d["cond_index"]), str(d["condition"]), int(d["room_seed"])
            assert str(d["morph"]) == MORPH
            ok = np.isfinite(clip["body_motion"]).all() and d["frames"].shape == (EP, 256, 256, 3)
            if not ok:
                fails.append(f"loader {c}")
            com_err = max(com_err, float(np.abs(hex_com(d["state_link_names"], d["state_link_pose"], morph=MORPH)
                                                - d["com_pos"]).max()))
            z = d["com_pos"][:, 2]
            zr = float(z.min() / np.median(z))
            tl = float(tilt_deg(d["state_abdomen_quat"]).max())
            c10p = str(d["c10_counterpart"])
            w, st, ep10, ep, sd, ts, phs = heldout_window(i, c)
            ok_win = int(d["window_start"]) == st and np.array_equal(d["cpg_phase"], np.load(
                os.path.join(ROOT, c10p))["cpg_phase"])
            if not ok_win:
                fails.append(f"window {c}")
        seeds.append(seed)
        zmin_rel.append(zr); tilt_max.append(tl)
        m08 = clip["body_motion"].astype(np.float64).mean(0)
        m10 = E.load(os.path.join(ROOT, c10p), E.HEXAPOD)["body_motion"].astype(np.float64).mean(0)
        m10a = np.mean(c10_all[i], 0)
        ch, sg = sign_ch[FAMILY[i]]
        sign_ok = bool(np.sign(m08[ch]) == sg)
        ratio = m08[ch] / m10[ch]
        rows.append((i, c, m08, m10, m10a, ratio, sign_ok, zr, tl))
        f = lambda v: " ".join(f"{x:+.3f}" for x in v)  # noqa: E731
        flag = "" if sign_ok and abs(ratio) >= 0.25 else "  <- barely moves / wrong sign" if not sign_ok or abs(ratio) < 0.25 else ""
        print(f"{i:2d} {c:<16} {f(m08)} | {f(m10)} | {f(m10a)} | {ratio:+.2f} | {'ok' if sign_ok else 'WRONG'} | "
              f"{zr:.2f} | {tl:.1f}{flag}")
    # walks: CoM from masses, falls over the whole walk
    walk_z, walk_tilt = [], []
    for c in ORDER:
        with np.load(os.path.join(WALKS, f"{c}.npz"), allow_pickle=True) as d:
            com_err_walk = max(com_err_walk, float(np.abs(hex_com(d["state_link_names"], d["state_link_pose"], morph=MORPH)
                                                          - d["com_pos"]).max()))
            z = d["com_pos"][:, 2]
            walk_z.append(float(z.min() / np.median(z))); walk_tilt.append(float(tilt_deg(d["state_abdomen_quat"]).max()))
    print(f"\nCoM: com_pos (collector, live scene masses) vs wm.data.com.hex_com(morph={MORPH}) max |d| main {com_err:.2e} m, "
          f"walks {com_err_walk:.2e} m")
    if max(com_err, com_err_walk) > 1e-9:
        fails.append("com")
    print(f"rooms: {sorted(seeds) == list(range(200, 224))} (seeds {min(seeds)}-{max(seeds)})")
    if sorted(seeds) != list(range(200, 224)):
        fails.append("rooms")
    print(f"falls: main clips min CoM z / median {min(zmin_rel):.3f}, max tilt {max(tilt_max):.1f} deg; whole walks "
          f"min CoM z / median {min(walk_z):.3f}, max tilt {max(walk_tilt):.1f} deg")
    if min(zmin_rel) < 0.5 or max(tilt_max) > 45:
        fails.append("falls")
    weak = [r[1] for r in rows if (not r[6]) or abs(r[5]) < 0.25]
    print(f"conditions where c08 moves < 25% of c10 on the dominant channel or with the wrong sign: {weak or 'none'}")
    np.save(os.path.join(WALKS, "achieved_froude.npy"), np.array([r[2] for r in sorted(rows)]))
    print("\nCHECK", "PASS" if not fails else f"FAIL {fails}")


# ------------------------------------------------------------------------------------------------ video
def do_video(a):
    """results/check/c08_det_samples.mp4: (1) c10 vs c08, same heldout condition and room, side by side, for 4
    conditions; (2) one c08 branch group, 4 of the 24 commands. Froude traces: fwd red, lat green, yaw blue."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import wm.data.embodiment as E
    W_, H_ = 256, 100
    cols = [(255, 80, 80), (80, 200, 80), (90, 140, 255)]

    def trace(bm, t, mark=None):
        im = Image.new("RGB", (W_, H_), (20, 20, 20)); dr = ImageDraw.Draw(im)
        lo, hi = min(bm.min(), -0.05), max(bm.max(), 0.05)
        y = lambda v: H_ - 4 - (v - lo) / (hi - lo) * (H_ - 22)  # noqa: E731
        x = lambda k: 4 + k * (W_ - 8) / (len(bm) - 1)  # noqa: E731
        dr.line([(0, y(0)), (W_, y(0))], fill=(80, 80, 80))
        if mark is not None:
            dr.line([(x(mark), 16), (x(mark), H_)], fill=(200, 200, 0))
        for ch in range(3):
            dr.line([(x(k), y(bm[k, ch])) for k in range(len(bm))], fill=cols[ch], width=2)
        dr.line([(x(t), 16), (x(t), H_)], fill=(255, 255, 255))
        dr.text((3, 2), f"Froude fwd {bm[t, 0]:+.3f} lat {bm[t, 1]:+.3f} yaw {bm[t, 2]:+.3f}", fill=(230, 230, 230))
        return np.asarray(im)

    def tile(fr, lines, bm, t, mark=None):
        im = Image.fromarray(fr); dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W_ - 1, 12 * len(lines) + 2], fill=(0, 0, 0))
        for k, (s, col) in enumerate(lines):
            dr.text((3, 2 + 12 * k), s, fill=col)
        return np.concatenate([np.asarray(im), trace(bm, t, mark)], 0)

    def head(img, title):
        h = Image.new("RGB", (img.shape[1], 18), (40, 40, 40)); ImageDraw.Draw(h).text((4, 3), title, fill=(255, 255, 255))
        return np.concatenate([np.asarray(h), img], 0)

    dst = os.path.join(ROOT, "results/check/c08_det_samples.mp4")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    wr = imageio.get_writer(dst, fps=8)
    pair_conds = ["speed_c7.1", "turn_s0.29", "side_L_lvl2", "speed_c8.15_bwd"]
    blocks = []
    for c in pair_conds:
        i = ORDER.index(c)
        w, st, ep10, ep, seed, ts, phs = heldout_window(i, c)
        p10 = os.path.join(CW, "c10_clips_heldout", f"hexapod_ep{ep10}.npz")
        p08 = main_path(ep)
        blocks.append((c, seed, [(np.load(p10)["frames"], "c10f10t10 (trained body)", E.load(p10, E.HEXAPOD)["body_motion"]),
                                 (np.load(p08)["frames"], "c08f09t09 (held-out body)", E.load(p08, E.HEXAPOD)["body_motion"])]))
    for k in range(0, len(blocks), 2):
        pair = blocks[k:k + 2]
        for t in list(range(EP)) + [EP - 1] * 4:
            img = np.concatenate([tile(fr[t], [(f"{c} room {seed}", (255, 255, 255)), (name, (255, 255, 0))], bm, t)
                                  for c, seed, two in pair for fr, name, bm in two], 1)
            wr.append_data(head(img, "same heldout condition, same room, same window: c10 vs c08 (CoM Froude)"))
    c = "side_L_lvl2"
    i = ORDER.index(c)
    w, st, ep10, ep, seed, ts, phs = heldout_window(i, c)
    t_b = ts[1]
    cmds = [c, "speed_c5.8", "turn_s0.15_neg", "speed_c7.1_bwd"]
    data = []
    for cn in cmds:
        p = cf_path(ep, t_b, ORDER.index(cn))
        with np.load(p, allow_pickle=True) as f:
            data.append((f["frames"], cn, bool(f["cf_own"]), E.load(p, E.HEXAPOD)["body_motion"]))
    same = all(np.array_equal(d[0][:PREFIX + 1], data[0][0][:PREFIX + 1]) for d in data)
    for t in list(range(NB)) + [NB - 1] * 6:
        img = np.concatenate([tile(fr[t], [(f"c08 {c} ->", (255, 255, 255)),
                                           (f"{name}{' (own = no switch)' if own else ''}", (255, 255, 0) if own else (255, 255, 255)),
                                           (f"t{t - PREFIX:+d} " + ("prefix" if t < PREFIX else "BRANCH" if t == PREFIX else "new command"),
                                            (200, 200, 200))], bm, t, PREFIX) for fr, name, own, bm in data], 1)
        wr.append_data(head(img, f"c08 branch group: {c} heldout room {seed}, branch at window frame {t_b}; prefix identical: {same}"))
    wr.close()
    print("->", os.path.relpath(dst, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("drive", "worker", "check", "video"))
    ap.add_argument("--ports", type=int, nargs="+")
    ap.add_argument("--port", type=int)
    ap.add_argument("--conds", nargs="+", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--stall", type=float, default=600.0)
    ap.add_argument("--max_restarts", type=int, default=12)
    ap.add_argument("--stop", action="store_true", help="drive: kill the instances when all workers are done")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(PHYS, exist_ok=True)
    if a.step == "drive":
        do_drive(a)
    elif a.step == "check":
        do_check(a)
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
