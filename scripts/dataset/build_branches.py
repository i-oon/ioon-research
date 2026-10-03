"""DATA_PLAN v2 stage 3: counterfactual branches of the main clips, both bodies
-> data/counterfactual_walks/b1_branches_{train,val,heldout} (current B1 branches) and, for the hexapod,
data/counterfactual_walks/_superseded/c10_replay_noise/hex_cf_* (superseded by collect_c10_walks_and_branches.py, F305).
Branch points (step `points`) -> data/counterfactual_walks/branch_points.json, keyed by the stage-1/2 source clips
(SRC_DIR below, now under _superseded/; the current c10 / b1 clips share their episode numbers).

Per main clip: 3 branch points t (window frame, 10 <= t <= 45), chosen so the gait phase at t is spread
(targets u + k/3, k = 0, 1, 2, with a per-clip offset u = frac(0.618 * clip) / 3 so the phases fill the cycle
over the split; best triple with >= 6 frames between points). Hexapod phase = the CPG clock `cpg_phase`
(the CF lift oscillator, cycles mod 1); B1 phase = foot-contact phase (front-right touchdowns of the walk,
`collect_b1_walks.contact_phase`, at the policy step of frame t).

At every point, all 24 commands of the body (its own 24 matched conditions, command index = condition index
of collect_c10_replay_superseded.ORDER; the clip's own index = no-switch control). Output = 31 frames: window frames
t-10 .. t (prefix, branch frame at index 10) + 20 frames of the new command; `segment` 0/1 (1 from index 10;
the new command acts from the step after frame t), `first_pair` = 10, `froude_height` = source clip median
CoM z.

Hexapod: the source walk is replayed from frame 0 (physics only, `collect_ik.py` with the walk's own plan /
arguments; `--stop_after`) up to window frame t + 20; from the step after t the command cross-fades
causally over 4 frames, cmd = (1 - w) cmd_old + w cmd_new, w = 0.25, 0.5, 0.75, 1 at t+1..t+4, both
commands on the same continuous gait clock (`cpg_commands(xfade=...)`; pace = blended rate). Prefix
commands are checked bit-identical to the source walk. Rendered with render_hex_replay (links mode) in the
source's room (seed) with the source window's re-centring (`offset_xy`); prefix frames are the branch's own.

B1: the source walk's MuJoCo rollout is reproduced (build_b1_cf_branches.Roll, the walk's own settings,
checked against the walk's joint_pos) and snapshotted at frame t's policy step; each command is run from the
exact restored state. Poses translated by the source window's offset (orientation kept, as the main clips),
rendered with render_b1_replay.ego_setup / pose_and_capture in the source room; prefix + branch frame = the
source's frames.

    .venv/bin/python3 scripts/dataset/build_branches.py points
    .venv/bin/python3 scripts/dataset/build_branches.py hex --ports 23110 23120 ...
    .venv/bin/python3 scripts/dataset/build_branches.py b1 --ports 23110 ...
    .venv/bin/python3 scripts/dataset/build_branches.py check
    .venv/bin/python3 scripts/dataset/build_branches.py video
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render", "sim/scene"):
    sys.path.insert(0, os.path.join(ROOT, p))
from collect_c10_replay_superseded import ORDER, FAMILY, COMMON, LIVE_ROOM, KEYS  # noqa: E402
from collect_switch_hex import COND  # noqa: E402

CW = os.path.join(ROOT, "data/counterfactual_walks")
PREFIX, BRANCH, FADE = 10, 20, 4
TMIN, TMAX = PREFIX, 65 - BRANCH
NB = PREFIX + 1 + BRANCH
SPLITS = ("train", "val", "heldout")
POINTS = os.path.join(CW, "branch_points.json")
PY = os.path.join(ROOT, ".venv/bin/python3")
CENTRE_NPZ = os.path.join(CW, "c10_walks/_work/centre_pose.npz")   # fitted c10 centre pose (hex_main_walks moved to _superseded/c10_replay_noise)
PER = 165
KEEP = np.unique(np.round(np.arange(0, PER, 2.5)).astype(int))


SRC_DIR = {"hex": "_superseded/c10_replay_noise/hex_main",     # stage-1 replay clips (branch-point keys)
           "b1": "_superseded/b1_camera_yawed/b1_main"}         # stage-2 cut, legacy camera mount (re-rendered -> b1_clips_*)
OUT_DIR = {"hex": "_superseded/c10_replay_noise/hex_cf", "b1": "b1_branches"}


def main_files(body, split):
    return sorted(glob.glob(os.path.join(CW, f"{SRC_DIR[body]}_{split}", "*.npz")))


def code_of(ep, t, ci):
    return ep * 10000 + t * 100 + ci


def out_path(body, split, ep, t, ci):
    pre = "hexapod" if body == "hex" else "b1"
    return os.path.join(CW, f"{OUT_DIR[body]}_{split}", f"{pre}_ep{code_of(ep, t, ci)}.npz")


# ----------------------------------------------------------------------------------------------- points
def circ(a, b):
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1 - d)


def choose(phase, u):
    """phase[t] for t in 0..65 -> 3 branch frames (sorted by target)."""
    targets = [(u + k / 3) % 1.0 for k in range(3)]
    ts = np.arange(TMIN, TMAX + 1)
    A, B, C = np.meshgrid(ts, ts, ts, indexing="ij")
    S = np.sort(np.stack([A, B, C], -1), -1)
    okm = (S[..., 1] - S[..., 0] >= 6) & (S[..., 2] - S[..., 1] >= 6)
    cost = sum(circ(phase[X], g) for X, g in zip((A, B, C), targets))
    cost = np.where(okm, cost, np.inf)
    k = np.unravel_index(int(np.argmin(cost)), cost.shape)
    best = (cost[k], (int(ts[k[0]]), int(ts[k[1]]), int(ts[k[2]])))
    return list(best[1]), targets


def hex_phase(d):
    return np.asarray(d["cpg_phase"], float)


def b1_phase(d):
    from collect_b1_walks import contact_phase
    W = np.load(os.path.join(ROOT, str(d["source_walk"])), allow_pickle=True)
    st = int(d["window_start"])
    ph, P = contact_phase(W["foot_contact"], [st + int(k) for k in KEEP])
    return ph, P


def do_points(a):
    out = {}
    for body in ("hex", "b1"):
        for split in SPLITS:
            for ci, p in enumerate(main_files(body, split)):
                with np.load(p, allow_pickle=True) as d:
                    ph, P = (hex_phase(d), None) if body == "hex" else b1_phase(d)
                    cond_i = int(d["cond_index"])
                    ep = int(d["expert_episode"])
                u = ((0.6180339887 * (cond_i * 4 + ci % 4 + 7 * SPLITS.index(split))) % 1.0) / 3
                ts, tg = choose(ph, u)
                out[os.path.relpath(p, ROOT)] = dict(body=body, split=split, ep=ep, cond_index=cond_i,
                                                     t=ts, phase=[float(ph[t]) for t in ts], target=tg,
                                                     period=P)
    json.dump(out, open(POINTS, "w"), indent=1)
    for body in ("hex", "b1"):
        ph = np.array([x for v in out.values() if v["body"] == body for x in v["phase"]])
        err = np.array([circ(np.array(v["phase"]), np.array(v["target"])).max() for v in out.values() if v["body"] == body])
        print(f"{body}: {len(ph)} points; phase histogram (10 bins) {np.histogram(ph, 10, (0, 1))[0].tolist()}; "
              f"max |phase - target| {err.max():.3f}")


def load_points():
    return json.load(open(POINTS))


# ----------------------------------------------------------------------------------------------- hexapod
_walk_cache = {}
_lock = threading.Lock()


def walk(path):
    with _lock:
        if path not in _walk_cache:
            _walk_cache[path] = dict(np.load(os.path.join(ROOT, path), allow_pickle=True))
        return _walk_cache[path]


def hex_plan(W, T, ci_new):
    """Full-length plan of the walk + cross-fade to condition ci_new from the step after frame T."""
    N = len(W["actions"])
    w = np.clip((np.arange(N) - T) / FADE, 0.0, 1.0)
    new = COND[ORDER[ci_new]][1]
    plan = {k: np.asarray(W[f"plan_{k}"], float).copy() for k in KEYS}
    pace_new = np.full(N, float(new["pace"]))
    plan["pace"] = (1.0 - w) * plan["pace"] + w * pace_new
    plan["xf_w"] = w
    for k in KEYS:
        if k != "pace":
            plan[f"xf_{k}"] = np.full(N, float(new[k]))
    return plan, w


def hex_physics(W, T, ci_new, port, work):
    i = int(W["cond_index"])
    N = len(W["actions"])
    plan, w = hex_plan(W, T, ci_new)
    os.makedirs(work, exist_ok=True)
    for f in glob.glob(os.path.join(work, "c10f10t10_*.npz")):
        os.remove(f)
    pf = os.path.join(work, "plan.json")
    json.dump({k: v.tolist() for k, v in plan.items()}, open(pf, "w"))
    cmd = [PY, os.path.join(ROOT, "sim/collect/collect_ik.py"), "--port", str(port),
           "--morphs", "c10f10t10=medauroidea_c10f10t10.ttt", "--episodes", str(40000 + 10 * i),
           "--ego_seed", "0", "--plan", pf, "--out", work, "--centre_from", CENTRE_NPZ,
           "--record_state", "--no_frames", "--frames", str(N), "--stop_after", str(T + BRANCH + 1),
           "--record_from", str(T - PREFIX)] + COMMON + \
        ["--ego_box", str(LIVE_ROOM)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    src = glob.glob(os.path.join(work, "c10f10t10_*.npz"))[0]
    rec = dict(np.load(src, allow_pickle=True))
    assert len(rec["actions"]) == T + BRANCH + 1, (len(rec["actions"]), T)
    return rec, plan, w


PER_FRAME_HEX = ("actions", "forces", "head", "body_quat", "state_abdomen_pos", "state_abdomen_quat",
                 "state_joint_pos", "state_link_pose", "state_sim_time", "cpg_phase", "cpg_cycles_total", "com_pos")


_hb = {}


def hex_behaviour():
    """cond_index -> the `behaviour` tag the hexapod main clips carry."""
    if not _hb:
        for s in SPLITS:
            for p in main_files("hex", s):
                with np.load(p, allow_pickle=True) as f:
                    _hb[int(f["cond_index"])] = str(f["behaviour"])
    return _hb


def hex_job(job, port, sim, work):
    from render_hex_replay import render, CAM_POSE_CONVENTION
    srcp, split, t, ci, phase = job
    with np.load(os.path.join(ROOT, srcp), allow_pickle=True) as f:
        d = {k: f[k] for k in f.files if k != "frames"}
    ep, st, i = int(d["expert_episode"]), int(d["window_start"]), int(d["cond_index"])
    dst = out_path("hex", split, ep, t, ci)
    if os.path.exists(dst):
        return "skip"
    W = walk(str(d["source_walk"]))
    T = st + t
    t0 = time.time()
    rec, plan, w = hex_physics(W, T, ci, port, work)
    tp = time.time() - t0
    # prefix commands (frames 0..T, i.e. everything up to and including the branch frame) bit-identical
    pre_ok = bool(np.array_equal(rec["actions"][:T + 1], W["actions"][:T + 1]))
    if not pre_ok:
        raise RuntimeError(f"{srcp} t{t} c{ci}: prefix commands differ from the source walk")
    a = T - PREFIX
    # state_* / com_pos rows start at frame `state_from` (= a, --record_from); everything else at frame 0
    sf = int(rec.get("state_from", 0))
    for k in [k for k in rec if k.startswith("state_") and k not in ("state_joint_names", "state_link_names",
                                                                      "state_link_mass", "state_from")] + ["com_pos", "cam_pose"]:
        if k in rec and np.ndim(rec[k]) > 0 and sf:
            pad = np.zeros((sf,) + np.shape(rec[k])[1:], np.asarray(rec[k]).dtype)
            rec[k] = np.concatenate([pad, rec[k]])             # placeholder rows before a, never used
    assert len(rec["state_link_pose"]) == T + BRANCH + 1
    frames, off, R, cam = render(sim, rec, a, NB, int(d["room_seed"]), room=8.0, return_cam=True,
                                 offset_xy=d["offset_xy"])
    out = {k: np.asarray(rec[k][a:a + NB]) for k in PER_FRAME_HEX}
    out["head"] = out["head"].astype(np.float64); out["head"][:, :2] += off; out["head"] = out["head"].astype(np.float32)
    out["state_abdomen_pos"][:, :2] += off
    out["state_link_pose"][:, :, :2] += off
    out["com_pos"][:, :2] += off
    for k in KEYS:
        out[f"plan_{k}"] = plan[k][a:a + NB]
        if k != "pace":
            out[f"xf_{k}"] = plan[f"xf_{k}"][a:a + NB]
    out["xf_w"] = w[a:a + NB]
    froude_h = float(np.median(np.asarray(d["com_pos"])[:, 2]))
    name = ORDER[ci]
    beh_src = str(d["behaviour"])
    out.update(frames=frames, cam_pose=cam, cam_pose_convention=np.array(CAM_POSE_CONVENTION),
               foot_order=d["foot_order"], morph=d["morph"], scale=d["scale"], gait="cpg",
               state_joint_names=d["state_joint_names"], state_link_names=d["state_link_names"],
               condition=np.array(name), cond_index=np.array(ci), family=np.array(FAMILY[ci]),
               family_level=np.array(ci % 4), behaviour=np.array(hex_behaviour()[ci]), level=np.array(ci % 4),
               embodiment=np.array("hexapod"), expert_episode=np.array(code_of(ep, t, ci)), dt=np.float64(0.05),
               room_seed=d["room_seed"], ego_seed=d["room_seed"], room_size=np.float64(R["size"]), split=np.array(split),
               offset_xy=off, centre_from=d["centre_from"],
               segment=(np.arange(NB) >= PREFIX).astype(np.int8), first_pair=np.int64(PREFIX),
               froude_height=np.float64(froude_h),
               cf_source=np.array(os.path.basename(srcp)), cf_source_path=np.array(srcp), cf_source_episode=np.array(ep),
               cf_source_condition=d["condition"], cf_source_cond_index=np.array(i), cf_source_family=np.array(FAMILY[i]),
               cf_source_behaviour=np.array(beh_src), cf_source_walk=d["source_walk"],
               cf_t=np.array(t), cf_walk_frame=np.array(T), cf_window_start=np.array(st),
               cf_branch_index=np.array(PREFIX), cf_gait_phase=np.float64(phase), cf_gait_phase_kind=np.array("cpg clock"),
               cf_command_index=np.array(ci), cf_command_name=np.array(name), cf_target_condition=np.array(name),
               cf_own=np.array(ci == i), cf_fade_frames=np.array(FADE), cf_prefix_cmd_identical=np.array(pre_ok),
               cf_settings=np.array(json.dumps(dict(collect_ik=COMMON + ["--ego_box", str(LIVE_ROOM), "--ego_seed", "0",
                                                                        "--episodes", str(40000 + 10 * i)],
                                                    centre_from=os.path.relpath(CENTRE_NPZ, ROOT), walk_frames=len(W["actions"]),
                                                    stop_after=T + BRANCH + 1, fade="cmd=(1-w)cmd_old+w cmd_new, "
                                                    "w=clip((frame-T)/4,0,1), pace blended, one clock"))),
               render=np.array("render_hex_replay links mode, source room + source window offset, fov 90, room 8 m"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)
    return f"phys {tp:.1f}s total {time.time() - t0:.1f}s"


def hex_jobs(splits, limit_sources=None, only_cmds=None):
    P = load_points()
    jobs = []
    for srcp, v in sorted(P.items()):
        if v["body"] != "hex" or v["split"] not in splits:
            continue
        if limit_sources and os.path.basename(srcp) not in limit_sources:
            continue
        for t, ph in zip(v["t"], v["phase"]):
            for ci in (only_cmds if only_cmds is not None else range(24)):
                jobs.append((srcp, v["split"], t, ci, ph))
    return jobs


def run_pool(jobs, ports, fn, tag):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    t0 = time.time()
    done = [0]
    todo = [j for j in jobs]
    # jobs of one source together on one worker (walk cache), sources dealt round-robin
    by_src = {}
    for j in todo:
        by_src.setdefault(j[0], []).append(j)
    shares = [[] for _ in ports]
    for k, (s, js) in enumerate(sorted(by_src.items())):
        shares[k % len(ports)].extend(js)

    def worker(slot):
        port = ports[slot]
        sim = RemoteAPIClient("localhost", port=port).require("sim")
        work = tempfile.mkdtemp(prefix=f"cf_{tag}_{port}_")
        for j in shares[slot]:
            for attempt in range(3):
                try:
                    msg = fn(j, port, sim, work)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  RETRY {j[:4]} port {port}: {repr(e)[:300]}", flush=True)
                    if attempt == 2:
                        raise
                    sim = RemoteAPIClient("localhost", port=port).require("sim")
            with _lock:
                done[0] += 1
                n = done[0]
            if n % 25 == 0 or n <= 3 or "legacy" in str(msg):
                el = time.time() - t0
                print(f"[{tag}] {n}/{len(todo)} {el / 60:.1f} min, eta {el / n * (len(todo) - n) / 60:.1f} min ({msg})",
                      flush=True)
    with ThreadPoolExecutor(len(ports)) as ex:
        list(ex.map(worker, range(len(ports))))
    print(f"[{tag}] DONE {len(todo)} jobs in {(time.time() - t0) / 60:.1f} min", flush=True)


COPPELIA = os.path.expanduser("~/CoppeliaSim")


def restart_instance(port, log_dir):
    """Kill the CoppeliaSim instance on `port` (and its launcher) and start a fresh one on the same port."""
    import signal
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
        if f"rpcPort={port} " in ln + " " and "grep" not in ln:
            try:
                os.kill(int(ln.split()[0]), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    time.sleep(2)
    subprocess.Popen(f"tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort={port} "
                     f"-GzmqRemoteApi.cntPort={port + 1}", shell=True, cwd=COPPELIA, start_new_session=True,
                     stdout=open(os.path.join(log_dir, f"csim_{port}.log"), "a"), stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        r = subprocess.run([PY, "-c", f"from coppeliasim_zmqremoteapi_client import RemoteAPIClient as C; "
                                      f"print(C('localhost', port={port}).require('sim').getSimulationState())"],
                           capture_output=True, text=True, timeout=30) if True else None
        if r.returncode == 0:
            return
    raise RuntimeError(f"CoppeliaSim on {port} did not come back")


def hex_one(a):
    """One branch job in its own process (called by do_hex with a timeout)."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    job = json.loads(a.job)
    sim = RemoteAPIClient("localhost", port=a.ports[0]).require("sim")
    work = tempfile.mkdtemp(prefix=f"cf_hex1_{a.ports[0]}_")
    print("HEX_ONE", hex_job(tuple(job), a.ports[0], sim, work), flush=True)


def do_hex(a):
    """Each branch runs in a child process with a timeout; a child that hangs (a CoppeliaSim instance stopped
    answering: 4 of 6 instances hung on 2026-10-02 03:10-03:45 with the in-process version) is killed, its
    instance restarted on the same port, and the job retried."""
    import signal
    jobs = [j for j in hex_jobs(a.splits, a.sources, a.cmds) if not os.path.exists(
        out_path("hex", j[1], int(os.path.basename(j[0])[len("hexapod_ep"):-4]), j[2], j[3]))]
    print(f"hexapod: {len(jobs)} branch jobs to do on ports {a.ports}", flush=True)
    log_dir = tempfile.mkdtemp(prefix="cf_hex_logs_")
    by_src = {}
    for j in jobs:
        by_src.setdefault(j[0], []).append(j)
    shares = [[] for _ in a.ports]
    for k, (src, js) in enumerate(sorted(by_src.items())):
        shares[k % len(a.ports)].extend(js)
    t0, done = time.time(), [0]

    def worker(slot):
        port = a.ports[slot]
        for j in shares[slot]:
            for attempt in range(4):
                cmd = [PY, os.path.abspath(__file__), "hex_one", "--ports", str(port), "--job", json.dumps(list(j))]
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     start_new_session=True)
                try:
                    out, _ = p.communicate(timeout=a.job_timeout)
                    if p.returncode == 0:
                        msg = [ln for ln in out.splitlines() if ln.startswith("HEX_ONE")][-1]
                        break
                    print(f"  FAIL {j[:4]} port {port} attempt {attempt}: {out[-600:]}", flush=True)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.communicate()
                    print(f"  TIMEOUT {j[:4]} port {port} attempt {attempt}: restarting the instance", flush=True)
                tmp = out_path("hex", j[1], int(os.path.basename(j[0])[len("hexapod_ep"):-4]), j[2], j[3])[:-4] + ".tmp.npz"
                if os.path.exists(tmp):
                    os.remove(tmp)
                restart_instance(port, log_dir)
            else:
                raise RuntimeError(f"job {j} failed 4 times")
            with _lock:
                done[0] += 1
                n = done[0]
            if n % 25 == 0 or n <= 3:
                el = time.time() - t0
                print(f"[hex] {n}/{len(jobs)} {el / 60:.1f} min, eta {el / n * (len(jobs) - n) / 60:.1f} min ({msg})", flush=True)
    with ThreadPoolExecutor(len(a.ports)) as ex:
        list(ex.map(worker, range(len(a.ports))))
    print(f"[hex] DONE {len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------------------------- B1
def b1_jobs(splits, limit_sources=None):
    P = load_points()
    out = []
    for srcp, v in sorted(P.items()):
        if v["body"] == "b1" and v["split"] in splits and (not limit_sources or os.path.basename(srcp) in limit_sources):
            out.append((srcp, v["split"], v["t"], None, v["phase"]))
    return out


def b1_cmd_as_run(cmd3):
    """The command exactly as the walk ran it: collect_b1_walks.rollout passes it to rollout_b1_mujoco.py on
    the command line as `--vx %.6f` etc., so the walk saw the 6-decimal value, not the stored full-precision
    tune_cmd. Continuing with the full-precision value is a command change of up to 5e-7 at the branch point
    (base 2e-5 m off the walk within 20 frames, joints 3e-7; the pre-2026-10-02 v4 B1 branches)."""
    return tuple(float(f"{float(x):.6f}") for x in cmd3)


def b1_cmds():
    tun = json.load(open(os.path.join(CW, "b1_walks/tuning.json")))
    return [b1_cmd_as_run(tun[str(i)]["cmd"]) for i in range(24)]


def b1_job(job, port, sim, work, only_cmds=None):
    """All branches of one B1 source clip (3 points x 24 commands)."""
    from build_b1_cf_branches import Roll, run_branch, POLICY_WARMUP
    from render_b1_replay import ego_setup, pose_and_capture, CAM_POSE_CONVENTION
    from collect_b1_walks import B1_NAME
    from wm.data.com import b1_com
    srcp, split, ts, _, phases = job
    srcp = remount_path(srcp)          # corrected-mount re-render of the v4 main clip (same state, new frames)
    d = dict(np.load(os.path.join(ROOT, srcp), allow_pickle=True))
    ep, st, i = int(d["expert_episode"]), int(d["window_start"]), int(d["cond_index"])
    cis = only_cmds if only_cmds is not None else list(range(24))
    if all(os.path.exists(out_path("b1", split, ep, t, ci)) for t in ts for ci in cis):
        return "skip"
    t0 = time.time()
    W = walk(str(d["source_walk"]))
    cmds = b1_cmds()
    own = b1_cmd_as_run(W["tune_cmd"])
    assert np.allclose(own, cmds[i], atol=0, rtol=0), (own, cmds[i])
    kp, ki = float(W["rollout_head_kp"]), float(W["rollout_head_ki"])
    model = os.path.join(ROOT, str(W["rollout_model"]))
    r = Roll("gait3", (kp, ki), model)
    want = dict(heading_kp=kp, heading_ki=ki, heading_ki_clip=float(W["rollout_head_ki_clip"]), policy="gait3",
                policy_checkpoint=str(W["rollout_checkpoint"]), gait_freq=float(W["rollout_gait_freq"]),
                model=str(W["rollout_model"]))
    if r.settings() != want:
        raise RuntimeError(f"Roll settings {r.settings()} != walk's {want}")
    assert int(W["rollout_warmup"]) == 25 and int(W["rollout_policy_warmup"]) == POLICY_WARMUP
    assert float(W["rollout_cmd_noise"]) == 0.0 and float(W["rollout_yaw0"]) == 0.0
    for _ in range(POLICY_WARMUP):
        r.step(*own)
    off = np.asarray(d["offset_xy"], float)
    froude_h = float(np.median(np.asarray(d["com_pos"])[:, 2]))
    root_h, joints, cam = ego_setup(sim, os.path.join(ROOT, "sim/env/b1_flat.ttt"), d["base_pos"][0], d["base_quat"][0],
                                    int(d["room_seed"]))
    # reproduce the rollout up to the last branch point, checking every recorded step against the walk
    steps = {t: st + int(KEEP[t]) for t in ts}
    snaps, rep_err = {}, 0.0
    for s in range(max(steps.values()) + 1):
        o = r.step(*own)
        # bit-exact against the walk (float32 storage) on every recorded field, base included
        rep_err = max(rep_err, max(float(np.abs(np.asarray(o[k], np.float32) - W[k][s]).max())
                                   for k in ("base_pos", "base_quat", "joint_pos", "joint_vel", "action", "command")))
        for t, sv in steps.items():
            if sv == s:
                snaps[t] = r.snapshot()
    if rep_err != 0.0:
        raise RuntimeError(f"{srcp}: rollout does not reproduce the walk (max |dq| {rep_err:.1e})")
    msgs = []
    for t, ph in zip(ts, phases):
        # the branch frame re-rendered from the reproduced state vs the stored source frame
        f_t, _ = pose_and_capture(sim, root_h, joints, cam, d["base_pos"][t], d["base_quat"][t], d["joint_pos"][t])
        mae_t = float(np.abs(f_t.astype(int) - d["frames"][t].astype(int)).mean())
        offs = [int(v) for v in np.round(2.5 * (t + np.arange(1, BRANCH + 1))) - KEEP[t]]
        for ci in cis:
            dst = out_path("b1", split, ep, t, ci)
            if os.path.exists(dst):
                continue
            br, fell = run_branch(r, snaps[t], cmds[ci], offs)
            P = br["base_pos"].astype(np.float64).copy(); P[:, :2] += off
            Q = br["base_quat"].astype(np.float64)
            fr, cp = [], []
            for k in range(BRANCH):
                f_, c_ = pose_and_capture(sim, root_h, joints, cam, P[k].astype(np.float32), Q[k].astype(np.float32),
                                          br["joint_pos"][k].astype(np.float32))
                fr.append(f_); cp.append(c_)
            a0 = t - PREFIX
            data = {}
            for k, v in (("base_pos", P), ("base_quat", Q), ("joint_pos", br["joint_pos"]), ("joint_vel", br["joint_vel"]),
                         ("action", br["action"]), ("command", br["command"]), ("foot_contact", br["foot_contact"])):
                data[k] = np.concatenate([d[k][a0:t + 1], np.asarray(v, np.float32)]).astype(np.float32)
            data["frames"] = np.concatenate([d["frames"][a0:t + 1], np.asarray(fr, np.uint8)])
            data["com_pos"] = np.concatenate([d["com_pos"][a0:t + 1],
                                              b1_com(data["base_pos"][PREFIX + 1:], data["base_quat"][PREFIX + 1:],
                                                     data["joint_pos"][PREFIX + 1:], model)])
            data["cam_pose"] = np.concatenate([d["cam_pose"][a0:t + 1], np.asarray(cp)])
            name = B1_NAME[FAMILY[ci]][ci % 4]
            data.update(
                cam_pose_convention=np.array(CAM_POSE_CONVENTION), joint_order_sdk=d["joint_order_sdk"], dt=d["dt"],
                fps=d["fps"], condition=np.array(name), cond_index=np.array(ci), family=np.array(FAMILY[ci]),
                family_level=np.array(ci % 4), hex_condition=np.array(ORDER[ci]),
                behaviour=np.array({"fwd": "speed", "bwd": "speed", "turn_left": "turn", "turn_right": "turn"}.get(
                    FAMILY[ci], "side")), level=np.array(ci % 4), embodiment=np.array("b1"), policy=np.array("gait3"),
                expert_episode=np.array(code_of(ep, t, ci)), room_seed=d["room_seed"], ego_seed=d["room_seed"],
                split=np.array(split), offset_xy=off,
                segment=(np.arange(NB) >= PREFIX).astype(np.int8), first_pair=np.int64(PREFIX),
                froude_height=np.float64(froude_h),
                cf_source=np.array(os.path.basename(srcp)), cf_source_path=np.array(srcp),
                cf_source_episode=np.array(ep), cf_source_condition=d["condition"], cf_source_cond_index=np.array(i),
                cf_source_family=np.array(FAMILY[i]), cf_source_walk=d["source_walk"], cf_t=np.array(t),
                cf_walk_step=np.array(steps[t]), cf_window_start=np.array(st), cf_branch_index=np.array(PREFIX),
                cf_gait_phase=np.float64(ph), cf_gait_phase_kind=np.array("foot contact (FR touchdowns)"),
                cf_command_index=np.array(ci), cf_command_name=np.array(name), cf_target_condition=np.array(name),
                cf_command=np.array(cmds[ci], np.float32), cf_own=np.array(ci == i), cf_fell=np.array(fell),
                cf_replay_err=np.float64(rep_err), cf_branch_frame_rerender_mae=np.float64(mae_t),
                render=np.array("render_b1_replay ego_setup (--ego --match_floor --ground_uv_mult 1.0), source room "
                                "+ source window offset; prefix and branch frame = source frames"),
                **{k: np.array(v) for k, v in r.settings().items() if k != "policy"})
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, **data)
            os.replace(tmp, dst)
        msgs.append(f"t{t} mae {mae_t:.2f}")
    return f"{os.path.basename(srcp)} rep {rep_err:.1e} {' '.join(msgs)} {time.time() - t0:.0f}s"


def remount_path(srcp):
    """B1 sources: the camera-mount-corrected re-render of the v4 main clip (see do_b1_remount)."""
    return srcp.replace(SRC_DIR["b1"] + "_", "b1_clips_")


def b1_remount_job(job, port, sim, work):
    """Re-render one v4 B1 main clip with the corrected camera mount (render_b1_replay default rule: camera on
    the base's own forward axis) -> data/counterfactual_walks/b1_clips_<split>/ (same file name, every other
    field unchanged; frames + cam_pose new; legacy-mount frames NOT kept, they are in _superseded/b1_camera_yawed/b1_main_<split>)."""
    from render_b1_replay import ego_setup, pose_and_capture, CAM_POSE_CONVENTION
    srcp = job[0]
    dst = os.path.join(ROOT, remount_path(srcp))
    if os.path.exists(dst):
        return "skip"
    d = dict(np.load(os.path.join(ROOT, srcp), allow_pickle=True))
    msg = ""
    if job[4]:          # first clips: the in-process legacy setup reproduces the stored frames + cam_pose
        r, j, c = ego_setup(sim, os.path.join(ROOT, "sim/env/b1_flat.ttt"), d["base_pos"][0], d["base_quat"][0],
                            int(d["room_seed"]), legacy_mount=True)
        fl, cl = zip(*[pose_and_capture(sim, r, j, c, d["base_pos"][t], d["base_quat"][t], d["joint_pos"][t])
                       for t in range(len(d["frames"]))])
        msg = (f"legacy in-process vs stored: pixel MAE {np.abs(np.asarray(fl, int) - d['frames'].astype(int)).mean():.3f}"
               f" cam_pose max|d| {np.abs(np.asarray(cl)[:, :3] - d['cam_pose'][:, :3]).max():.1e}; ")
    r, j, c = ego_setup(sim, os.path.join(ROOT, "sim/env/b1_flat.ttt"), d["base_pos"][0], d["base_quat"][0],
                        int(d["room_seed"]))
    fr, cp = zip(*[pose_and_capture(sim, r, j, c, d["base_pos"][t], d["base_quat"][t], d["joint_pos"][t])
                   for t in range(len(d["frames"]))])
    d.update(frames=np.asarray(fr, np.uint8), cam_pose=np.asarray(cp, np.float64),
             cam_pose_convention=np.array(CAM_POSE_CONVENTION), cam_pose_parent=np.array("base_visual"),
             remount_from=np.array(srcp),
             render=np.array(str(d["render"]) + " | RE-RENDERED 2026-10-02 with the corrected ego mount "
                             "(render_b1_replay default: camera on the base's own forward axis; the original used "
                             "the clip's start heading -> camera yawed by that heading relative to the body)"))
    d.pop("cam_pose_local", None)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, dst)
    return msg + "ok"


def do_b1_remount(a):
    jobs = [(srcp, None, None, None, k < 2) for k, srcp in enumerate(
        sorted(os.path.relpath(p, ROOT) for s in a.splits for p in main_files("b1", s)))]
    run_pool(jobs, a.ports, b1_remount_job, "b1remount")


def _group_files(body, split, cond_index, copy, cmds):
    """The branch files of the main clip (split, condition, copy), its first branch point, for `cmds`."""
    for p in main_files(body, split):
        with np.load(p, allow_pickle=True) as f:
            if int(f["cond_index"]) == cond_index and str(f["copy"]) == copy:
                ep = int(f["expert_episode"])
                t = load_points()[os.path.relpath(p, ROOT)]["t"][0]
                return [out_path(body, split, ep, t, c) for c in cmds], t
    raise FileNotFoundError((body, split, cond_index, copy))


def do_video(a):
    """results/check/cf_branches_samples.mp4: per block, one branch point x 4 commands side by side (ego view),
    labelled, with the CoM Froude traces (fwd red, lat green, yaw blue; the branch frame marked)."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    import wm.data.embodiment as E
    W_, H_ = 256, 110
    cols = [(255, 80, 80), (80, 200, 80), (90, 140, 255)]
    ix = ORDER.index
    blocks = [  # (title, [(body, split, cond, copy)], commands)
        ("hexapod, source speed_c7.1 (train)", [("hex", "train", ix("speed_c7.1"), "train0")],
         [ix("speed_c7.1"), ix("turn_s0.56"), ix("side_R_lvl2"), ix("speed_c7.1_bwd")]),
        ("hexapod, source turn_s0.29_neg (val)", [("hex", "val", ix("turn_s0.29_neg"), "val")],
         [ix("turn_s0.29_neg"), ix("speed_c8.8"), ix("side_L_lvl3"), ix("turn_s0.29")]),
        ("B1, source speed_vx0.38 (train)", [("b1", "train", ix("speed_c7.1"), "train0")],
         [ix("speed_c7.1"), ix("turn_s0.56"), ix("side_R_lvl2"), ix("speed_c7.1_bwd")]),
        ("B1, source turn_w0.024_neg (val)", [("b1", "val", ix("turn_s0.29_neg"), "val")],
         [ix("turn_s0.29_neg"), ix("speed_c8.8"), ix("side_L_lvl3"), ix("turn_s0.29")]),
        ("matched group: hexapod (top) vs B1 (bottom), source side_L_lvl2 (heldout)",
         [("hex", "heldout", ix("side_L_lvl2"), "heldout"), ("b1", "heldout", ix("side_L_lvl2"), "heldout")],
         [ix("side_L_lvl2"), ix("speed_c5.8"), ix("turn_s0.15_neg"), ix("side_R_lvl3")]),
    ]

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

    os.makedirs(os.path.join(ROOT, "results/check"), exist_ok=True)
    dst = os.path.join(ROOT, "results/check/cf_branches_samples.mp4")
    wr = imageio.get_writer(dst, fps=6)
    for title, srcs, cmds in blocks:
        rows = []
        for body, split, ci, copy in srcs:
            paths, t_b = _group_files(body, split, ci, copy, cmds)
            spec = E.HEXAPOD if body == "hex" else E.B1
            data = []
            for p in paths:
                with np.load(p, allow_pickle=True) as f:
                    data.append((f["frames"], str(f["cf_command_name"]), str(f["cf_source_condition"]), bool(f["cf_own"]),
                                 int(f["room_seed"]), float(f["cf_gait_phase"])))
                data[-1] = data[-1] + (E.load(p, spec)["body_motion"],)
            rows.append((body, split, t_b, data))
        for t in list(range(NB)) + [NB - 1] * 6:
            out = []
            for body, split, t_b, data in rows:
                tiles = []
                for fr, name, srcc, own, seed, ph, bm in data:
                    im = Image.fromarray(fr[t]); dr = ImageDraw.Draw(im)
                    dr.rectangle([0, 0, W_ - 1, 38], fill=(0, 0, 0))
                    dr.text((3, 2), f"{'hexapod' if body == 'hex' else 'B1'} {srcc} ->", fill=(255, 255, 255))
                    dr.text((3, 14), f"{name}{' (own = no switch)' if own else ''}", fill=(255, 255, 0) if own else (255, 255, 255))
                    phase = "prefix (before switch)" if t < PREFIX else ("BRANCH FRAME" if t == PREFIX else "new command")
                    dr.text((3, 26), f"{split} room {seed} t{t - PREFIX:+d} {phase} ph {ph:.2f}", fill=(200, 200, 200))
                    tiles.append(np.concatenate([np.asarray(im), trace(bm, t, W_)], 0))
                out.append(np.concatenate(tiles, 1))
            img = np.concatenate(out, 0)
            if img.shape[0] < 2 * (W_ + H_):        # one frame size for the whole video
                img = np.concatenate([img, np.zeros((2 * (W_ + H_) - img.shape[0],) + img.shape[1:], np.uint8)], 0)
            head = Image.new("RGB", (img.shape[1], 18), (40, 40, 40)); ImageDraw.Draw(head).text((4, 3), title, fill=(255, 255, 255))
            wr.append_data(np.concatenate([np.asarray(head), img], 0))
    wr.close()
    print("->", os.path.relpath(dst, ROOT))


def do_b1(a):
    if len(a.ports) > 1:
        # **one worker PROCESS per port, never threads**: MuJoCo state is not shared-thread safe and
        # wm.data.com.b1_com keeps one module-global (MjModel, MjData) per model, so two threads computing
        # CoM at once raced on one MjData -> SIGSEGV (2026-10-02 02:03, first full run)
        procs = []
        for k, port in enumerate(a.ports):
            cmd = [PY, os.path.abspath(__file__), "b1", "--ports", str(port), "--shard", str(k), "--nshards", str(len(a.ports)),
                   "--splits", *a.splits] + (["--sources", *a.sources] if a.sources else []) + \
                  (["--cmds", *map(str, a.cmds)] if a.cmds is not None else [])
            procs.append(subprocess.Popen(cmd, cwd=ROOT))
        codes = [p.wait() for p in procs]
        print(f"B1 shards exit codes {codes}", flush=True)
        if any(codes):
            raise SystemExit(1)
        return
    jobs = b1_jobs(a.splits, a.sources)
    jobs = [j for k, j in enumerate(jobs) if k % a.nshards == a.shard]
    print(f"B1: {len(jobs)} source clips on ports {a.ports} (shard {a.shard}/{a.nshards})", flush=True)
    fn = (lambda j, port, sim, work: b1_job(j, port, sim, work, a.cmds)) if a.cmds is not None else b1_job
    run_pool(jobs, a.ports, fn, "b1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("points", "hex", "hex_one", "b1", "b1_remount", "video"))
    ap.add_argument("--ports", type=int, nargs="+", default=[23110])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--cmds", type=int, nargs="*", default=None)
    ap.add_argument("--job", default="")
    ap.add_argument("--job_timeout", type=float, default=300.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.chdir(ROOT)
    {"points": do_points, "hex": do_hex, "b1": do_b1, "b1_remount": do_b1_remount, "video": do_video, "hex_one": hex_one}[a.step](a)


if __name__ == "__main__":
    main()
