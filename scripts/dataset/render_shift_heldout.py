"""Render-shift heldout TEST set: the heldout clips / branches of c10, B1 and c08 re-rendered with the ROOM SIZE drawn
from ONE range shared by all bodies, instead of the per-body rule (hexapods: 8 m box, wall height and floor tile scaled
by the camera-mount height; B1: everything scaled by its mount height, room_for). Physics untouched: every frame is a
kinematic replay of the state stored in the source file (hexapod: links mode, `state_link_pose`; B1: base pose + 12
joints), exactly as the source was rendered (render_hex_replay.setup / render_b1_replay.ego_setup order and calls).

World scale rule (rs): size S (box width, m), wall/ceiling height 3 S / 8, floor-texture tile S / 16 -- the room_for
proportions at k = S / 8, so the whole room scales as one object; camera mount / offset / FOV 90 / pitch levelling
unchanged (they belong to the body). Textures and colours: the source file's room seed (same as before). Size: one
value per room seed (= per heldout condition), drawn once from a fixed seed (SIZE_SEED), stratified log-uniform over
size_range(); c10, B1 and c08 clips of one condition get the same size (they already share the room seed), and every
branch uses its source clip's size (all 24 commands of a group in one room, as in the source data).

Only heldout data is read; outputs are new dirs data/counterfactual_walks/rs_{c10,b1,c08}_{clips,branches}_heldout.
Every non-frame field is copied bit-for-bit, except `room_size` (set to S; hexapods had 8.0) plus the new fields
`rs_room_size_original`, `rs_room_height`, `rs_ground_uv`, `rs_size_seed`, `rs_render`. `cam_pose` is re-measured
and must equal the source (the camera does not depend on the room); kept from the source.

    PY=.venv/bin/python3; S=scripts/dataset/render_shift_heldout.py
    $PY $S launch --ports 25300 25310 25320 25330 25340 25350   # own instances, pids in results/check/render_shift/pids.json
    $PY $S probe --port 25300            # mount heights -> current room per body, chosen range
    $PY $S gate_a --ports ...            # original size through this path == stored frames
    $PY $S review --ports ...            # contact sheets at small / mid / large size
    $PY $S render --ports ...            # everything (skips existing)
    $PY $S verify
    $PY $S stop
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


import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "sim/render", "sim/scene"):
    sys.path.insert(0, os.path.join(ROOT, p))

CW = os.path.join(ROOT, "data/counterfactual_walks")
CHK = os.path.join(ROOT, "results/check/render_shift")
PIDS = os.path.join(CHK, "pids.json")
GROUPS_JSON = os.path.join(CW, "rs_branch_groups_heldout.json")
BODIES = ("c10", "b1", "c08")
HEX_SCENE = {"c10": "medauroidea_c10f10t10.ttt", "c08": "medauroidea_c08f09t09.ttt"}
B1_SCENE = os.path.join(ROOT, "sim/env/b1_flat.ttt")
COPPELIA = os.path.expanduser("~/CoppeliaSim")
SIZE_SEED = 20261005
# Shared range: see size_range() (probe: hexapods 8.0 m, B1 8 x 0.6 / 0.272 = 17.6 m).
RS_RENDER = ("render_shift: kinematic replay of the stored state, room size S from one shared range (rs_room_size), "
             "wall height 3S/8, floor tile S/16, room seed + camera + FOV 90 as the source")


# ------------------------------------------------------------------------------------------------ sizes
def b1_mount_z():
    f = os.path.join(CHK, "probe.json")
    return json.load(open(f))["b1"]["mount_z"]


def size_range():
    """lo = max(smallest current room / 1.5, floor-always-visible bound), hi = largest current room x 1.5.
    Floor bound: with FOV 90 and a level camera the floor stays in view while the wall ahead is farther than the
    camera height, so half-width >= max camera excursion from the room centre + max camera height, over all bodies
    and all heldout files, +5 %. At 8 / 1.5 = 5.33 m the B1 forward clips ended with the wall filling the frame
    (review sheet, first try): B1 excursion 2.09 m + camera height 0.94 m -> 6.36 m."""
    from ego_camera import REF_MOUNT_Z
    P = json.load(open(os.path.join(CHK, "probe.json")))
    b1 = 8.0 * P["b1"]["mount_z"] / REF_MOUNT_Z
    # 2026-10-05 user review: lo = 8.0 m (the hexapods' original room); at 6.36 m the B1 view was mostly wall
    return 8.0, b1 * 1.5


def sizes():
    """room seed (200..223) -> S, stratified log-uniform over the range (each of 24 equal log-bins once)."""
    lo, hi = size_range()
    rng = np.random.default_rng(SIZE_SEED)
    perm = rng.permutation(24)
    u = rng.random(24)
    s = np.exp(np.log(lo) + (perm + u) / 24.0 * (np.log(hi) - np.log(lo)))
    return {200 + i: float(s[i]) for i in range(24)}


def visible_floor_top(sim):
    h = sim.getObject("/Floor/box")
    return sim.getObjectPosition(h, sim.handle_world)[2] + sim.getShapeBB(h)[0][2] / 2


def hold_floor(sim, top_src):
    """rs only. **The visible floor is `/Floor/box`, not `/Floor`** (layers 128 vs 32768): scale_floor holds the
    invisible `/Floor` top at 0 and the visible box sinks by 0.1 (k - 1) -- in the source data 0.108 m (hexapod,
    8 m room) and 0.359 m (B1, 17.6 m). Kept at the source level (top_src, per body), so the camera-to-floor
    geometry is exactly the source's and only the room changes; without this the drop would grow with S (0.59 m at
    26 m). The walls are built down to that floor (rs_walls): the slot under walls based at z = 0 is the void seen
    as a thin black seam in the source frames and would be a wide band in small rooms."""
    top = visible_floor_top(sim)
    f = sim.getObject("/Floor")
    q = sim.getObjectPosition(f, sim.handle_world)
    sim.setObjectPosition(f, sim.handle_world, [q[0], q[1], q[2] + (top_src - top)])
    return visible_floor_top(sim)


def rs_far(sim, cam, S):
    """Far clipping plane >= 2 S (hexapod scene authors 20 m, B1 30 m): at S = 26.5 m a far ceiling corner is
    ~22 m from the hexapod camera and would be clipped to black."""
    far = max(float(sim.getObjectFloatParam(cam, sim.visionfloatparam_far_clipping)), 2.0 * S)
    sim.setObjectFloatParam(cam, sim.visionfloatparam_far_clipping, far)
    return far


def rs_walls(sim, R, seed, centre, top_src):
    """build_texture_box exactly as the source (same wall geometry, so the cube-mapped texture is identical), plus a
    skirt under each side wall from the visible floor (top_src <= 0) to z = 0, in the wall texture's mean colour.
    (Making the walls taller instead shifts the cube-mapped texture by half the extension: 27 % of pixels changed at
    the source size.)"""
    from ego_camera import build_texture_box, _tinted, _wall_texture
    hs = build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed, centre=centre)
    d = -float(top_src)
    if d <= 1e-6:
        return hs
    rng = np.random.default_rng(seed * 977 + 11)                  # build_texture_box's colour draw
    colours = [tuple(0.45 + 0.35 * rng.random(3)) for _ in range(5)]
    for i, h in enumerate(hs[:4]):                                 # 4 side walls; hs[4] = ceiling
        mean = _tinted(_wall_texture(seed + i), colours[i]).reshape(-1, 3).mean(0) / 255.0
        bb = sim.getShapeBB(h)[0]
        q = sim.getObjectPosition(h, sim.handle_world)
        k = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [float(bb[0]), float(bb[1]), d], 0)
        sim.setObjectPosition(k, sim.handle_world, [q[0], q[1], -d / 2])
        sim.setObjectInt32Param(k, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(k, sim.shapeintparam_respondable, 0)
        sim.setShapeColor(k, None, sim.colorcomponent_ambient_diffuse, [float(v) for v in mean])
        sim.setObjectAlias(k, f"egoSkirt{i}")
    return hs


def src_floor_top(sim, size_src):
    """Visible-floor top the source render had (scale_floor at the source room size), measured on the loaded,
    unscaled scene: authored top - authored half-thickness x (k - 1)."""
    h = sim.getObject("/Floor/box")
    bb = sim.getShapeBB(h)[0]
    k = max(1.0, size_src * 1.3 / float(bb[0]))
    if k <= 1.001:
        k = 1.0
    return visible_floor_top(sim) - bb[2] / 2 * (k - 1)


def rs_room(S):
    return dict(size=float(S), height=3.0 * S / 8.0, ground_uv=S / 16.0)


# ------------------------------------------------------------------------------------------------ render
def hex_render(sim, d, scene, override=None, frames_on=True, centre=(0.0, 0.0)):
    """render_hex_replay.setup + links-mode loop on the stored (already offset) state; room centred at (0, 0) as
    every hexapod heldout clip / branch was. override: dict(size, height, ground_uv) or None (= source rule)."""
    from render_hex_replay import settle, capture, cam_pose, ENV, SENSOR, TRACK, EGO_FOV
    from ego_camera import (attach_ego, build_texture_box, insect_forward, randomise_ground, room_for, scale_floor,
                            WALK_PITCH)
    seed = int(d["room_seed"])
    settle(sim)
    sim.loadScene(os.path.join(ENV, scene))
    settle(sim)
    cam, track = sim.getObject("/" + SENSOR), sim.getObject(TRACK)
    R = room_for(sim.getObjectPosition(track, sim.handle_world)[2])
    R["size"] = 8.0
    top_src = src_floor_top(sim, R["size"])
    if override:
        R.update(override)
    scale_floor(sim, R["size"])
    randomise_ground(sim, seed=seed, uv=R["ground_uv"])
    attach_ego(sim, cam, track, insect_forward(sim), (0, 0, 0), offset_frac=R["offset_frac"],
               pitch_comp=WALK_PITCH["hexapod"])
    if override:
        rs_walls(sim, R, seed, (float(centre[0]), float(centre[1])), top_src)
        R["floor_top"] = hold_floor(sim, top_src)
        R["far"] = rs_far(sim, cam, R["size"])
    else:
        build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed,
                          centre=(float(centre[0]), float(centre[1])))
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(EGO_FOV)))
    lh = [sim.getObject(str(n)) for n in d["state_link_names"]]
    lp = np.asarray(d["state_link_pose"], float)
    fr, cp = [], []
    for t in range(len(lp)):
        for h, p in zip(lh, lp[t]):
            sim.setObjectPose(h, sim.handle_world, [float(v) for v in p])
        if frames_on:
            fr.append(capture(sim, cam))
        cp.append(cam_pose(sim, cam))
    return np.asarray(fr, np.uint8), np.asarray(cp, np.float64), R


def b1_render(sim, d, centre_pos, override=None):
    """render_b1_replay.ego_setup (box, mount, scale_floor, ground; room centred on the source clip's frame-0 base) +
    pose_and_capture on every stored frame."""
    from render_b1_replay import pose_and_capture
    root, joints, cam, R = b1_setup(sim, int(d["room_seed"]), centre_pos, override)
    fr, cp = zip(*[pose_and_capture(sim, root, joints, cam, d["base_pos"][t], d["base_quat"][t], d["joint_pos"][t])
                   for t in range(len(d["base_pos"]))])
    return np.asarray(fr, np.uint8), np.asarray(cp, np.float64), R


def b1_setup(sim, seed, centre_pos, override=None):
    """b1_render's scene setup (also used live by sim/control/close_loop_b1_physics_froude.py --rr_room):
    returns (root, joints, cam, R); frames come from render_b1_replay.pose_and_capture."""
    from render_b1_replay import settle, mount_heading, JOINT_ALIASES_SDK, SENSOR, ROOT_ALIAS
    from ego_camera import (attach_ego, build_texture_box, randomise_ground, room_for, scale_floor, WALK_PITCH)
    settle(sim)
    sim.loadScene(B1_SCENE)
    settle(sim)
    jm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [jm[a] for a in JOINT_ALIASES_SDK]
    sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root = sm[ROOT_ALIAS]
    cam = sim.getObject("/" + SENSOR)
    R = room_for(sim.getObjectPosition(root, sim.handle_world)[2])
    top_src = src_floor_top(sim, R["size"])
    if override:
        R.update(override)
    if override:
        rs_walls(sim, R, seed, (float(centre_pos[0]), float(centre_pos[1])), top_src)
    else:
        build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=seed,
                          centre=(float(centre_pos[0]), float(centre_pos[1])))
    psi = mount_heading(sim, root)
    attach_ego(sim, cam, root, [float(np.cos(psi)), float(np.sin(psi)), 0.0], (0, 0, 0),
               offset_frac=R["offset_frac"], pitch_comp=WALK_PITCH["b1"])
    scale_floor(sim, R["size"])
    randomise_ground(sim, seed=seed, uv=R["ground_uv"])
    if override:
        R["floor_top"] = hold_floor(sim, top_src)
        R["far"] = rs_far(sim, cam, R["size"])
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(90.0)))
    return root, joints, cam, R


def load(p):
    with np.load(p, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def render_file(sim, body, src, override, offset=(0.0, 0.0)):
    """offset: clip start (default room centre) relative to the room centre, i.e. room centre = default - offset."""
    d = load(src)
    o = np.asarray(offset, float)
    if body == "b1":
        cp_src = str(d["cf_source_path"]) if "cf_source_path" in d else None
        centre = (load(os.path.join(ROOT, cp_src))["base_pos"][0] if cp_src else d["base_pos"][0])
        fr, cp, R = b1_render(sim, d, np.asarray(centre[:2], float) - o, override)
    else:
        fr, cp, R = hex_render(sim, d, HEX_SCENE[body], override, centre=-o)
    return d, fr, cp, R


# ------------------------------------------------------------------------------------------------ jobs
def src_dir(body, kind):
    return os.path.join(CW, f"{body}_{kind}_heldout")


def dst_dir(body, kind):
    return os.path.join(CW, f"rs_{body}_{kind}_heldout")


def branch_groups():
    """24 of the 72 groups per body: per heldout clip (= condition i) branch point j = i % 3 of its 3 points (points
    ordered by target phase, branch_points_current.json); c08 points = its c10 counterpart's, so c10 / c08 share keys."""
    P = json.load(open(os.path.join(CW, "branch_points_current.json")))
    out = {}
    for body in BODIES:
        g = []
        for p in sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz"))):
            v = P[os.path.relpath(p, ROOT)]
            i = int(v["cond_index"]); j = i % 3
            g.append(dict(source=os.path.basename(p), cond_index=i, point_index=j, t=int(v["t"][j]),
                          target_phase=float(v["target"][j]), ep=int(v["ep"])))
        out[body] = g
    return out


def jobs(kinds=("clips", "branches")):
    G = branch_groups()
    J = []
    for body in BODIES:
        if "clips" in kinds:
            for p in sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz"))):
                J.append((body, p, os.path.join(dst_dir(body, "clips"), os.path.basename(p))))
        if "branches" in kinds:
            pre = "b1" if body == "b1" else "hexapod"
            for g in G[body]:
                for ci in range(24):
                    n = f"{pre}_ep{g['ep'] * 10000 + g['t'] * 100 + ci}.npz"
                    J.append((body, os.path.join(src_dir(body, "branches"), n), os.path.join(dst_dir(body, "branches"), n)))
    return J




def run_pool(J, ports, fn, tag):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    q = list(J); lock = threading.Lock(); done = [0]; errs = []; t0 = time.time()

    def worker(port):
        sim = RemoteAPIClient("localhost", port=port).require("sim")
        while True:
            with lock:
                if not q:
                    return
                j = q.pop(0)
            try:
                msg = fn(sim, j)
            except Exception as e:      # noqa: BLE001
                msg = f"ERROR {e!r}"; errs.append((j, repr(e)))
            with lock:
                done[0] += 1
                if done[0] % 25 == 0 or done[0] <= 3 or "ERROR" in msg:
                    print(f"[{tag}] {done[0]}/{len(J)} {time.time() - t0:.0f}s {os.path.basename(str(j[1]))}: {msg}",
                          flush=True)
    th = [threading.Thread(target=worker, args=(p,)) for p in ports]
    [t.start() for t in th]; [t.join() for t in th]
    print(f"[{tag}] done {done[0]} jobs, {len(errs)} errors, {time.time() - t0:.0f}s", flush=True)
    return errs


# ------------------------------------------------------------------------------------------------ commands
def do_launch(a):
    os.makedirs(CHK, exist_ok=True)
    pids = json.load(open(PIDS)) if os.path.exists(PIDS) else {}
    for port in a.ports:
        assert port != 23000
        pr = subprocess.Popen(f"tail -f /dev/null | ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort={port} "
                              f"-GzmqRemoteApi.cntPort={port + 1}", shell=True, cwd=COPPELIA, start_new_session=True,
                              stdout=open(os.path.join(CHK, f"csim_{port}.log"), "a"), stderr=subprocess.STDOUT)
        pids[str(port)] = pr.pid           # session leader = process group id of everything it starts
    json.dump(pids, open(PIDS, "w"))
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    for port in a.ports:
        for _ in range(60):
            try:
                RemoteAPIClient("localhost", port=port).require("sim").getSimulationState()
                print(f"port {port} up (pgid {pids[str(port)]})"); break
            except Exception:              # noqa: BLE001
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
        print(f"stopped port {port} (pgid {pg})"); pids.pop(port)
    json.dump(pids, open(PIDS, "w"))


def do_probe(a):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from render_hex_replay import settle, ENV, TRACK
    from render_b1_replay import ROOT_ALIAS
    from ego_camera import room_for
    sim = RemoteAPIClient("localhost", port=a.port).require("sim")
    out = {}
    for body in BODIES:
        settle(sim)
        if body == "b1":
            sim.loadScene(B1_SCENE); settle(sim)
            sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
            z = sim.getObjectPosition(sm[ROOT_ALIAS], sim.handle_world)[2]
            R = room_for(z)
        else:
            sim.loadScene(os.path.join(ENV, HEX_SCENE[body])); settle(sim)
            z = sim.getObjectPosition(sim.getObject(TRACK), sim.handle_world)[2]
            R = room_for(z); R["size"] = 8.0
        out[body] = dict(mount_z=float(z), size=float(R["size"]), height=float(R["height"]),
                         ground_uv=float(R["ground_uv"]), k=float(R["scale"]))
    # camera excursion from the room centre (Chebyshev), all heldout clips + branches
    for body in BODIES:
        mx = 0.0; mz = 0.0; cen = {}
        for kind in ("clips", "branches"):
            for p in sorted(glob.glob(os.path.join(src_dir(body, kind), "*.npz"))):
                with np.load(p, allow_pickle=True) as f:
                    cp = f["cam_pose"][:, :2]
                    if body == "b1":
                        s = str(f["cf_source_path"]) if "cf_source_path" in f.files else os.path.relpath(p, ROOT)
                        if s not in cen:
                            cen[s] = np.load(os.path.join(ROOT, s))["base_pos"][0, :2]
                        c = cen[s]
                    else:
                        c = np.zeros(2)
                    mx = max(mx, float(np.abs(cp - c).max()))
                    mz = max(mz, float(f["cam_pose"][:, 2].max()))
        out[body]["max_cam_excursion"] = mx
        out[body]["max_cam_z"] = mz
    os.makedirs(CHK, exist_ok=True)
    json.dump(out, open(os.path.join(CHK, "probe.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    lo, hi = size_range()
    S = sizes()
    print(f"range {lo:.3f} .. {hi:.3f} m; per room seed: " + " ".join(f"{k}:{v:.2f}" for k, v in S.items()))
    json.dump(dict(range=[lo, hi], size_seed=SIZE_SEED, rule="stratified log-uniform, one size per room seed",
                   sizes={str(k): v for k, v in S.items()}, probe=out),
              open(os.path.join(CW, "rs_room_sizes_heldout.json"), "w"), indent=1)


def src_room(body):
    P = json.load(open(os.path.join(CHK, "probe.json")))
    return rs_room(P[body]["size"])


def do_gate_a(a):
    """Original room through this path vs stored frames: 2 clips + 2 branches per body (one branch with an own
    source-room prefix)."""
    J = []
    for body in BODIES:
        cl = sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz")))
        br = [j for j in jobs(("branches",)) if j[0] == body]
        J += [(body, cl[0]), (body, cl[13]), (body, br[5][1]), (body, br[300][1])]
    res = []

    def fn(sim, j):
        body, src = j
        d, fr, cp, R = render_file(sim, body, src, src_room(body) if a.rs_path else None)
        diff = np.abs(fr.astype(int) - d["frames"].astype(int))
        r = float(np.corrcoef(fr.ravel().astype(float), d["frames"].ravel().astype(float))[0, 1])
        m = dict(body=body, file=os.path.relpath(src, ROOT), mae=float(diff.mean()), max=int(diff.max()),
                 frac_px_diff=float((diff.max(-1) > 0).mean()), corr=r,
                 cam_pose_maxdiff=float(np.abs(cp - d["cam_pose"]).max()), room=R["size"])
        res.append(m)
        return f"mae {m['mae']:.4f} max {m['max']} corr {r:.6f} cam {m['cam_pose_maxdiff']:.1e}"
    run_pool(J, a.ports, fn, "gate_a")
    json.dump(res, open(os.path.join(CHK, "gate_a_rs_path.json" if a.rs_path else "gate_a.json"), "w"), indent=1)


def do_review(a):
    """Per body, 2 clips at the smallest / middle / largest size of the range (+ the size assigned to them), 4 frames
    each -> contact sheets; plus the row-profile corr vs the source frames (FOV / floor / wall sanity)."""
    from PIL import Image, ImageDraw
    lo, hi = size_range()
    tests = [("small", lo), ("mid", float(np.sqrt(lo * hi))), ("large", hi)]
    sys.path.insert(0, os.path.join(ROOT, "sim/scene"))
    from ego_camera import check_ego_view, ego_view_profile  # noqa: F401
    J = []
    for body in BODIES:
        cl = sorted(glob.glob(os.path.join(src_dir(body, "clips"), "*.npz")))
        for src in (cl[2], cl[17]):          # pick 2 conditions
            for name, S in tests:
                J.append((body, src, name, S))
    out = {}

    def fn(sim, j):
        body, src, name, S = j
        d, fr, cp, R = render_file(sim, body, src, rs_room(S))
        out[(body, src, name)] = (fr, d["frames"], S, float(np.abs(cp - d["cam_pose"]).max()))
        return f"{name} {S:.2f} m"
    run_pool(J, a.ports, fn, "review")
    rep = []
    for body in BODIES:
        rows = []
        srcs = sorted({k[1] for k in out if k[0] == body})
        for src in srcs:
            ref = out[(body, src, "small")][1]
            for name in ("orig", "small", "mid", "large"):
                fr = ref if name == "orig" else out[(body, src, name)][0]
                S = {"orig": None}.get(name, out[(body, src, name)][2] if name != "orig" else None)
                idx = [0, 20, 40, 65]
                tiles = [Image.fromarray(fr[t]).resize((192, 192)) for t in idx]
                row = Image.new("RGB", (192 * 4 + 150, 192), (0, 0, 0))
                for k, im in enumerate(tiles):
                    row.paste(im, (150 + 192 * k, 0))
                dr = ImageDraw.Draw(row)
                dr.text((4, 60), f"{os.path.basename(src)}", fill=(255, 255, 255))
                dr.text((4, 80), f"{name}" + (f" {S:.1f} m" if S else " (stored)"), fill=(255, 255, 0))
                rows.append(row)
                if name != "orig":
                    prof = float(np.corrcoef(ego_view_profile(fr[0]), ego_view_profile(ref[0]))[0, 1])
                    # floor fraction: rows below the image centre that are brighter / darker -- report luminance by band
                    lum = fr.mean(-1).mean(0)
                    rep.append(dict(body=body, file=os.path.basename(src), size=name, S=S, rowprofile_corr_vs_stored=prof,
                                    lum_top=float(lum[:32].mean()), lum_mid=float(lum[112:144].mean()),
                                    lum_bottom=float(lum[-32:].mean()), min_frame_std=float(fr.reshape(len(fr), -1).std(1).min()),
                                    cam_pose_maxdiff=out[(body, src, name)][3]))
        sheet = Image.new("RGB", (rows[0].width, 192 * len(rows)))
        for k, r in enumerate(rows):
            sheet.paste(r, (0, 192 * k))
        sheet.save(os.path.join(CHK, f"sheet_{body}.png"))
        # short video: small / mid / large side by side, first clip
        import imageio
        src = srcs[0]
        with imageio.get_writer(os.path.join(CHK, f"video_{body}.mp4"), fps=10) as w:
            for t in range(66):
                w.append_data(np.concatenate([out[(body, src, 'small')][1][t]] +
                                             [out[(body, src, n)][0][t] for n in ("small", "mid", "large")], 1))
    json.dump(rep, open(os.path.join(CHK, "review.json"), "w"), indent=1)
    for r in rep:
        print(r)


def write_rs(body, src, dst, d, fr, cp, S, R):
    if not np.allclose(cp, d["cam_pose"], atol=1e-9, rtol=0):
        raise RuntimeError(f"cam_pose changed: {np.abs(cp - d['cam_pose']).max():.2e}")
    if fr.shape != d["frames"].shape:
        raise RuntimeError(f"frames shape {fr.shape} != {d['frames'].shape}")
    out = {k: v for k, v in d.items() if k != "frames"}
    out.update(frames=fr, room_size=np.float64(S),
               rs_room_size_original=np.float64(d["room_size"]) if "room_size" in d else np.float64(np.nan),
               rs_room_height=np.float64(R["height"]), rs_ground_uv=np.float64(R["ground_uv"]),
               rs_size_seed=np.int64(SIZE_SEED), rs_source=np.array(os.path.relpath(src, ROOT)),
               rs_render=np.array(RS_RENDER))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f".tmp{os.getpid()}_{threading.get_ident()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)


def do_render(a):
    S = sizes()
    J = [j for j in jobs() if a.overwrite or not os.path.exists(j[2])]
    if a.body:
        J = [j for j in J if j[0] in a.body]
    if a.limit:
        J = J[:a.limit]
    print(f"{len(J)} files to render")

    def fn(sim, j):
        body, src, dst = j
        d0 = load(src)
        sz = S[int(d0["room_seed"])]
        d, fr, cp, R = render_file(sim, body, src, rs_room(sz))
        write_rs(body, src, dst, d, fr, cp, sz, R)
        return f"S {sz:.2f}"
    json.dump(branch_groups(), open(GROUPS_JSON, "w"), indent=1)
    run_pool(J, a.ports, fn, "render")


def do_verify(a):
    S = sizes()
    J = jobs()
    bad, n = [], {}
    NEW = ("room_size", "rs_room_size_original", "rs_room_height", "rs_ground_uv", "rs_size_seed", "rs_source", "rs_render")
    for body, src, dst in J:
        key = (body, "branches" if "branches" in dst else "clips")
        n[key] = n.get(key, 0) + 1
        if not os.path.exists(dst):
            bad.append((dst, "missing")); continue
        s, r = load(src), load(dst)
        ks = set(s) - {"frames", "room_size"}
        if set(r) - set(NEW) - {"frames"} != ks:
            bad.append((dst, f"keys {set(r) ^ set(s)}"))
        for k in ks:
            if s[k].dtype != r[k].dtype or s[k].shape != r[k].shape or s[k].tobytes() != r[k].tobytes():
                bad.append((dst, f"field {k} differs"))
        if r["frames"].shape != s["frames"].shape or r["frames"].dtype != np.uint8:
            bad.append((dst, "frames shape"))
        if float(r["room_size"]) != S[int(s["room_seed"])]:
            bad.append((dst, "room_size"))
    for body in BODIES:
        for kind in ("clips", "branches"):
            fs = glob.glob(os.path.join(dst_dir(body, kind), "*.npz"))
            tmp = glob.glob(os.path.join(dst_dir(body, kind), "*.tmp*"))
            print(f"rs_{body}_{kind}_heldout: {len(fs)} files (expected {n[(body, kind)]}), tmp leftovers {len(tmp)}")
    print(f"{len(bad)} problems"); [print(b) for b in bad[:30]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("launch", "stop", "probe", "gate_a", "review", "render", "verify"))
    ap.add_argument("--ports", type=int, nargs="+", default=[25300])
    ap.add_argument("--port", type=int, default=25300)
    ap.add_argument("--body", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="render: replace existing rs files (atomic)")
    ap.add_argument("--rs_path", action="store_true", help="gate_a through the rs path (floor hold, closed walls, far "
                                                            "clip) at the source room size")
    a = ap.parse_args()
    dict(launch=do_launch, stop=do_stop, probe=do_probe, gate_a=do_gate_a, review=do_review, render=do_render,
         verify=do_verify)[a.cmd](a)


if __name__ == "__main__":
    main()
