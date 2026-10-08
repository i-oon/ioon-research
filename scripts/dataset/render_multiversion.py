"""Multi-version rendering of the rr TRAINING data: extra room versions v2, v3 of every file in
data/counterfactual_walks/rr_{c10,b1}_{clips,branches}_train -> rrv{2,3}_{c10,b1}_{clips,branches}_train (same names).

Why (FINDINGS F320): in rr every training clip lives in its own random room, so z mixes room and motion; rendering each
clip (and all its branches) in several rooms forces z to ignore the room.

Same procedure as render_random_room.py (rr), physics unchanged (kinematic replay of the stored state through the
render_shift_heldout rs path: visible floor held at the source level, wall skirts, far clip >= 2S, camera mount + FOV 90
as the source). Per version, one NEW room seed per source room seed (v2: 1000 + s, v3: 1100 + s, s = 0..47), so a clip and
all its branches share the version's room, as in rr. The new seed sets textures / colours / ground; room_seed itself is
kept (it is the pairing key) and the texture seed is stored as rr_room_seed.
  size S   log-uniform rr range (8.0 .. 26.47 m), stratified over the 48 seeds (each of 48 equal log-bins once),
           seed SIZE_SEED[v];
  offset o uniform (seed [OFFSET_SEED[v], source room seed]) over the box where every camera position of the c10 and B1
           clip + all branches of that source seed stays >= rr margin (2 x body's max camera height) inside each wall;
           midpoint + rr_offset_shrunk if an axis has none (rr rule).
Fields: rr source fields bit-for-bit except frames and the room fields (room_size, rr_room_size, rr_room_offset,
rr_size_seed, rr_offset_seed, rr_offset_shrunk, rr_room_height, rr_ground_uv, rr_source, rr_render) + rr_room_seed,
rr_version. Writes: tmp + fsync + rename + dir fsync; resumable (existing outputs skipped).

    S=scripts/dataset/render_multiversion.py
    $PY $S place                          # -> rr_v2_rooms.json, rr_v3_rooms.json
    $PY $S launch --ports P               # one instance per call;  $PY $S stop  (by process group)
    $PY $S gate_a --ports ...             # rr room through this path == stored rr frames
    $PY $S render --ports ... --stage v2_c10 v2_b1 v3_c10 v3_b1
    $PY $S sheet --ports ...              # results/check/multiversion/sheet_{c10,b1}.png
    $PY $S verify
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
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
import render_shift_heldout as RS  # noqa: E402
import render_random_room as RR  # noqa: E402

CW = RS.CW
CHK = os.path.join(ROOT, "results/check/multiversion")
PIDS = os.path.join(CHK, "pids.json")
VERSIONS = {"v2": dict(seed_base=1000, size_seed=20261102, offset_seed=20261112),
            "v3": dict(seed_base=1100, size_seed=20261103, offset_seed=20261113)}
BODIES = ("c10", "b1")
KINDS = ("clips", "branches")
EXISTING_SEEDS = set(range(48)) | set(range(100, 124)) | set(range(200, 224)) | set(range(300, 696)) | \
    set(range(700, 880)) | set(range(900, 924))
ROOM_FIELDS = {"room_size", "rr_room_size", "rr_room_offset", "rr_size_seed", "rr_offset_seed", "rr_offset_shrunk",
               "rr_room_height", "rr_ground_uv", "rr_source", "rr_render"}
NEW_FIELDS = {"rr_room_seed", "rr_version"}


def rooms_path(v):
    return os.path.join(CW, f"rr_{v}_rooms.json")


def rr_dir(body, kind):
    return os.path.join(CW, f"rr_{body}_{kind}_train")


def out_dir(v, body, kind):
    return os.path.join(CW, f"rr{v}_{body}_{kind}_train")


def sizes(v):
    lo, hi = RS.size_range()
    n = 48
    rng = np.random.default_rng(VERSIONS[v]["size_seed"])
    perm, u = rng.permutation(n), rng.random(n)
    s = np.exp(np.log(lo) + (perm + u) / n * (np.log(hi) - np.log(lo)))
    return {sd: float(s[sd]) for sd in range(n)}


def do_place(a):
    RM = RR.rooms()
    margin = RM["margin"]
    pts = {}
    for body in BODIES:
        cen = {}
        for kind in KINDS:
            for p in sorted(glob.glob(os.path.join(rr_dir(body, kind), "*.npz"))):
                with np.load(p, allow_pickle=True) as f:
                    seed = int(f["room_seed"]); cp = np.asarray(f["cam_pose"], float)
                    if body == "b1":
                        src = str(f["cf_source_path"]) if "cf_source_path" in f.files else str(f["rr_source"])
                        if src not in cen:
                            with np.load(os.path.join(ROOT, src)) as g:
                                cen[src] = np.asarray(g["base_pos"][0, :2], float)
                        c = cen[src]
                    else:
                        c = np.zeros(2)
                xy = cp[:, :2] - c
                lo_, hi_ = pts.get((seed, body), (np.full(2, np.inf), np.full(2, -np.inf)))
                pts[(seed, body)] = (np.minimum(lo_, xy.min(0)), np.maximum(hi_, xy.max(0)))
        print(f"{body}: scanned", flush=True)
    for v, cfg in VERSIONS.items():
        S = sizes(v)
        rooms = {}
        for seed, sz in S.items():
            new_seed = cfg["seed_base"] + seed
            assert new_seed not in EXISTING_SEEDS
            h = sz / 2
            lo, hi = np.full(2, -np.inf), np.full(2, np.inf)
            bodies = [b for b in BODIES if (seed, b) in pts]
            for b in bodies:
                pmin, pmax = pts[(seed, b)]
                lo = np.maximum(lo, -(h - margin[b]) - pmin)
                hi = np.minimum(hi, (h - margin[b]) - pmax)
            u = np.random.default_rng([cfg["offset_seed"], seed]).random(2)
            shrunk = [bool(lo[i] > hi[i]) for i in range(2)]
            o = np.where(lo <= hi, lo + u * (hi - lo), (lo + hi) / 2)
            clear = min(float(min(h - np.abs(pts[(seed, b)][0] + o).max(), h - np.abs(pts[(seed, b)][1] + o).max())
                              - margin[b]) for b in bodies)
            rooms[str(seed)] = dict(room_seed=new_seed, size=sz, offset=[float(o[0]), float(o[1])], valid_lo=lo.tolist(),
                                    valid_hi=hi.tolist(), shrunk=shrunk, bodies=bodies,
                                    min_clearance_beyond_margin=clear)
        js = dict(version=v, size_range=list(RS.size_range()), size_seed=cfg["size_seed"],
                  offset_seed=cfg["offset_seed"], margin=margin, key="source rr room seed (0..47)",
                  rule="new room seed = seed_base + source seed; size stratified log-uniform over 48 seeds; offset "
                       "uniform over the valid box (c10 + B1, clip + branches, camera >= margin inside every wall), "
                       "midpoint if an axis has none", rooms=rooms)
        durable_json(js, rooms_path(v))
        cl = [r["min_clearance_beyond_margin"] for r in rooms.values()]
        print(f"{v}: seeds {cfg['seed_base']}..{cfg['seed_base'] + 47}; size {min(S.values()):.2f}..{max(S.values()):.2f}"
              f" (median {np.median(list(S.values())):.2f}); {sum(any(r['shrunk']) for r in rooms.values())} shrunk; "
              f"clearance min {min(cl):.3f}; |offset| max {max(np.abs(r['offset']).max() for r in rooms.values()):.2f}")


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path); fsync_dir(os.path.dirname(path))


def durable_npz(out, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst[:-4] + f".tmp{os.getpid()}_{threading.get_ident()}.npz"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, dst); fsync_dir(os.path.dirname(dst))


def render(sim, body, src, texture_seed, size, offset):
    """RS.render_file with the room texture seed replaced; returns (d, frames, cam_pose, R)."""
    d = RS.load(src)
    d2 = dict(d, room_seed=np.int64(texture_seed))
    o = np.asarray(offset, float)
    ov = RS.rs_room(size)
    if body == "b1":
        cp_src = str(d["cf_source_path"]) if "cf_source_path" in d else None
        centre = (RS.load(os.path.join(ROOT, cp_src))["base_pos"][0] if cp_src else d["base_pos"][0])
        fr, cp, R = RS.b1_render(sim, d2, np.asarray(centre[:2], float) - o, ov)
    else:
        fr, cp, R = RS.hex_render(sim, d2, RS.HEX_SCENE[body], ov, centre=-o)
    return d, fr, cp, R


def jobs(v, body):
    J = []
    for kind in KINDS:
        for p in sorted(glob.glob(os.path.join(rr_dir(body, kind), "*.npz"))):
            J.append((v, body, p, os.path.join(out_dir(v, body, kind), os.path.basename(p))))
    return J


def do_render(a):
    RMs = {v: json.load(open(rooms_path(v))) for v in VERSIONS}
    for st in a.stage:
        v, body = st.split("_")
        for k in KINDS:                         # stale tmp files from an interrupted run
            for t in glob.glob(os.path.join(out_dir(v, body, k), "*.tmp*")):
                os.remove(t)
        J = [j for j in jobs(v, body) if not os.path.exists(j[3])]
        print(f"{st}: {len(J)} files to render", flush=True)
        free = os.statvfs(CW); free = free.f_bavail * free.f_frsize / 1e9
        if free < a.min_free_gb + 5:
            print(f"STOP: free {free:.1f} GB"); return
        room_of = RMs[v]["rooms"]

        def fn(sim, j):
            v_, body_, src, dst = j
            d = RS.load(src)
            room = room_of[str(int(d["room_seed"]))]
            d, fr, cp, R = render(sim, body_, src, room["room_seed"], room["size"], room["offset"])
            if not np.array_equal(cp, d["cam_pose"]):
                raise RuntimeError(f"cam_pose changed {np.abs(cp - d['cam_pose']).max():.2e}")
            if fr.shape != d["frames"].shape:
                raise RuntimeError("frames shape")
            out = {k: x for k, x in d.items() if k != "frames"}
            out.update(frames=fr, room_size=np.float64(room["size"]), rr_room_size=np.float64(room["size"]),
                       rr_room_offset=np.asarray(room["offset"], np.float64),
                       rr_size_seed=np.int64(RMs[v_]["size_seed"]), rr_offset_seed=np.int64(RMs[v_]["offset_seed"]),
                       rr_offset_shrunk=np.asarray(room["shrunk"], bool), rr_room_height=np.float64(R["height"]),
                       rr_ground_uv=np.float64(R["ground_uv"]), rr_source=np.array(os.path.relpath(src, ROOT)),
                       rr_render=np.array(RR.RR_RENDER + f"; multi-version {v_}: room texture seed rr_room_seed "
                                                         f"(room_seed kept as the source pairing key)"),
                       rr_room_seed=np.int64(room["room_seed"]), rr_version=np.array(v_))
            durable_npz(out, dst)
            return f"S {room['size']:.2f} seed {room['room_seed']}"
        errs = RS.run_pool(J, a.ports, fn, st)
        if errs:
            print(f"{st}: {len(errs)} errors, stopping"); return


def do_gate_a(a):
    """rr room (source texture seed, rr size + offset) through this path == stored rr frames."""
    RM = RR.rooms()["rooms"]
    J = []
    for body in BODIES:
        for kind in KINDS:
            fs = sorted(glob.glob(os.path.join(rr_dir(body, kind), "*.npz")))
            J += [(body, fs[0]), (body, fs[len(fs) // 2]), (body, fs[-1])]
    res = []

    def fn(sim, j):
        body, src = j
        s0 = int(np.load(src)["room_seed"]); room = RM[str(s0)]
        d, fr, cp, R = render(sim, body, src, s0, room["size"], room["offset"])
        diff = np.abs(fr.astype(int) - d["frames"].astype(int))
        r = dict(file=os.path.relpath(src, ROOT), max=int(diff.max()), mae=float(diff.mean()),
                 cam=float(np.abs(cp - d["cam_pose"]).max()), exact=bool(np.array_equal(fr, d["frames"])))
        res.append(r)
        return str(r)
    RS.run_pool(J, a.ports, fn, "gate_a")
    os.makedirs(CHK, exist_ok=True)
    durable_json(res, os.path.join(CHK, "gate_a.json"))
    print(f"exact {sum(r['exact'] for r in res)}/{len(res)}")


def do_sheet(a):
    from PIL import Image, ImageDraw
    os.makedirs(CHK, exist_ok=True)
    for body in BODIES:
        fs = sorted(glob.glob(os.path.join(rr_dir(body, "clips"), "*.npz")))
        p = fs[a.clip_index]
        rows = []
        for lab, q in [("rr", p)] + [(v, os.path.join(out_dir(v, body, "clips"), os.path.basename(p))) for v in VERSIONS]:
            with np.load(q) as f:
                F = f["frames"]; S = float(f["rr_room_size"]); o = f["rr_room_offset"]
                sd = int(f["rr_room_seed"]) if "rr_room_seed" in f.files else int(f["room_seed"])
            row = Image.new("RGB", (192 * 4 + 200, 192))
            for k, t in enumerate([0, len(F) // 3, 2 * len(F) // 3, len(F) - 1]):
                row.paste(Image.fromarray(F[t]).resize((192, 192)), (200 + 192 * k, 0))
            dr = ImageDraw.Draw(row)
            dr.text((4, 50), f"{body} {os.path.basename(p)}", fill=(255, 255, 255))
            dr.text((4, 75), f"{lab}: room seed {sd}", fill=(255, 255, 0))
            dr.text((4, 100), f"S {S:.1f} m off {o[0]:+.1f},{o[1]:+.1f}", fill=(255, 255, 0))
            rows.append(row)
        sheet = Image.new("RGB", (rows[0].width, 192 * len(rows)))
        for k, r in enumerate(rows):
            sheet.paste(r, (0, 192 * k))
        out = os.path.join(CHK, f"sheet_{body}.png")
        sheet.save(out); print(out)


def do_verify(a):
    bad = []
    for v in VERSIONS:
        RM = json.load(open(rooms_path(v)))["rooms"]
        for body in BODIES:
            n = {}
            for _, _, src, dst in jobs(v, body):
                key = os.path.basename(os.path.dirname(dst)); n[key] = n.get(key, 0) + 1
                if not os.path.exists(dst):
                    bad.append((dst, "missing")); continue
                s, r = RS.load(src), RS.load(dst)
                ks = set(s) - ROOM_FIELDS - {"frames"}
                if set(r) - NEW_FIELDS - ROOM_FIELDS - {"frames"} != ks or not (ROOM_FIELDS <= set(r)):
                    bad.append((dst, "keys"))
                for k in ks:
                    if s[k].dtype != r[k].dtype or s[k].shape != r[k].shape or s[k].tobytes() != r[k].tobytes():
                        bad.append((dst, f"field {k}"))
                room = RM[str(int(s["room_seed"]))]
                if (float(r["room_size"]) != room["size"] or list(r["rr_room_offset"]) != room["offset"]
                        or int(r["rr_room_seed"]) != room["room_seed"] or str(r["rr_version"]) != v):
                    bad.append((dst, "room"))
                if r["frames"].shape != s["frames"].shape or r["frames"].dtype != s["frames"].dtype:
                    bad.append((dst, "frames"))
            for key, c in n.items():
                d = os.path.join(CW, key)
                print(f"{key}: {len(glob.glob(os.path.join(d, '*.npz')))} files (expected {c}), "
                      f"tmp {len(glob.glob(os.path.join(d, '*.tmp*')))}", flush=True)
    print(f"{len(bad)} problems"); [print(b) for b in bad[:30]]


def do_launch(a):
    os.makedirs(CHK, exist_ok=True)
    pids = json.load(open(PIDS)) if os.path.exists(PIDS) else {}
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    for port in a.ports:
        assert port != 23000 and str(port) not in pids
        pr = subprocess.Popen(f"tail -f /dev/null | nice -n 10 ./coppeliaSim.sh -h -GzmqRemoteApi.rpcPort={port} "
                              f"-GzmqRemoteApi.cntPort={port + 1}", shell=True, cwd=RS.COPPELIA,
                              start_new_session=True, stdout=open(os.path.join(CHK, f"csim_{port}.log"), "a"),
                              stderr=subprocess.STDOUT)
        pids[str(port)] = pr.pid
        durable_json(pids, PIDS)
        for _ in range(60):
            try:
                RemoteAPIClient("localhost", port=port).require("sim").getSimulationState()
                print(f"port {port} up (pgid {pr.pid})"); break
            except Exception:  # noqa: BLE001
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
    durable_json(pids, PIDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("place", "launch", "stop", "gate_a", "render", "sheet", "verify"))
    ap.add_argument("--ports", type=int, nargs="+", default=[25500])
    ap.add_argument("--stage", nargs="+", default=["v2_c10", "v2_b1", "v3_c10", "v3_b1"])
    ap.add_argument("--min_free_gb", type=float, default=60.0)
    ap.add_argument("--clip_index", type=int, default=5)
    a = ap.parse_args()
    dict(place=do_place, launch=do_launch, stop=do_stop, gate_a=do_gate_a, render=do_render, sheet=do_sheet,
         verify=do_verify)[a.cmd](a)


if __name__ == "__main__":
    main()
