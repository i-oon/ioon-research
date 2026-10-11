"""Data for the wall PREDICTION CHECK (2026-10-09, protocol locked with the user before collection).

Question it serves: from held-out start states near a wall, does the clearance read from the FTM's imagined embeddings
rank the 24 candidate commands like their TRUE future clearance? (No control here.)

Protocol:
- start states: the feasibility pilot's scenarios (`b1_wall_feasibility.py`: wall / corner x front distance 1 / 1.5 / 2 / 3 m
  x heading 0 / 30 / 60 deg x goal forward level 1 / 3) x 2 replicas (different warm-up length and position jitter = a
  different start state / "episode"). Replica 0 -> FIT set (readouts are fitted on it), replica 1 -> TEST set.
- every candidate is HELD CONSTANT for 2 s (40 frames) from the start state, in MuJoCo (exact restore) -> a
  constant-command prediction test, not the closed loop (which re-chooses every 0.1 s).
- frames at 0.05 s (frame j = policy step round(2.5 j)); FTM stride 5 frames = 0.25 s, so imagined steps 1 / 2 / 4 / 8 =
  0.25 / 0.5 / 1 / 2 s (8 steps is beyond every trained chain: report separately).
- rendering: the ORIGINAL B1 room (17.65 m, walls fixed at +-8.8 m around the world origin, NOT re-centred), one room look
  per start state: fit looks 3000 + i, test looks 3100 + i (seeds used by no other data); all 24 futures of a state share
  its look. No start state, episode or look is in both sets.
- labels per frame: footprint signed clearance (`clearance`), front distance (`front_dist`), base pose; Froude via the loader.

    PY=.venv/bin/python3; S=scripts/diagnostics/wall_pilot/b1_wall_predcheck_data.py
    $PY $S physics --workers 5          # -> data/wall_predcheck/b1/_physics/*.npz (no frames)
    $PY $S render --ports P1 P2 ...     # -> data/wall_predcheck/b1/{fit,test}/*.npz (frames), own CoppeliaSim instances
    $PY $S sheet                        # -> results/check/wall_predcheck/sheet.png
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "scripts/diagnostics/wall_pilot"))
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
sys.path.insert(0, ROOT)
import b1_wall_feasibility as F  # noqa: E402

OUT = os.path.join(ROOT, "data/wall_predcheck/b1")
CHK = os.path.join(ROOT, "results/check/wall_predcheck")
NF = 40                                        # future frames (2 s)
FIELDS = ("base_pos", "base_quat", "joint_pos", "joint_vel", "action", "command", "foot_contact")


def states():
    S = []
    for rep, split in ((0, "fit"), (1, "test")):
        i = 0
        for k in ("wall", "corner"):
            for d in F.DISTS:
                for th in F.THETAS:
                    for sp in range(len(F.SPEEDS)):
                        S.append(dict(kind=k, d=d, th=th, sp=sp, split=split, rep=rep, idx=i,
                                      look=(3000 if split == "fit" else 3100) + i))
                        i += 1
    return S


def physics_job(st):
    import zlib
    from build_b1_cf_branches import run_branch
    cmds, fam = F.commands()
    rng = np.random.default_rng(zlib.crc32(repr(("predcheck", st["kind"], st["d"], st["th"], st["sp"], st["rep"])).encode()))
    jit = rng.uniform(-0.3, 0.3, 2)
    extra = int(rng.integers(0, 20)) + 20 * st["rep"]
    x = F.H - st["d"] - F.FOOT_L / 2 + jit[0]
    y = (F.H - 1.5 - F.FOOT_W / 2 + jit[1]) if st["kind"] == "corner" else jit[1]
    r, snap = F.start_state(None, cmds, F.FWD[st["sp"]], x, y, np.deg2rad(st["th"]), extra)
    r.restore(snap)
    s0 = {f: v for f, v in zip(FIELDS, (r.d.qpos[0:3].copy(), r.d.qpos[3:7].copy(), r.d.qpos[7:19].copy(),
                                         r.d.qvel[6:18].copy(), r.last.copy(), np.zeros(3), np.zeros(4)))}
    offs = [int(v) for v in np.round(2.5 * np.arange(1, NF + 1))]
    os.makedirs(os.path.join(OUT, "_physics"), exist_ok=True)
    n = 0
    for ci, c in enumerate(cmds):
        dst = os.path.join(OUT, "_physics", f"{st['split']}_s{st['idx']:02d}_c{ci:02d}.npz")
        if os.path.exists(dst):
            continue
        br, fell = run_branch(r, snap, c, offs)
        data = {f: np.concatenate([np.asarray(s0[f], np.float64)[None] if f not in ("action", "command", "foot_contact")
                                   else np.asarray(br[f][:1]) * 0, np.asarray(br[f])]) for f in FIELDS}
        pos, quat = data["base_pos"], data["base_quat"]
        yaw = np.array([F.yaw_of(q) for q in quat])
        data["clearance"] = np.array([F.clearance(p[:2], y_, None) for p, y_ in zip(pos, yaw)])
        data["front_dist"] = np.array([F.front_dist(p[:2], y_) for p, y_ in zip(pos, yaw)])
        from wm.data.com import b1_com
        data["com_pos"] = np.asarray(b1_com(data["base_pos"], data["base_quat"], data["joint_pos"],
                                            os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat.xml")), np.float64)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **data, cmd_index=np.int64(ci), cmd=np.asarray(c), fell=np.bool_(fell),
                            start=json.dumps(st), look=np.int64(st["look"]), split=np.array(st["split"]),
                            room_seed=np.int64(st["look"]), dt=np.float32(0.05), embodiment=np.array("b1"),
                            condition=np.array(f"cmd{ci:02d}"), protocol=np.array(
                                "constant command held 2 s from the start state; frame 0 = start state"))
        os.replace(tmp, dst)
        n += 1
    return f"{st['split']} s{st['idx']:02d} {st['kind']} d={st['d']} th={st['th']}: {n} futures"


def do_physics(a):
    S = states()
    with Pool(a.workers) as p:
        for m in p.imap_unordered(physics_job, S):
            print(m, flush=True)


def do_render(a):
    import render_shift_heldout as RS
    J = sorted(glob.glob(os.path.join(OUT, "_physics", "*.npz")))
    J = [p for p in J if not os.path.exists(os.path.join(OUT, str(np.load(p)["split"]), os.path.basename(p)))]
    print(f"{len(J)} futures to render")

    def fn(sim, p):
        d = dict(np.load(p, allow_pickle=True))
        dd = dict(base_pos=d["base_pos"].astype(np.float32), base_quat=d["base_quat"].astype(np.float32),
                  joint_pos=d["joint_pos"].astype(np.float32), room_seed=np.int64(d["look"]))
        fr, cp, R = RS.b1_render(sim, dd, np.zeros(2))          # room fixed at the world origin (not re-centred)
        dst = os.path.join(OUT, str(d["split"]), os.path.basename(p))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **d, frames=fr, cam_pose=cp, room_size=np.float64(R["size"]))
        os.replace(tmp, dst)
        return f"clear {d['clearance'].min():+.2f}"
    RS.SCENE_HOOK = None
    RS.run_pool(J, a.ports, fn, "render")


def do_sheet(a):
    from PIL import Image, ImageDraw
    os.makedirs(CHK, exist_ok=True)
    rows = []
    for s in ("test_s00", "test_s07", "test_s30"):
        tiles = []
        for ci in (0, 11, 15, 18):
            f = os.path.join(OUT, "test", f"{s}_c{ci:02d}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            for j in (0, 20, 40):
                im = Image.fromarray(d["frames"][j]).resize((128, 128))
                ImageDraw.Draw(im).text((3, 3), f"c{ci} {j * 0.05:.1f}s {d['clearance'][j]:+.2f}m", fill=(255, 255, 0))
                tiles.append(np.asarray(im))
        if tiles:
            rows.append(np.hstack(tiles))
    Image.fromarray(np.vstack(rows)).save(os.path.join(CHK, "sheet.png"))
    print("->", os.path.relpath(os.path.join(CHK, "sheet.png"), ROOT),
          "(rows: 3 test start states; per command 0 fwd / 11 left / 15 right / 18 side, frames 0 / 1 / 2 s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("physics", "render", "sheet"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--ports", type=int, nargs="+", default=[25740])
    a = ap.parse_args()
    {"physics": do_physics, "render": do_render, "sheet": do_sheet}[a.cmd](a)
