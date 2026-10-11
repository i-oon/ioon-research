"""Hexapod (c10) TURN-ON-THE-SPOT pilot (2026-10-11): can the CPG spin in place at the same dimensionless yaw rate
(Froude) as B1's added spin commands (B1: wz 0.3 -> yaw +0.075, wz 0.5 -> +0.125, forward ~0)?

Recipe = the turn recipe of `collect_switch_hex.COND` (pace 6/8.8, lead 0.25, WALK amplitudes) with the mirrored
forward sweep a0 scaled (0 = no forward drive) and `spin` swept; constant for 66 frames, run with `collect_ik.py
--plan` exactly as the switching clips; Froude (forward, lateral, yaw) from the training label code
(`wm.data.embodiment.load`), averaged over frames 20-65.

    .venv/bin/python3 scripts/diagnostics/wall_pilot/hex_spin_pilot.py --port P [--a0 0 0.05] [--spin 0.3 0.5 0.8]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
from collect_switch_hex import COND, KEYS, COMMON  # noqa: E402

CENTRE = os.path.join(ROOT, "data/counterfactual_walks/c10_walks/_work/centre_pose.npz")
OUT = os.path.join(ROOT, "results/check/hex_spin_pilot")
EP = 66


def run(port, a0, spin, pace, ep):
    from wm.data.embodiment import REGISTRY, load
    rec = dict(COND["turn_s0.56"][1]); rec.update(a0=a0, spin=spin, pace=pace)
    work = os.path.join(OUT, f"a0{a0:g}_spin{spin:+g}_pace{pace * 8.8:g}")
    os.makedirs(work, exist_ok=True)
    pf = os.path.join(work, "plan.json")
    json.dump({k: [float(rec[k])] * EP for k in KEYS}, open(pf, "w"))
    for f in glob.glob(os.path.join(work, "c10f10t10_*.npz")):
        os.remove(f)
    cmd = [sys.executable, os.path.join(ROOT, "sim/collect/collect_ik.py"), "--port", str(port),
           "--morphs", "c10f10t10=medauroidea_c10f10t10.ttt", "--episodes", str(ep), "--ego_seed", "0",
           "--plan", pf, "--out", work, "--centre_from", CENTRE, "--record_state"] + COMMON
    with open(os.path.join(work, "collect.log"), "w") as log:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    src = glob.glob(os.path.join(work, "c10f10t10_*.npz"))[0]
    d = dict(np.load(src, allow_pickle=True)); d.update(embodiment=np.array("hexapod"), condition=np.array("spin_pilot"))
    dst = os.path.join(work, "clip.npz"); np.savez_compressed(dst, **d)
    c = load(dst, REGISTRY["hexapod"], lazy_frames=True)
    bm = np.asarray(c["body_motion"])[20:]
    up = float(np.min(np.asarray(d["body_quat"])[:, 3] ** 2)) if "body_quat" in d else np.nan
    return bm.mean(0), bm.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--a0", type=float, nargs="+", default=[0.0])
    ap.add_argument("--spin", type=float, nargs="+", default=[0.3, 0.5, 0.8, -0.5])
    ap.add_argument("--pace", type=float, nargs="+", default=[6.0])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = ["recipe (turn_s0.56 recipe with a0 / spin / pace changed) | Froude forward / lateral / yaw (mean over frames "
             "20-65) | std"]
    ep = 90000
    for a0 in a.a0:
        for pc in a.pace:
            for sp in a.spin:
                m, s = run(a.port, a0, sp, pc / 8.8, ep); ep += 1
                lines.append(f"a0 {a0:g} spin {sp:+g} cycles {pc:g} | {m[0]:+.3f} {m[1]:+.3f} {m[2]:+.3f} | "
                             f"{s[0]:.3f} {s[1]:.3f} {s[2]:.3f}")
                print(lines[-1], flush=True)
    open(os.path.join(OUT, "summary.txt"), "a").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
