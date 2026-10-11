"""Motor babble for the B1, with behaviour switches INSIDE each clip -- the B1 twin of
`collect_babble_hex.py`.

The B1's walking policy (MuJoCo, `rollout_b1_mujoco.py`) follows velocity commands, so a babble clip
is a random sequence of commands: forward or backward walking with a random turn rate, or sideways
walking. Ranges are beh24's B1 recipe on the `gait3` policy (`recollect_b1_more.py`): |vx| 0.30-0.50,
|wz| up to 0.66, vy +0.12..+0.40 / -0.03..-0.33 (the policy's calibrated strafe range, which is
asymmetric). Segments last 15-25 rendered frames (20 Hz; 2.5 policy steps per frame), ramped over
10 policy steps. Each clip is rendered like beh24 (`render_b1_replay.py --ego --fps 20`, own room seed)
and stores the per-step commands as `command`.

    .venv/bin/python3 scripts/dataset/collect_babble_b1.py --pilot 3 --out data/egocentric/babble_b1_pilot
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
from recollect_b1_more import _face_forward  # noqa: E402

FRAMES, FPS = 66, 20.0
PER_FRAME = 50.0 / FPS                       # policy steps per rendered frame
STEPS = int(FRAMES * PER_FRAME)


def segment(rng):
    mode = rng.choice(["fwd", "bwd", "side"], p=[0.4, 0.25, 0.35])
    if mode == "side":
        vy = rng.uniform(0.12, 0.40) if rng.random() < 0.5 else -rng.uniform(0.034, 0.33)
        return mode, (0.0, vy, 0.0)
    vx = rng.uniform(0.30, 0.50) * (1 if mode == "fwd" else -1)
    wz = rng.uniform(-0.66, 0.66) if rng.random() < 0.7 else 0.0
    return mode, (vx, 0.0, wz)


def make_plan(rng, ramp=10):
    segs, f = [], 0
    while f < FRAMES:
        n = int(min(rng.integers(15, 26), FRAMES - f))
        if 0 < FRAMES - (f + n) < 8:        # a tail shorter than 8 frames would be mostly ramp: absorb it
            n = FRAMES - f
        segs.append((f, n) + segment(rng))
        f += n
    plan = np.zeros((STEPS, 3))
    for f0, n, _, cmd in segs:
        plan[int(f0 * PER_FRAME):int((f0 + n) * PER_FRAME)] = cmd
    for f0, _, _, _ in segs[1:]:
        c = int(f0 * PER_FRAME)
        lo, hi = max(1, c - ramp // 2), min(STEPS - 1, c + ramp // 2)
        plan[lo:hi] = np.linspace(plan[lo - 1], plan[hi], hi - lo + 2)[1:-1]
    return plan, [(f0, n, m) for f0, n, m, _ in segs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_traj")
    os.makedirs(tmp, exist_ok=True)
    py = os.path.join(ROOT, ".venv", "bin", "python3")
    for i in range(args.pilot or args.clips):
        rng = np.random.default_rng((args.seed, i))
        plan, segs = make_plan(rng)
        pf = os.path.join(tmp, f"plan_{i:03d}.json")
        with open(pf, "w") as fh:
            json.dump(plan.tolist(), fh)
        traj = os.path.join(tmp, f"traj_{i:03d}.npz")
        subprocess.run([py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"), "--steps", str(STEPS),
                        "--cmd_plan", pf, "--out", traj], check=True, capture_output=True)
        with np.load(traj, allow_pickle=True) as T:
            data = {k: T[k] for k in T.files}
        data["base_pos"], data["base_quat"] = _face_forward(data["base_pos"], data["base_quat"])
        piece = os.path.join(tmp, f"b1_babble_{i:03d}.npz")
        np.savez_compressed(piece, **data)
        subprocess.run([py, os.path.join(ROOT, "sim/render/render_b1_replay.py"), "--port", str(args.port),
                        "--scene", args.scene, "--traj", piece, "--out", out, "--fps", str(FPS),
                        "--floor_scale", "3.0", "--spawn", "0", "0", "--ego",
                        "--ego_seed", str(args.seed * 1000 + i)], check=True, capture_output=True)
        src = os.path.join(out, os.path.basename(piece))
        with np.load(src, allow_pickle=True) as b:
            merged = {k: b[k] for k in b.files}
        merged.update(condition=np.array("babble"), behaviour=np.array("babble"), level=np.array(-1),
                      expert_episode=np.array(i), embodiment=np.array("b1"), policy=np.array("gait3"),
                      command=plan)
        os.remove(src)
        np.savez_compressed(os.path.join(out, f"b1_babble_{i:03d}.npz"), **merged)
        print(f"clip {i}: " + " ".join(f"{m}[{f0}:{f0 + n}]" for f0, n, m in segs), flush=True)
        if args.pilot:
            report(os.path.join(out, f"b1_babble_{i:03d}.npz"), segs)


def report(path, segs):
    from wm.data.embodiment import REGISTRY, load
    bm = np.asarray(load(path, REGISTRY["b1"])["body_motion"])[:, :3]
    for f0, n, m in segs:
        a, b = f0 + 2, min(f0 + n - 2, len(bm))
        if b > a:
            v = np.median(bm[a:b], 0)
            print(f"    {m:<5} [{f0:2d}:{f0 + n:2d}]  fwd {v[0]:+.3f}  lat {v[1]:+.3f}  yaw {v[2]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
