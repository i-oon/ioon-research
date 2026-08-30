"""Which "zero action" means *do not move* rather than *fall over*, on each robot.

    .venv/bin/python3 scripts/diagnostics/null_action.py

**A prerequisite for the ActSWM rebuild, not a nicety.** The action-sensitivity hinge contrasts a
rollout driven by the real actions against one driven by a **null** action. If the null makes the
robot collapse, the hinge is trained to separate "walking" from "falling", which is trivially easy
and teaches nothing about the action channel. The null has to mean *no commanded motion* and it has
to mean that on **both** bodies, since the same objective trains on both.

Three candidates, each held constant for three seconds:

    hold      the pose the robot settled into, commanded back to itself
    neutral   the stance the scene or the dataset starts from
    zero      the literal zero vector in the action space

Reported per robot: does the existing fall check fire, how far the body travelled (should be about
nothing), and how much it jitters while doing it.
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))

from wm.data.embodiment import REGISTRY, load  # noqa: E402


def summarise(pos, quat, dt, embodiment, settled_z):
    from plan_without_library import channels_of
    pos = np.asarray(pos, np.float64)
    ch = channels_of(pos, np.asarray(quat, np.float64), dt, embodiment)
    travel = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
    fell = bool((pos[:, 2] < 0.6 * settled_z).any())
    # jitter: how much the body wobbles once it should be still, in millimetres per step
    jitter = float(np.median(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)) * 1000)
    return {"travel": travel, "fell": fell, "jitter_mm": jitter,
            "fwd": float(np.median(ch[:, 0])), "lat": float(np.median(ch[:, 1])),
            "yaw": float(np.median(ch[:, 2])), "min_z": float(pos[:, 2].min()),
            "settled_z": settled_z}


def row(name, r):
    verdict = "FALLS" if r["fell"] else ("still" if r["travel"] < 0.02 else "drifts")
    print(f"  {name:<9}{r['travel']:>9.4f}{r['jitter_mm']:>10.2f}{r['fwd']:>+9.3f}"
          f"{r['lat']:>+8.3f}{r['yaw']:>+8.3f}{r['min_z']:>9.4f}   {verdict}")


def b1(args):
    import mujoco
    from rollout_b1_mujoco import MODEL
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    with np.load(os.path.join(ROOT, args.b1_clip), allow_pickle=True) as raw:
        z0 = float(raw["base_pos"][0][2])
        q0 = np.asarray(raw["base_quat"][0], np.float64)
        j0 = np.asarray(raw["joint_pos"][0], np.float64)
    sub = int(round(0.05 / m.opt.timestep))

    def run(kind):
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = [0.0, 0.0, z0]; d.qpos[3:7] = q0; d.qpos[7:19] = j0
        d.ctrl[:] = j0
        mujoco.mj_forward(m, d)
        for _ in range(25):
            mujoco.mj_step(m, d)
        settled_z, settled_j = float(d.qpos[2]), d.qpos[7:19].copy()
        target = {"hold": settled_j, "stance": j0, "mean": args.b1_mean,
                  "zero": np.zeros(12)}[kind]
        pos, quat = [d.qpos[0:3].copy()], [d.qpos[3:7].copy()]
        for _ in range(args.steps):
            d.ctrl[:] = target
            for _ in range(sub):
                mujoco.mj_step(m, d)
            pos.append(d.qpos[0:3].copy()); quat.append(d.qpos[3:7].copy())
        return summarise(pos, quat, 0.05, "b1", settled_z)

    print(f"\nB1 (MuJoCo), {args.steps} steps of 50 ms")
    print(f"  {'null':<9}{'travel m':>9}{'jitter mm':>10}{'fwd':>9}{'lat':>8}{'yaw':>8}"
          f"{'min z':>9}")
    return {k: run(k) for k in ("hold", "stance", "mean", "zero")}


def hexapod(args):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    clip = load(os.path.join(ROOT, args.hex_clip), REGISTRY["hexapod"])
    cmds = np.asarray(clip["actions"], np.float32)[:args.steps]
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    print(f"\nhexapod c10f10t10 (CoppeliaSim), {len(cmds)} steps of 50 ms")
    print(f"  {'null':<9}{'travel m':>9}{'jitter mm':>10}{'fwd':>9}{'lat':>8}{'yaw':>8}"
          f"{'min z':>9}")
    out = {}
    for kind in ("hold", "stance", "mean", "zero"):
        target = {"hold": cmds[0], "stance": cmds[0], "mean": args.hex_mean,
                  "zero": np.zeros(18, np.float32)}[kind]
        # `hold` and `neutral` coincide here: the warmup holds the clip's first command, so the
        # pose the robot settles into *is* the dataset's neutral stance. Kept as separate rows so
        # the two robots' tables line up.
        _f, _a, _fo, heads, oris = drive_and_record(
            sim, "medauroidea_stick_insect.ttt", cmds, 0.0, 20,
            cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0),
            policy=lambda obs, t, v=target: v)
        heads = np.asarray(heads, np.float64)
        out[kind] = summarise(heads, oris, 0.05, "hexapod", float(np.median(heads[:5, 2])))
        row(kind, out[kind])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1_clip", default="data/beh12_b1_flat/b1_ep3.npz")
    ap.add_argument("--hex_clip", default="data/beh12_c10f10t10_flat/hexapod_ep100.npz")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--skip_hex", action="store_true")
    args = ap.parse_args()

    # **the dataset-mean pose, which is a different object from the stance the clips start in.**
    # A mean over a gait cycle averages swing against stance; whether that posture holds the robot
    # up is a question, not an assumption -- F137 measured sampling around it travelling backwards.
    args.b1_mean = np.concatenate([load(f, REGISTRY["b1"])["actions"]
                                   for f in sorted(glob.glob(os.path.join(
                                       ROOT, "data/beh12_b1_flat/*.npz")))]).mean(0).astype(np.float64)
    args.hex_mean = np.concatenate([load(f, REGISTRY["hexapod"])["actions"]
                                    for f in sorted(glob.glob(os.path.join(
                                        ROOT, "data/beh12_c10f10t10_flat/*.npz")))]).mean(0).astype(np.float32)

    rb = b1(args)
    for k, v in rb.items():
        row(k, v)
    rh = {} if args.skip_hex else hexapod(args)

    print("\n  `travel` should be about nothing, `jitter` is the body's own wobble per step, and")
    print("  the verdict is the existing fall check. **A null that falls trains the hinge to")
    print("  separate walking from collapsing, which is not the contrast the objective wants.**")
    both = [k for k in rb if k in rh and not rb[k]["fell"] and not rh[k]["fell"]
            and rb[k]["travel"] < 0.05 and rh[k]["travel"] < 0.05]
    print(f"\n  usable on BOTH bodies: {both if both else 'none -- see the table'}")


if __name__ == "__main__":
    main()
