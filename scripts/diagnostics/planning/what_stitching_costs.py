"""Does switching between recorded clips cost travel, or was that an artefact of a residual?

F102 decomposed the closed loop's speed shortfall into three terms and reported the third --
stitching -- as **a residual rather than a measurement**: actual divided by (picks x replay),
which absorbs every error in the other two. On three runs it read 1.08 forward, 0.59 turning, 0.20
sideways, which would say switching clips costs the turning and lateral channels badly. That is a
hypothesis and this file tests it directly.

**The test.** Take the exact sequence of commands the loop executed -- candidate A's row 12,
candidate C's row 13, and so on -- and replay it in MuJoCo with no planner and no vision. Compare
against replaying, alone, the single candidate the loop chose most often. Same physics, same
seeding, same episode length. **The difference is what the switching itself costs**, with
selection quality and replay fidelity held out of it, because both sequences are replays.

    .venv/bin/python3 scripts/diagnostics/planning/what_stitching_costs.py \\
        results/wm/closed_loop/b1_physics3/*.npz --data data/allocentric/beh12_b1_flat
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
import mujoco  # noqa: E402

from diagnostics.planning.score_closed_loop import channels  # noqa: E402
from rollout_b1_mujoco import ACTION_SCALE, DEFAULT_IL, MODEL, il_to_sdk  # noqa: E402


def run_actions(actions, seed_clip, sub=10, settle=25):
    """Replay a command sequence from the demonstration's own starting state."""
    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    with np.load(seed_clip, allow_pickle=True) as z:
        d.qpos[0:3] = [0.0, 0.0, float(z["base_pos"][0][2])]
        d.qpos[3:7] = np.asarray(z["base_quat"][0], np.float64)
        d.qpos[7:19] = np.asarray(z["joint_pos"][0], np.float64)
        d.qvel[6:18] = np.asarray(z["joint_vel"][0], np.float64)
        d.ctrl[:] = np.asarray(z["joint_pos"][0], np.float64)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    P, Q = [], []
    for a in actions:
        d.ctrl[:] = np.clip(il_to_sdk(DEFAULT_IL + ACTION_SCALE * a),
                            m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(sub):
            mujoco.mj_step(m, d)
        P.append(d.qpos[0:3].copy()); Q.append(d.qpos[3:7].copy())
    return np.array(P), np.array(Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    lib, cond_of = {}, {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
            lib.setdefault(c, np.asarray(z["action"], np.float32))
            cond_of.setdefault(c, p)

    print(f"  {'run':<24}{'switches':>9}{'stitched':>10}{'single':>9}{'ratio':>8}   channel")
    for run in args.runs:
        with np.load(run, allow_pickle=True) as d:
            chosen = [str(c) for c in np.asarray(d["chosen"], str)]
            want = str(d["condition"])
            demo = str(d["demo"])
        seed = os.path.join(ROOT, args.data, demo)

        stitched, planned = [], []
        for t, c in enumerate(chosen):
            name = c.split("warm:")[-1]
            src = lib.get(name, lib[want])
            stitched.append(src[min(t, len(src) - 1)])
            if not c.startswith("warm:"):
                planned.append(name)
        top = collections.Counter(planned).most_common(1)[0][0] if planned else want
        single = [lib[top][min(t, len(lib[top]) - 1)] for t in range(len(chosen))]

        ps, qs = run_actions(np.array(stitched), seed)
        pu, qu = run_actions(np.array(single), seed)
        k = int(np.argmax(np.abs(np.median(channels(
            *[np.asarray(v, np.float64) for v in (pu, qu)], 0.05, "b1"), 0))))
        vs = float(np.median(channels(ps.astype("float64"), qs.astype("float64"), 0.05, "b1")[:, k]))
        vu = float(np.median(channels(pu.astype("float64"), qu.astype("float64"), 0.05, "b1")[:, k]))
        sw = sum(1 for i in range(1, len(planned)) if planned[i] != planned[i - 1])
        names = ("forward", "lateral", "yaw")
        print(f"  {os.path.basename(run):<24}{sw:>9}{vs:>10.3f}{vu:>9.3f}"
              f"{vs / vu if abs(vu) > 1e-6 else float('nan'):>8.2f}   {names[k]}")

    print("\n  stitched  the exact command sequence the loop executed, replayed with no planner")
    print("  single    the candidate it chose most often, replayed alone, same seeding")
    print("  ratio     below 1.0 means switching between clips costs travel on that channel")


if __name__ == "__main__":
    main()
