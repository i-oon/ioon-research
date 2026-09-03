"""Does replaying a recorded hexapod clip reproduce its speed, as F93 implies it must?

F102 decomposed the closed loop's speed shortfall into three terms. The second -- **what replaying
a recorded command sequence costs before any planner is involved** -- was measured on the B1 at
**0.84, 0.76, 0.99** of the clip's own recorded speed, and attributed to the B1's action being a
policy's response to state (F93). The hexapod's commands come from IK and a clock and read no
state at all, **so this term should be absent there.** That is a prediction, and it has not been
tested; if it fails, the explanation for the B1 is incomplete.

Replays each condition's own commands through the same physics the closed loop uses, with the same
camera and spawn, and compares the achieved body speed against the clip's recording.

    .venv/bin/python3 scripts/diagnostics/planning/hexapod_replay_fidelity.py \\
        --data data/allocentric/beh12_c08f09t09_flat --scene medauroidea_c08f09t09.ttt
"""
import argparse
import glob
import os
import sys

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))

from collect_ik import drive_and_record  # noqa: E402
from diagnostics.planning.score_closed_loop import channels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--conditions", nargs="*", default=["speed_c5.8", "turn_s0.05",
                                                        "side_R_lvl1"])
    ap.add_argument("--travel", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cam_dx", type=float, default=-0.6)
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()

    picks = {}
    for p in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        with np.load(p, allow_pickle=True) as z:
            c = str(z["condition"])
        if c in args.conditions and c not in picks:
            picks[c] = p

    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    names = ("forward", "lateral", "yaw")
    print(f"  {'condition':<14}{'channel':>9}{'recorded':>10}{'replayed':>10}{'ratio':>8}")
    ratios = []
    for cond in args.conditions:
        if cond not in picks:
            continue
        with np.load(picks[cond], allow_pickle=True) as z:
            cmds = np.asarray(z["actions"], np.float32)
            rec = channels(z["head"].astype("float64"), z["body_quat"].astype("float64"),
                           0.05, "hexapod")
        _f, _a, _fo, heads, oris = drive_and_record(
            sim, args.scene, cmds, args.travel, args.warmup, cam_dx=args.cam_dx)
        rep = channels(np.asarray(heads, "float64"), np.asarray(oris, "float64"), 0.05, "hexapod")
        k = int(np.argmax(np.abs(np.median(rec, 0))))
        a, b = float(np.median(rec[:, k])), float(np.median(rep[:, k]))
        ratios.append(b / a if abs(a) > 1e-9 else float("nan"))
        print(f"  {cond:<14}{names[k]:>9}{a:>10.3f}{b:>10.3f}{ratios[-1]:>8.2f}")

    if ratios:
        m = float(np.nanmean(ratios))
        print(f"\n  mean {m:.2f}. The B1's equivalent is 0.84 / 0.76 / 0.99.\n")
        if m > 0.95:
            print("**The term is absent on the hexapod, as predicted.** Its commands are written by")
            print("a clock and read no state, so re-issuing them reproduces the motion. The replay")
            print("loss in the decomposition belongs to the B1 alone, and F93's account of why is")
            print("complete: an action that is a response cannot be re-issued, one that is a plan can.")
        else:
            print("**The prediction fails.** The hexapod loses speed on replay too, so 'the action")
            print("is a response' does not explain the B1's loss on its own, and the decomposition")
            print("in F102 needs a different second term.")


if __name__ == "__main__":
    main()
