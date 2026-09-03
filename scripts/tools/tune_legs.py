"""Per-leg offsets and gains that make six unequal legs trace the same stroke.

This animal's legs are not interchangeable: the pairs measure 0.771, 0.489 and 0.638 long and
attach 0.29 apart along the body. One oscillator amplitude therefore produces three different
strokes -- measured on `c10f10t10`, the front feet lift 0.111 and reach 0.038 deeper than the middle
pair, which lift only 0.045. Whichever feet reach lowest carry the robot and the rest brush the
floor, so the contact pattern is decided by leg length rather than by the phase the oscillator asks
for, and no tripod appears however the signals are arranged.

Two numbers per leg fix that, and both are read off the kinematics rather than guessed:

    gain     scales the lift and extend amplitudes so every foot rises the same distance
    offset   shifts the standing pose so every stroke bottoms out at the same height

Purely kinematic -- joints are set and the foot is read, with no simulation -- so the whole search
costs seconds.

  .venv/bin/python3 scripts/tools/tune_legs.py --port 23004 --bias <clip.npz>
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGS = ["FL", "ML", "HL", "FR", "MR", "HR"]
SEG = ["m1", "m2", "m3"]
TRIPOD_A = {"FL", "HL", "MR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23004)
    ap.add_argument("--morph", default="c10f10t10")
    ap.add_argument("--bias", required=True, help="clip npz whose mean actions give the pose")
    ap.add_argument("--amps", type=float, nargs=3, default=(0.25, 0.20, 0.30))
    ap.add_argument("--lead", type=float, default=0.25)
    ap.add_argument("--ft_phase", type=float, default=0.125)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--damp", type=float, default=0.5,
                    help="fraction of each solved step to take, since gain and offset interact")
    ap.add_argument("--max_offset", type=float, default=0.20,
                    help="rad; keeps the standing pose inside the joint range")
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    sim.loadScene(os.path.join(ROOT, "sim", "env", f"medauroidea_{args.morph}.ttt"))
    joints = {l: [sim.getObject(f"/{s}_{l}") for s in SEG] for l in LEGS}
    feet = {l: sim.getObject(f"/foot_{l}") for l in LEGS}
    body = sim.getObject("/abdomen")
    hips = np.array([sim.getObjectPosition(sim.getObject(f"/m1_{l}"), body) for l in LEGS])
    up = int(np.argmin(np.ptp(hips, axis=0)))

    bias = np.load(args.bias, allow_pickle=True)["actions"].mean(0).astype(float)
    for a, c in (("FL", "FR"), ("ML", "MR"), ("HL", "HR")):
        ia, ic = LEGS.index(a) * 3, LEGS.index(c) * 3
        for k in range(3):
            half = 0.5 * (bias[ia + k] - bias[ic + k])
            bias[ia + k], bias[ic + k] = half, -half

    phase = np.linspace(0, 2 * np.pi, args.samples, endpoint=False)
    wave = np.stack([np.sin(phase + 2 * np.pi * args.lead), np.sin(phase),
                     np.sin(phase + 2 * np.pi * args.ft_phase)], axis=1)

    def stroke(leg, gain, offset):
        i = LEGS.index(leg)
        s = 1.0 if leg in TRIPOD_A else -1.0
        m = 1.0 if leg.endswith("L") else -1.0
        hs = []
        for w in wave:
            for k in range(3):
                a = args.amps[k] * (gain if k > 0 else 1.0)
                q = bias[i * 3 + k] + s * m * a * w[k] + (offset if k > 0 else 0.0)
                sim.setJointPosition(joints[leg][k], float(q))
            hs.append(sim.getObjectPosition(feet[leg], body)[up])
        return float(min(hs)), float(max(hs) - min(hs))

    gain = {l: 1.0 for l in LEGS}
    offset = {l: 0.0 for l in LEGS}
    for r in range(args.rounds):
        # gain and offset interact -- raising the amplitude lowers the stroke's bottom as well as
        # its top -- so each is solved against a *freshly measured* state rather than against the
        # numbers that opened the round. Reading `low` once and then using it after the gain had
        # already changed is what sent the first version of this to gain 3.2 and offset 0.74 rad.
        lift = {l: stroke(l, gain[l], offset[l])[1] for l in LEGS}
        t_lift = float(np.median(list(lift.values())))
        for l in LEGS:
            step = t_lift / max(lift[l], 1e-6)
            gain[l] = float(np.clip(gain[l] * (1 + args.damp * (step - 1)), 0.4, 2.0))

        low = {l: stroke(l, gain[l], offset[l])[0] for l in LEGS}
        t_low = float(np.median(list(low.values())))
        for l in LEGS:
            probe = stroke(l, gain[l], offset[l] + 0.02)[0] - low[l]
            if abs(probe) < 1e-6:
                continue
            want = (t_low - low[l]) * 0.02 / probe
            offset[l] = float(np.clip(offset[l] + args.damp * want, -args.max_offset,
                                      args.max_offset))

        now = {l: stroke(l, gain[l], offset[l]) for l in LEGS}
        print(f"round {r + 1}  lift spread {np.ptp([v[1] for v in now.values()]):.4f}   "
              f"depth spread {np.ptp([v[0] for v in now.values()]):.4f}")

    print(f"\n{'leg':<5}{'gain':>9}{'offset':>10}")
    for l in LEGS:
        print(f"{l:<5}{gain[l]:>9.3f}{offset[l]:>10.4f}")
    out = os.path.join(ROOT, "results", "wm", "dataset", f"legtune_{args.morph}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, legs=np.array(LEGS),
             gain=np.array([gain[l] for l in LEGS], np.float32),
             offset=np.array([offset[l] for l in LEGS], np.float32))
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
