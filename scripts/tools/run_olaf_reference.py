"""Run the lab's Olaf scene as it ships and record its gait, as the reference to compare against.

`--gait cpg` in `sim/collect/collect_ik.py` is a port of the oscillator embedded in
`student_Locomotion_Control_olaf_6legs.ttt`, and its foot contacts do not come out as the clean
alternating tripod its sign pattern suggests. Two explanations fit that: the port is wrong, or the
original never produced one either and the expectation was mine. Running the original settles it,
and nothing else does -- six failed guesses about the mechanism preceded this.

Nothing here drives the robot. The scene's own script does, through `sysCall_actuation`; this only
starts the simulation and reads what happens.

  .venv/bin/python3 scripts/tools/run_olaf_reference.py --port 23004
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE = "student_Locomotion_Control_olaf_6legs.ttt"
LEGS = ["L1", "L2", "L3", "R1", "R2", "R3"]      # the scene's own names
OURS = ["FL", "ML", "HL", "FR", "MR", "HR"]      # what its script calls them: L1=FL, L2=ML, ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=23004)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--out", default="results/wm/dataset/olaf_reference.npz")
    args = ap.parse_args()

    sim = RemoteAPIClient(port=args.port).require("sim")
    sim.loadScene(os.path.join(ROOT, SCENE))
    force = [sim.getObject(f"/forceSensor_{leg}") for leg in LEGS]
    joints = [sim.getObject(f"/joint_{j}_{leg}") for leg in LEGS for j in (1, 2, 3)]
    body = sim.getObject("/Olaf")

    sim.setStepping(True)
    sim.startSimulation()
    contacts, pos, quat, jpos = [], [], [], []
    for _ in range(args.frames):
        sim.step()
        contacts.append([sim.readForceSensor(h)[1][2] for h in force])
        pos.append(sim.getObjectPosition(body, sim.handle_world))
        quat.append(sim.getObjectQuaternion(body, sim.handle_world))
        jpos.append([sim.getJointPosition(h) for h in joints])
    sim.stopSimulation()

    out = dict(forces=np.abs(np.asarray(contacts, np.float32)),
               head=np.asarray(pos, np.float32),
               body_quat=np.asarray(quat, np.float32),
               actions=np.asarray(jpos, np.float32),
               foot_order=np.array(OURS))
    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)

    f = out["forces"]
    print(f"{len(f)} frames, force range {f.min():.3f} to {f.max():.3f} N")
    print(f"travel {np.linalg.norm(out['head'][-1, :2] - out['head'][0, :2]):.2f} m, "
          f"hip {np.median(out['head'][:, 2]):.3f} m")
    print(f"-> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
