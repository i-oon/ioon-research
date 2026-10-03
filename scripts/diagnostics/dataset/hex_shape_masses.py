"""Read mass, local CoM and dynamic flags of every shape under /abdomen in a hexapod scene (--scene, default
medauroidea_c10f10t10.ttt; c08: --scene medauroidea_c08f09t09.ttt --out sim/env/medauroidea_c08f09t09_masses.json).
Needs a headless CoppeliaSim on --port. Writes JSON (name order = sim.getObjectsInTree(abdomen, shape),
same as collect_ik --record_state's state_link_names)."""
import argparse, json, os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=23987); ap.add_argument("--out"); ap.add_argument("--scene", default="medauroidea_c10f10t10.ttt")
a = ap.parse_args()
sim = RemoteAPIClient("localhost", port=a.port).require("sim")
sim.loadScene(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../sim/env", a.scene)))
body = sim.getObject("/abdomen")
rows = []
for h in sim.getObjectsInTree(body, sim.object_shape_type):
    name = sim.getObjectAlias(h, 1)
    static = sim.getObjectInt32Param(h, sim.shapeintparam_static)
    respondable = sim.getObjectInt32Param(h, sim.shapeintparam_respondable)
    mass = sim.getShapeMass(h)
    try:
        m2, inertia, com = sim.getShapeInertia(h)  # com as a 4x3 matrix in shape frame (v4.5+)
        com = [com[3], com[7], com[11]]
    except Exception as e:
        com = None
    rows.append(dict(name=name, static=static, respondable=respondable, mass=mass, com_local=com))
    print(rows[-1])
json.dump(rows, open(a.out, "w"), indent=1)
