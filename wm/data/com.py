"""Centre-of-mass (CoM) reference point for Froude labels (FINDINGS F301).

The loader (`wm.data.embodiment`) measures body velocity and the Froude height at the per-frame `com_pos`
field (world, metres) whenever a clip carries it. These helpers compute that field, identically for the
collectors and for back-filling existing clips.

Hexapod (CoppeliaSim): mass-weighted mean of the dynamic (non-static) shapes' frame origins from
`state_link_pose` (collect_ik --record_state). Every shape's CoM sits at its frame origin (getShapeInertia
transform = identity); static, non-respondable visual spheres are excluded. For medauroidea_c10f10t10 this
is 29 shapes, 1.920 kg (masses in sim/env/medauroidea_c10f10t10_masses.json, read from the scene by
scripts/diagnostics/dataset/hex_shape_masses.py). Offset from the head (walking frame): (-0.246, -0.001, -0.030) m.
Other hexapod bodies use their own scene's masses: `hex_com(..., morph="c08f09t09")` reads
sim/env/medauroidea_c08f09t09_masses.json (same 29 shape names and masses as c10, 1.920 kg; the CoM differs
through the link geometry only). A morph without a masses JSON raises instead of silently using c10's.

B1 (MuJoCo): subtree CoM of the root body (trunk, body 1) from qpos = (base_pos, base_quat (w,x,y,z),
joint_pos in SDK / qpos order), via mj_fwdPosition. Offset from the base: (-0.0185, 0, -0.033) m (base frame).
Both are rigid-motion equivariant, so they may be computed before or after a planar re-centring / rotation
as long as the pose used is the stored one.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEX_MASSES = os.path.join(ROOT, "sim/env/medauroidea_c10f10t10_masses.json")


def hex_masses_path(morph="c10f10t10"):
    """Masses JSON of hexapod body `morph` (written from its scene by hex_shape_masses.py --scene)."""
    p = os.path.join(ROOT, f"sim/env/medauroidea_{morph}_masses.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no masses for {morph}: run scripts/diagnostics/dataset/hex_shape_masses.py "
                                f"--scene medauroidea_{morph}.ttt --out {os.path.relpath(p, ROOT)}")
    return p
B1_MODEL = os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat.xml")


def hex_masses(path=HEX_MASSES):
    """{shape name: mass} of the dynamic shapes (static == 0)."""
    return {r["name"]: float(r["mass"]) for r in json.load(open(path)) if r["static"] == 0}


def hex_com(link_names, link_pose, masses=None, morph="c10f10t10"):
    """(T, 3) CoM from `state_link_names` (S,) and `state_link_pose` (T, S, 7). `masses`: {name: mass}
    of the shapes to include (default: the dynamic shapes of body `morph`'s scene, e.g. "c08f09t09")."""
    masses = hex_masses(hex_masses_path(str(morph))) if masses is None else masses
    names = [str(n) for n in link_names]
    idx = [names.index(n) for n in masses]
    m = np.array([masses[n] for n in masses], np.float64)
    lp = np.asarray(link_pose, np.float64)
    return (lp[:, idx, :3] * m[None, :, None]).sum(1) / m.sum()


_B1 = {}


def b1_com(base_pos, base_quat, joint_pos, model_path=B1_MODEL):
    """(T, 3) MuJoCo subtree CoM of the root body for each recorded pose."""
    import mujoco
    if model_path not in _B1:
        mdl = mujoco.MjModel.from_xml_path(model_path)
        _B1[model_path] = (mdl, mujoco.MjData(mdl))
    mdl, md = _B1[model_path]
    bp, bq, jp = (np.asarray(x, np.float64) for x in (base_pos, base_quat, joint_pos))
    out = np.zeros((len(bp), 3))
    for t in range(len(bp)):
        md.qpos[0:3] = bp[t]; md.qpos[3:7] = bq[t]; md.qpos[7:19] = jp[t]
        mujoco.mj_fwdPosition(mdl, md)
        out[t] = md.subtree_com[1]
    return out
