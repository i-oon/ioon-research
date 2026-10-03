"""Froude labels of the walks measured at different body reference points (diagnostic only; made for F301 on the
stage-1 replay walks, re-pointed 2026-10-03 to the current deterministic walks, same fields and window rule).

Hexapod (data/counterfactual_walks/c10_walks, 4 windows x 66 frames per condition):
  head  -- the `head` field (current loader reference)
  com   -- mass-weighted mean of the 24 dynamic shapes' frame origins (state_link_pose); every
           shape's CoM sits at its frame origin (getShapeInertia transform = identity), masses from
           the scene (hex_masses.json, written by hex_shape_masses.py); static, non-respondable
           visual spheres excluded
  hip   -- centroid of the six coxa (m1_*) joints; all are children of /abdomen at fixed local
           offsets, so this is a rigid point of the abdomen: abdomen pose * (0, 0, -0.32222)
B1 (data/counterfactual_walks/b1_walks, 4 windows x 165 steps -> 66 frames via KEEP):
  base  -- base_pos (current), com -- MuJoCo subtree_com[root body] from qpos replay (mj_fwdPosition).
Labels use wm.data.embodiment.body_velocity / yaw_rate with the loader's height (median of the
CURRENT reference's z: head for hexapod, base for B1) for every reference, so only the point differs.
"""
import glob, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
import wm.data.embodiment as E
OUT = sys.argv[1]
G = 9.81

def raw_v(pos, quat, dt, emb, h):
    v = np.gradient(pos[:, :2], dt, axis=0); f = E.forward_axis(quat, emb)
    left = np.stack([-f[:, 1], f[:, 0]], 1)
    return np.stack([(v * f).sum(1), (v * left).sum(1)], 1) / np.sqrt(G * h)

def labels(refs, quat, dt, emb, h):
    out = {}
    yaw = E.yaw_rate(quat, dt, emb, h)[:, 0]
    for k, p in refs.items():
        lab = E.body_velocity(p, quat, dt, emb, height=h)
        out[k] = dict(mean=[lab[:, 0].mean(), lab[:, 1].mean(), yaw.mean()],
                      std=[lab[:, 0].std(), lab[:, 1].std(), yaw.std()],
                      rawstd=list(raw_v(p, quat, dt, emb, h).std(0)))
    return out

# ---------------- hexapod
masses = {r["name"]: r["mass"] for r in json.load(open(os.path.join(OUT, "hex_masses.json"))) if r["static"] == 0}
HIP_LOCAL = np.array([0, 0, -(0.47105 + 0.31555 + 0.18005) / 3])
hex_rows = {}; offs = []
HEX_WALKS = sorted(glob.glob(os.path.join(ROOT, "data/counterfactual_walks/c10_walks/*.npz")))
assert len(HEX_WALKS) == 24, f"expected 24 hexapod walks in data/counterfactual_walks/c10_walks, found {len(HEX_WALKS)}"
for f in HEX_WALKS:
    d = np.load(f); names = list(d["state_link_names"]); lp = d["state_link_pose"]
    idx = [names.index(n) for n in masses]; m = np.array([masses[n] for n in masses])
    com = (lp[:, idx, :3] * m[None, :, None]).sum(1) / m.sum()
    ab = R.from_quat(d["state_abdomen_quat"])
    hip = d["state_abdomen_pos"] + ab.apply(HIP_LOCAL)
    head = d["head"].astype(np.float64); quat = d["body_quat"].astype(np.float64); dt = float(d["dt"])
    # offsets in the walking frame (fwd, left, up) for reporting
    fa = E.forward_axis(quat, "hexapod"); la = np.stack([-fa[:, 1], fa[:, 0]], 1)
    for p, tag in ((com, "com"), (hip, "hip")):
        dl = p - head; offs.append((tag, (dl[:, :2] * fa).sum(1).mean(), (dl[:, :2] * la).sum(1).mean(), dl[:, 2].mean()))
    per = []
    for ws in d["window_starts"]:
        s = slice(int(ws), int(ws) + 66)
        h = float(np.median(head[s, 2]))
        per.append(labels(dict(head=head[s], com=com[s], hip=hip[s]), quat[s], dt, "hexapod", h)
                   | dict(z=dict(head=h, com=float(np.median(com[s, 2])), hip=float(np.median(hip[s, 2])))))
    hex_rows[os.path.basename(f)[:-4]] = per
o = np.array([x[1:] for x in offs if x[0] == "com"]); oh = np.array([x[1:] for x in offs if x[0] == "hip"])
print(f"hexapod dynamic mass {m.sum():.3f} kg over {len(m)} shapes")
print("mean offset from head in walking frame (fwd,left,up) m: com", o.mean(0).round(3), " hip", oh.mean(0).round(3))

# sanity: loader label on a train clip vs the references of its walk window (the loader is CoM-referenced since F301,
# with the CoM-z height; this script uses the head height for every reference, so expect com close, not equal)
tc = sorted(glob.glob(os.path.join(ROOT, "data/counterfactual_walks/c10_clips_train/*.npz")))[0]
dd = np.load(tc); mot = E._hexapod(dd)["body_motion"]
src = os.path.basename(str(dd["source_walk"])).replace(".npz", ""); wi = int(dd["window_index"])
print("sanity", os.path.basename(tc), src, wi, "loader", mot.mean(0).round(4),
      "com-ref", np.round(hex_rows[src][wi]["com"]["mean"], 4), "head-ref", np.round(hex_rows[src][wi]["head"]["mean"], 4))

def table(rows, refs, title):
    print(f"\n{title}: clip-mean over 4 windows (fwd, lat, yaw) | within-clip std of 1 s label (fwd, lat) | raw per-frame std (fwd, lat)")
    hdr = "cond".ljust(17) + "".join(f"{r+' fwd':>9}{r+' lat':>9}" for r in refs) + f"{'yaw':>8}  " + \
          "".join(f"{r+' sdL':>9}" for r in refs) + "  " + "".join(f"{r+' rawL':>10}" for r in refs) + "".join(f"{r+' rawF':>10}" for r in refs)
    print(hdr)
    for c, per in rows.items():
        mm = {r: np.mean([p[r]["mean"] for p in per], 0) for r in refs}
        sd = {r: np.mean([p[r]["std"] for p in per], 0) for r in refs}
        rs = {r: np.mean([p[r]["rawstd"] for p in per], 0) for r in refs}
        print(c.ljust(17) + "".join(f"{mm[r][0]:9.4f}{mm[r][1]:9.4f}" for r in refs) + f"{mm[refs[0]][2]:8.4f}  "
              + "".join(f"{sd[r][1]:9.4f}" for r in refs) + "  " + "".join(f"{rs[r][1]:10.3f}" for r in refs)
              + "".join(f"{rs[r][0]:10.3f}" for r in refs))
table(hex_rows, ["head", "com", "hip"], "HEXAPOD")
zs = np.array([[p["z"][k] for k in ("head", "com", "hip")] for per in hex_rows.values() for p in per])
print("hexapod median z per clip (head, com, hip): mean", zs.mean(0).round(4), "range", zs.min(0).round(3), zs.max(0).round(3))

# ---------------- B1
import mujoco
from collect_b1_walks import KEEP, PER
mdl = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "sim/assets/b1_mujoco/b1_flat.xml")); md = mujoco.MjData(mdl)
root = 1  # first body under world (trunk)
print(f"\nB1 model mass {mujoco.mj_getTotalmass(mdl):.3f} kg; root body '{mdl.body(root).name}' subtree mass {mdl.body_subtreemass[root]:.3f}")
b1_rows = {}; boff = []
for f in sorted(glob.glob(os.path.join(ROOT, "data/counterfactual_walks/b1_walks/*.npz"))):
    d = np.load(f); bp = d["base_pos"].astype(np.float64); bq = d["base_quat"].astype(np.float64); jp = d["joint_pos"].astype(np.float64)
    com = np.zeros_like(bp)
    for t in range(len(bp)):
        md.qpos[0:3] = bp[t]; md.qpos[3:7] = bq[t]; md.qpos[7:19] = jp[t]
        mujoco.mj_fwdPosition(mdl, md); com[t] = md.subtree_com[root]
    rot = R.from_quat(bq[:, [1, 2, 3, 0]])
    boff.append(rot.inv().apply(com - bp))
    per = []
    for ws in d["window_starts"]:
        idx = int(ws) + KEEP
        s_bp, s_com, s_q = bp[idx], com[idx], bq[idx]
        h = float(np.median(s_bp[:, 2]))
        per.append(labels(dict(base=s_bp, com=s_com), s_q, 0.05, "b1", h))
    b1_rows[os.path.basename(f)[:-4]] = per
bo = np.concatenate(boff)
print("B1 CoM offset from base in base frame (x fwd, y left, z up) m: mean", bo.mean(0).round(4), "std", bo.std(0).round(4))
table(b1_rows, ["base", "com"], "B1")
json.dump(dict(hex=hex_rows, b1=b1_rows), open(os.path.join(OUT, "centre_reference.json"), "w"), default=float)
