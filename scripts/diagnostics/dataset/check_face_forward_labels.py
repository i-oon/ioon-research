"""How much does `_face_forward`'s quaternion aliasing bug change the B1 Froude labels?

`recollect_b1_more._face_forward` / `recollect_b1_turns._face_forward` update `quat[:, 0]` and then read
`qw = quat[:, 0]` (a view) for the x/y/z components, so the stored `base_quat` is neither a pure yaw
rotation of the physics nor unit norm. For every reproducible B1 clip this re-runs the source rollout
exactly (build_b1_cf_branches.locate), takes the RAW world base_pos/base_quat of its window, applies a
CORRECT face-forward (rotate about the first point by -yaw0, quaternion product from unmodified copies,
normalised), and computes body_motion through the loader's own code path (wm.data.embodiment._b1).
Compared against the labels the loader produces from the stored fields.

    .venv/bin/python3 scripts/diagnostics/dataset/check_face_forward_labels.py
CPU only.
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts/dataset"))
import build_b1_cf_branches as B  # noqa: E402
from wm.data.embodiment import REGISTRY  # noqa: E402

DIRS = ("beh24_b1_ego_flat_cleantrain", "beh24_b1_ego_flat_cleanval", "beh12_b1_ego_flat_cleantrain")
CH = ("fwd", "lat", "yaw")


class Fields(dict):
    @property
    def files(self):
        return list(self)


def correct_face_forward(pos, quat):
    pos, quat = np.asarray(pos, float).copy(), np.asarray(quat, float).copy()
    w, x, y, z = quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = pos[:, :2] - pos[0, :2]
    pos[:, 0], pos[:, 1] = pos[0, 0] + c * d[:, 0] - s * d[:, 1], pos[0, 1] + s * d[:, 0] + c * d[:, 1]
    rw, rz = np.cos(-yaw / 2), np.sin(-yaw / 2)
    qw, qx, qy, qz = (quat[:, i].copy() for i in range(4))
    out = np.stack([rw * qw - rz * qz, rw * qx - rz * qy, rw * qy + rz * qx, rw * qz + rz * qw], 1)
    return pos, out / np.linalg.norm(out, axis=1, keepdims=True)


def raw_window(loc):
    r = B.Roll(loc["policy"], loc["gains"], loc.get("model", B.MODEL))
    for _ in range(B.POLICY_WARMUP):
        r.step(*loc["cmd"])
    rec = [r.step(*loc["cmd"]) for _ in range(loc["off"] + B.PER)]
    idx = loc["off"] + B.KEEP
    return np.asarray([rec[i]["base_pos"] for i in idx]), np.asarray([rec[i]["base_quat"] for i in idx])


def labels(d, pos, quat):
    f = Fields(base_pos=pos, base_quat=quat, action=d["action"], foot_contact=d["foot_contact"], dt=d["dt"],
               expert_episode=d["expert_episode"])
    return np.asarray(REGISTRY["b1"].read(f)["body_motion"])[:, :3]


def source_script(ep, d):
    if ep < 1000:
        return "beh12 speed (older collector, PI 2.5/1.0)"
    if ep < 2000:
        return "recollect_b1_turns.py"
    return "recollect_b1_more.py"


def main():
    rows = []
    for dn in DIRS:
        for p in sorted(glob.glob(os.path.join(ROOT, "data/egocentric_v3", dn, "*.npz"))):
            d = np.load(p, allow_pickle=True)
            ep = int(d["expert_episode"])
            try:
                loc = B.locate(p)
            except RuntimeError:
                print(f"{dn}/{os.path.basename(p)}: not reproducible, skipped", flush=True)
                continue
            P, Q = raw_window(loc)
            Pc, Qc = correct_face_forward(P, Q)
            bug = labels(d, d["base_pos"].astype(np.float64), d["base_quat"].astype(np.float64))
            ok = labels(d, Pc, Qc)
            diff = np.abs(bug - ok)
            rows.append(dict(dir=dn, clip=os.path.basename(p), cond=str(d["condition"]), src=source_script(ep, d),
                             mx=diff.max(0), mean=diff.mean(0), cm_bug=bug.mean(0), cm_ok=ok.mean(0),
                             qnorm=float(np.abs(np.linalg.norm(d["base_quat"], axis=1) - 1).max()),
                             dpos=float(np.abs(Pc - d["base_pos"]).max())))
            print(f"{dn[:14]:<14} {os.path.basename(p):<15} {str(d['condition']):<16} max|d| "
                  + " ".join(f"{v:.4f}" for v in diff.max(0)) + "  mean|d| " + " ".join(f"{v:.4f}" for v in diff.mean(0))
                  + "  clip-mean d " + " ".join(f"{v:+.4f}" for v in (bug - ok).mean(0))
                  + f"  |q|-1 {rows[-1]['qnorm']:.3f} pos {rows[-1]['dpos']:.1e}", flush=True)

    print("\n=== summary per directory / source script (channels fwd lat yaw)")
    for key in ("dir", "src"):
        for g in sorted({r[key] for r in rows}):
            R = [r for r in rows if r[key] == g]
            aff = sum(r["mx"].max() > 1e-4 for r in R)
            mx = np.max([r["mx"] for r in R], 0)
            mn = np.mean([r["mean"] for r in R], 0)
            cm = np.max([np.abs(r["cm_bug"] - r["cm_ok"]) for r in R], 0)
            print(f"  {g:<44} n={len(R):<3} affected(>1e-4)={aff:<3} max " + " ".join(f"{v:.4f}" for v in mx)
                  + " | mean " + " ".join(f"{v:.4f}" for v in mn) + " | max clip-mean " + " ".join(f"{v:.4f}" for v in cm))
    allv = np.concatenate([np.abs(np.stack([r["cm_ok"] for r in rows]))], 0)
    print("  label scale, mean |clip-mean| (correct):", " ".join(f"{v:.4f}" for v in allv.mean(0)))

    lib = [r for r in rows if r["dir"] == "beh12_b1_ego_flat_cleantrain"]
    print(f"\n=== library ({len(lib)} clips): Spearman buggy vs correct clip-mean Froude")
    for j, c in enumerate(CH):
        rho = spearmanr([r["cm_bug"][j] for r in lib], [r["cm_ok"][j] for r in lib]).correlation
        print(f"  {c}: rho {rho:.4f}")


if __name__ == "__main__":
    main()
