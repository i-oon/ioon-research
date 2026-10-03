"""Unit check for the F293-fixed `_face_forward`: body-frame Froude must not depend on the start yaw.

A real raw B1 path (exact replay of a beh24 turn clip, where the bug was largest) is rotated by random
world yaws; each rotated copy goes through `recollect_b1_more._face_forward` and
`recollect_b1_turns._face_forward`, and the loader's body_motion (wm.data.embodiment, b1) is compared
with that of the un-rotated path. Pass: max diff < 1e-9 and every output quaternion unit (< 1e-12).
Also reports what the pre-fix function gives on the same inputs.

    .venv/bin/python3 scripts/diagnostics/dataset/check_face_forward_invariance.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts/dataset", "scripts/diagnostics/dataset"):
    sys.path.insert(0, os.path.join(ROOT, p))
import build_b1_cf_branches as B  # noqa: E402
from check_face_forward_labels import labels, raw_window  # noqa: E402
from fix_face_forward_f293 import buggy_face_forward  # noqa: E402
from recollect_b1_more import _face_forward as ff_more  # noqa: E402
from recollect_b1_turns import _face_forward as ff_turns  # noqa: E402


def rot(P, Q, a):
    c, s = np.cos(a), np.sin(a)
    P = P.copy()
    P[:, :2] = P[:, :2] @ np.array([[c, -s], [s, c]]).T
    rw, rz = np.cos(a / 2), np.sin(a / 2)
    w, x, y, z = Q.T
    return P, np.stack([rw * w - rz * z, rw * x - rz * y, rw * y + rz * x, rw * z + rz * w], 1)


def main():
    path = os.path.join(ROOT, "data/egocentric_v3/beh24_b1_ego_flat_cleantrain/b1_ep1303.npz")
    d = np.load(path, allow_pickle=True)
    P, Q = raw_window(B.locate(path))
    ref = labels(d, P, Q)
    rng = np.random.default_rng(0)
    worst = {"more": 0.0, "turns": 0.0, "buggy": 0.0}
    unit = 0.0
    for a in rng.uniform(-np.pi, np.pi, 50):
        Pr, Qr = rot(P, Q, a)
        for name, f in (("more", ff_more), ("turns", ff_turns), ("buggy", buggy_face_forward)):
            Pf, Qf = f(Pr, Qr)
            worst[name] = max(worst[name], float(np.abs(labels(d, Pf, Qf) - ref).max()))
            if name != "buggy":
                unit = max(unit, float(np.abs(np.linalg.norm(Qf, axis=1) - 1).max()))
    ok = worst["more"] < 1e-9 and worst["turns"] < 1e-9 and unit < 1e-12
    print(f"50 random yaws on {os.path.basename(path)}: max |Froude diff| recollect_b1_more {worst['more']:.1e}, "
          f"recollect_b1_turns {worst['turns']:.1e}; max | |q|-1 | {unit:.1e}; pre-fix function {worst['buggy']:.3f}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
