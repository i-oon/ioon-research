"""The 24 hexapod beh24 conditions and the split / room / window rules shared by every current data script
(collect_c10_walks_and_branches, build_branches, collect_b1_walks, collect_c08_test_set). Moved here unchanged
2026-10-03 from the stage-1 replay collector (the old stage-1 replay script, which imports them from here too).

Condition order / index (DATA_PLAN section 0 item 3): fwd speed 1-4, bwd 1-4, turn left 1-4, turn right 1-4,
side L 1-4, side R 1-4. Commands: collect_switch_hex.COND.

Windows: start = W0 + w * (66 + gap), w = 0..3, W0 = 10 frames (one gait cycle after the start), gap per condition
chosen from 8..30 frames to maximise the smallest circular distance between the 4 start gait phases.

Split rule (fixed): window w of condition index i -> role ROLES[(w + i) % 4], ROLES = (train k=0, train k=1, val,
heldout). Room seed: train 2*i + k, val 100 + i, heldout 200 + i.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
from collect_switch_hex import COND, KEYS, COMMON, CENTRE  # noqa: E402,F401  (re-exported)

EP = 66
W0 = 10
GAPS = range(8, 31)
BASE_CYC = 8.8
ROLES = ("train0", "train1", "val", "heldout")
LIVE_ROOM = 40.0     # physics walks: walls 20 m away so they can never be touched (no frames are taken)

ORDER = ([f"speed_c{c:g}" for c in (5.8, 7.1, 8.15, 8.8)]
         + [f"speed_c{c:g}_bwd" for c in (5.8, 7.1, 8.15, 8.8)]
         + [f"turn_s{v}" for v in ("0.05", "0.15", "0.29", "0.56")]
         + [f"turn_s{v}_neg" for v in ("0.05", "0.15", "0.29", "0.56")]
         + [f"side_L_lvl{i}" for i in range(4)] + [f"side_R_lvl{i}" for i in range(4)])
FAMILY = ["fwd"] * 4 + ["bwd"] * 4 + ["turn_left"] * 4 + ["turn_right"] * 4 + ["side_L"] * 4 + ["side_R"] * 4
assert sorted(ORDER) == sorted(COND) and len(ORDER) == 24


def seed_of(i, role):
    return {"train0": 2 * i, "train1": 2 * i + 1, "val": 100 + i, "heldout": 200 + i}[role]


def split_of(role):
    return "train" if role.startswith("train") else role


def cycles_per_frame(c):
    return COND[c][1]["pace"] * BASE_CYC / EP


def schedule(c):
    """(gap, starts, start phases, total frames) for condition c."""
    cpf = cycles_per_frame(c)
    best = None
    for g in GAPS:
        st = [W0 + w * (EP + g) for w in range(4)]
        ph = np.mod(np.array(st) * cpf, 1.0)
        d = np.abs(ph[:, None] - ph[None, :]); d = np.minimum(d, 1 - d)
        score = d[np.triu_indices(4, 1)].min()
        if best is None or score > best[0] + 1e-9:
            best = (score, g, st, ph)
    _, g, st, ph = best
    return g, st, ph, st[-1] + EP


def tilt_deg(q):
    """Angle of the body's own axes from their first-frame orientation, about a horizontal axis
    (yaw removed): angle between the body-frame image of world z now and at frame 0."""
    from scipy.spatial.transform import Rotation as Rt
    R = Rt.from_quat(q)                       # (x, y, z, w)
    up = R.inv().apply([0, 0, 1.0])           # world up in body coordinates
    return np.degrees(np.arccos(np.clip(up @ up[0], -1, 1)))
