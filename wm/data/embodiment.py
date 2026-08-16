"""Embodiment specifications.

Each embodiment stores clips in its own format and has its own action dimensionality.
The world model shares one visual encoder, ITM and FTM across all of them; only the motion
decoder needs an embodiment-specific output head, because 18-D hexapod joint targets and
12-D quadruped joint targets have no correspondence to share.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..bodies import CONTACT_THRESHOLD  # noqa: F401  single source of truth, do not redefine

G = 9.81
# One second, which is about one stride on either robot: the insect's is 0.95 s and the B1's is
# close to it. The window is given in seconds and converted per embodiment because they are
# recorded at different rates -- 20 Hz for the insect, 50 Hz for the B1.
BODY_WINDOW_S = 1.0


def body_motion(position, dt):
    """Forward and lateral speed over sqrt(g * h), smoothed across roughly one stride.

    Two choices carry this, and both were measured rather than assumed.

    **Froude, not m/s.** Dividing by sqrt(g * hip height) is what makes 0.18 m/s at 0.13 m and
    0.30 m/s at 0.56 m the same number. F56: the hexapod averages 0.155 and the B1 0.159 despite a
    four-fold size difference, so this is the level at which the two robots genuinely overlap.

    **Smoothed to a stride, not per frame.** Instantaneous forward speed is dominated by the body
    rocking with each step, which is a leg-level quantity with no cross-robot counterpart. Measured
    on the multi-speed insect set, between-clip speed variation sits at 0.63 of the within-clip
    rocking at a five-frame window and **1.45** at a stride-length window. Targeting the raw
    per-frame value would hand this head mostly rocking and almost no speed. Same lesson as F54's
    training window: the informative scale is the stride, not the timestep.
    """
    height = float(np.median(position[:, 2]))
    window = max(3, int(round(BODY_WINDOW_S / dt)))
    kernel = np.ones(window) / window
    out = []
    for axis in (0, 1):
        speed = np.gradient(position[:, axis].astype(np.float64), dt)
        out.append(np.convolve(speed, kernel, mode="same"))
    return (np.stack(out, axis=1) / np.sqrt(G * max(height, 1e-6))).astype(np.float32)


def _hexapod(data):
    return {
        "frames": data["frames"],
        "actions": data["actions"].astype(np.float32),
        "contact": (data["forces"].astype(np.float32) > CONTACT_THRESHOLD).astype(np.int64),
        "body_motion": body_motion(data["head"].astype(np.float64), HEXAPOD_DT),
        "group": int(data["expert_episode"]),
        "body": str(data["morph"]),
    }


def _b1(data):
    return {
        "frames": data["frames"],
        "actions": data["action"].astype(np.float32),
        "contact": (data["foot_contact"].astype(np.float32) > 0.5).astype(np.int64),
        "body_motion": body_motion(data["base_pos"].astype(np.float64), B1_DT),
        "group": 0,
        "body": "b1",
    }


@dataclass(frozen=True)
class Embodiment:
    name: str
    action_dim: int
    n_feet: int
    read: Callable


# Capture rates. The insect's comes from `sim_time` in expert_66k_aug3c_fcontact.csv; the B1's
# from DECIMATION 4 on the MuJoCo step its collector documents as 50 Hz. `body_motion` is a
# velocity, so these cannot be left implicit.
HEXAPOD_DT = 0.05
B1_DT = 0.02

HEXAPOD = Embodiment("hexapod", 18, 6, _hexapod)
B1 = Embodiment("b1", 12, 4, _b1)

REGISTRY = {e.name: e for e in (HEXAPOD, B1)}


def load(path, embodiment):
    with np.load(path, allow_pickle=True) as data:
        clip = embodiment.read(data)
    clip["embodiment"] = embodiment.name
    return clip
