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


def _hexapod(data):
    return {
        "frames": data["frames"],
        "actions": data["actions"].astype(np.float32),
        "contact": (data["forces"].astype(np.float32) > CONTACT_THRESHOLD).astype(np.int64),
        "group": int(data["expert_episode"]),
        "body": str(data["morph"]),
    }


def _b1(data):
    return {
        "frames": data["frames"],
        "actions": data["action"].astype(np.float32),
        "contact": (data["foot_contact"].astype(np.float32) > 0.5).astype(np.int64),
        "group": 0,
        "body": "b1",
    }


@dataclass(frozen=True)
class Embodiment:
    name: str
    action_dim: int
    n_feet: int
    read: Callable


HEXAPOD = Embodiment("hexapod", 18, 6, _hexapod)
B1 = Embodiment("b1", 12, 4, _b1)

REGISTRY = {e.name: e for e in (HEXAPOD, B1)}


def load(path, embodiment):
    with np.load(path, allow_pickle=True) as data:
        clip = embodiment.read(data)
    clip["embodiment"] = embodiment.name
    return clip
