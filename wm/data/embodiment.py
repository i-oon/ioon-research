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


# Which columns of `body_motion` are safe to supervise on. **Forward only.**
#
# The lateral channel looked like a free extra dimension and is an embodiment label in disguise:
# the insect drifts +0.021 and the B1 -0.025, opposite signs, because each gait has its own
# asymmetry. A single-feature classifier separates the robots at AUC 0.788 from that column alone,
# against 0.543 from forward speed. Training `z` to predict it is training `z` to encode which
# robot it is looking at, which is the failure this whole term exists to remove -- and it showed
# up immediately: the embodiment probe hit 0.824 at epoch 1 against the control's 0.537.
#
# Forward speed is what both robots are *doing*. Lateral drift is what each one happens to do
# wrong, separately.
# **Yaw is column 2** and is a candidate, not a default. It is the channel `data/beh12_*` was built
# to create: both robots now turn, matched to within 10% on dimensionless rate with forward speed
# held apart from it. Untrained it transfers at +0.10 +/- 0.19, which is zero -- but forward speed
# untrained is 0.31 against 0.90 trained (F66), so the untrained number does not settle it (F77).
# Set `cfg.body_channels` to (0, 2) to supervise it.
BODY_CHANNELS = (0,)


def _hexapod(data):
    dt = _dt_of(data, HEXAPOD_DT)
    position = data["head"].astype(np.float64)
    if "body_quat" in data.files:
        motion = np.concatenate(
            [body_velocity(position, data["body_quat"], dt, "hexapod"),
             yaw_rate(data["body_quat"], dt, "hexapod",
                      float(np.median(position[:, 2])))], axis=1)
    else:
        motion = body_motion(position, dt)      # pre-2026-08-22 clips carry no orientation
    return {
        "frames": data["frames"],
        "actions": data["actions"].astype(np.float32),
        "contact": (data["forces"].astype(np.float32) > CONTACT_THRESHOLD).astype(np.int64),
        "body_motion": motion,
        "group": int(data["expert_episode"]),
        "body": str(data["morph"]),
    }


def _b1(data):
    dt = _dt_of(data, B1_DT)
    position = data["base_pos"].astype(np.float64)
    if "base_quat" in data.files:
        motion = np.concatenate(
            [body_velocity(position, data["base_quat"], dt, "b1"),
             yaw_rate(data["base_quat"], dt, "b1",
                      float(np.median(position[:, 2])))], axis=1)
    else:
        motion = body_motion(position, dt)
    return {
        "frames": data["frames"],
        "actions": data["action"].astype(np.float32),
        "contact": (data["foot_contact"].astype(np.float32) > 0.5).astype(np.int64),
        "body_motion": motion,
        # the condition, so a cross-embodiment pairing can match behaviours rather than filenames
        "group": int(data["expert_episode"]) if "expert_episode" in data.files else 0,
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
#
# **These are fallbacks now, not the source of truth.** Clips written after 2026-08-22 carry their
# own `dt`, and `_dt_of` prefers it. F74: the B1 replay rendered one frame per 50 Hz rollout step
# while the insect records at 20 Hz, so a stored transition meant 20 ms on one robot and 50 ms on
# the other -- and `B1_DT = 0.02` was correct for the old clips and is wrong for the new ones. A
# constant that has to change when the data changes is a constant in the wrong place.
HEXAPOD_DT = 0.05
B1_DT = 0.02

# **The yaw target is scaled by body height, and that is a choice made against an argument, not by
# default.** Physically the moment arm of a turn is where the feet meet the ground, not the hip
# height -- measured from each model the B1 is 3.19x taller while the hexapod's stance is 1.39x
# wider (radius 0.576 m against 0.414 m), so the two candidate scales differ by 4.4x in the ratio
# between the robots, and by the embodiment gate stance radius looks better (AUC 0.571 against
# 0.637).
#
# **It is still wrong to use here, because the collection is matched on the height version.** The
# `--spin` levels in `data/beh12_hex` were solved so that w_hat = omega sqrt(h/g) lands on the B1's
# (F72), and under stance-radius scaling the same matched pair reads **0.130 against 0.066** -- a
# factor of two for two behaviours that are supposed to be the same. A shared head handed that can
# only fit both by learning which robot it is looking at, which is the exact shortcut this term
# exists to remove.
#
# The scale and the collection have to agree. Changing to stance radius means re-solving the four
# `--spin` levels against it first; until then this stays on height.
STANCE_RADIUS = {"hexapod": 0.576, "b1": 0.414}


def _dt_of(data, fallback):
    return float(data["dt"]) if "dt" in data.files else fallback


def forward_axis(quat, embodiment):
    """Unit vector, in the world floor plane, of the direction the robot actually travels.

    **Neither robot's convention is guessable and both have caught this project out.** The hexapod's
    `body_quat` is (x, y, z, w) off /abdomen whose fore-aft axis is **z pointing aft** -- its dot
    product with the direction straight walking actually travels is **-0.96**, so the axis has to be
    negated. The B1's `base_quat` is MuJoCo (w, x, y, z) with base x forward and no correction.
    Verified against straight walking rather than read off an axis name (F71 swapped left and right
    by trusting the name; F75 hid a sign inside a magnitude).
    """
    q = np.asarray(quat, dtype=np.float64)
    if embodiment == "hexapod":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx, fy = -2 * (x * z + w * y), -2 * (y * z - w * x)
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx, fy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
    v = np.stack([fx, fy], axis=1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def body_velocity(position, quat, dt, embodiment):
    """Forward and lateral speed **in the robot's own frame**, dimensionless, stride-averaged.

    **The world frame is the wrong frame and it is not a small error.** `body_motion` differenced
    world x and y, so its "forward" channel was walking speed multiplied by how much the robot still
    happened to point along world x. Straight walking hides this -- both robots start along +x -- but
    a turning robot rotates out of it: the hexapod's world-x speed falls **0.132 to 0.026** across
    the four turn levels while its actual walking speed is flat, **0.135 to 0.128**. The B1 does the
    same, 0.131 to 0.102 against a flat 0.132. Supervised on that, the shared head would be taught
    that turning means slowing down, by different amounts on the two robots -- which is a difference
    between the robots' turn rates wearing the label "speed".
    """
    height = float(np.median(position[:, 2]))
    scale = np.sqrt(G * max(height, 1e-6))
    v = np.gradient(position[:, :2].astype(np.float64), dt, axis=0)
    f = forward_axis(quat, embodiment)
    left = np.stack([-f[:, 1], f[:, 0]], axis=1)
    out = np.stack([(v * f).sum(1), (v * left).sum(1)], axis=1) / scale
    window = max(3, int(round(BODY_WINDOW_S / dt)))
    k = np.ones(window) / window
    return np.stack([np.convolve(out[:, c], k, mode="same") for c in (0, 1)], axis=1).astype(np.float32)


def heading(quat, embodiment):
    """Absolute heading angle per frame, in radians, with each robot's own convention.

    **Factored out of `yaw_rate` so there is exactly one place these formulas live.** The hexapod's
    `body_quat` is (x, y, z, w) off an **aft-pointing** abdomen axis and the B1's `base_quat` is
    MuJoCo's (w, x, y, z) with base x forward; hand-rolling either has already cost this project a
    week (F71, F117). `yaw_rate` differences this, so the aft-pointing axis cancels there as a
    constant -- **anything using the absolute angle must only ever compare two headings of the same
    robot**, never a heading against zero.
    """
    q = np.asarray(quat, dtype=np.float64)
    if embodiment == "hexapod":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        fx, fy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
    return np.arctan2(fy, fx)


def yaw_rate(quat, dt, embodiment, height):
    """Dimensionless turn rate, smoothed over the same window as `body_motion`.

    **The two robots store orientation differently and neither convention is guessable.** The
    hexapod's `body_quat` is (x, y, z, w) off /abdomen, whose **z axis points aft** -- F71 read left
    and right swapped by taking it as forward. The B1's `base_quat` is MuJoCo's (w, x, y, z) with
    base x forward. Only differences are used, so the aft-pointing axis cancels as a constant.

    F75: this must be **signed**. The pairing in F72 was built on |w_hat|, which hid that the two
    robots were turning opposite ways -- and in signed data that made yaw separate the robots at
    AUC 0.871, the exact failure the embodiment gate exists to catch.
    """
    omega = np.gradient(np.unwrap(heading(quat, embodiment)), dt)
    window = max(3, int(round(BODY_WINDOW_S / dt)))
    omega = np.convolve(omega, np.ones(window) / window, mode="same")
    return (omega * np.sqrt(max(height, 1e-6) / G)).astype(np.float32)[:, None]

HEXAPOD = Embodiment("hexapod", 18, 6, _hexapod)
B1 = Embodiment("b1", 12, 4, _b1)

REGISTRY = {e.name: e for e in (HEXAPOD, B1)}


def load(path, embodiment):
    with np.load(path, allow_pickle=True) as data:
        clip = embodiment.read(data)
    clip["embodiment"] = embodiment.name
    return clip
