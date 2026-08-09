"""Cross-augmentation views (LAC-WM Section 3.1).

One sampled parameter set defines one augmentation A; it is applied to both frames of a
pair so the transition itself carries no augmentation difference. Two independent samples
give the two views the ITM and FTM consume.

Horizontal flip is deliberately excluded: mirroring the image swaps the robot's left and
right legs while the supervised action vector keeps its original leg order, which would
make the motion-decoding target inconsistent with the observation.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentParams:
    y0: int
    x0: int
    size: int
    brightness: float
    contrast: float


def sample_params(rng, height, width, min_scale=0.85):
    size = int(round(min(height, width) * rng.uniform(min_scale, 1.0)))
    return AugmentParams(
        y0=int(rng.integers(0, height - size + 1)),
        x0=int(rng.integers(0, width - size + 1)),
        size=size,
        brightness=float(rng.uniform(-0.2, 0.2)),
        contrast=float(rng.uniform(0.8, 1.2)),
    )


def identity_params(height, width):
    """The augmentation that changes nothing: full frame, no photometric shift.

    Used when cross-augmentation is switched off. It keeps every call site identical so the
    only difference between an augmented and an un-augmented run is what the encoder sees.
    """
    return AugmentParams(y0=0, x0=0, size=min(height, width), brightness=0.0, contrast=1.0)


def apply(frame, params):
    height, width = frame.shape[:2]
    crop = frame[params.y0:params.y0 + params.size, params.x0:params.x0 + params.size]
    if crop.shape[:2] != (height, width):
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    out = crop.astype(np.float32)
    out = (out - 127.5) * params.contrast + 127.5 + params.brightness * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
