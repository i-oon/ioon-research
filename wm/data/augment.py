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
    blur: float = 0.0          # Gaussian sigma in pixels
    saturation: float = 1.0    # colour saturation factor
    hue: float = 0.0           # hue rotation, degrees
    noise: float = 0.0         # pixel noise std, fraction of 255
    noise_seed: int = 0
    gain: float = 1.0          # multiplicative brightness (log-uniform, see STRENGTH['bright_mult'])


# Strength of the photometric / crop randomisation. The defaults are the values every run before
# 2026-09-30 used; the extra terms (blur, saturation, hue, noise) draw random numbers only when their
# strength is > 0, so a default run reproduces its crops and photometric draws exactly.
# Set from the config by `configure(cfg)` (wm/train.py). No rotation or flip: rotating the image
# rotates the optical flow, which would contradict the Froude (left / right, turn) labels.
# prob < 1: each type (crop, brightness/contrast, gain, blur, colour, noise) is applied independently with this
# probability, so part of every batch stays close to the clean frame (safer on small data than always-on strong
# randomisation). bright_mult > 0: multiplicative brightness gain drawn log-uniform in [1/m, m] (Egocentric VSM's
# x0.1-10 is m = 10). clean > 0: that fraction of samples gets no augmentation at all. The defaults
# (1.0, 0.0, 0.0) draw no extra random numbers.
STRENGTH = dict(min_scale=0.85, brightness=0.2, contrast=0.2, blur=0.0, saturation=0.0, hue=0.0, noise=0.0,
                prob=1.0, bright_mult=0.0, clean=0.0)


def configure(cfg):
    for k in STRENGTH:
        v = getattr(cfg, f"aug_{k}", None)
        if v is not None:
            STRENGTH[k] = float(v)


def sample_params(rng, height, width, min_scale=None):
    S = STRENGTH
    min_scale = S["min_scale"] if min_scale is None else min_scale
    if S["clean"] > 0 and rng.uniform() < S["clean"]:
        return identity_params(height, width)
    p = S["prob"]

    def on():                   # no draw at prob 1, so default runs reproduce exactly
        return True if p >= 1.0 else bool(rng.uniform() < p)
    full = min(height, width)
    size = int(round(full * rng.uniform(min_scale, 1.0))) if on() else full
    base = dict(
        y0=int(rng.integers(0, height - size + 1)),
        x0=int(rng.integers(0, width - size + 1)),
        size=size,
        brightness=0.0, contrast=1.0,
    )
    if on():
        base["brightness"] = float(rng.uniform(-S["brightness"], S["brightness"]))
        base["contrast"] = float(rng.uniform(1.0 - S["contrast"], 1.0 + S["contrast"]))
    if S["bright_mult"] > 0 and on():
        m = np.log(S["bright_mult"])
        base["gain"] = float(np.exp(rng.uniform(-m, m)))
    if S["blur"] > 0 and on():
        base["blur"] = float(rng.uniform(0.0, S["blur"]))
    if S["saturation"] > 0 and on():
        base["saturation"] = float(rng.uniform(1.0 - S["saturation"], 1.0 + S["saturation"]))
    if S["hue"] > 0 and on():
        base["hue"] = float(rng.uniform(-S["hue"], S["hue"]))
    if S["noise"] > 0 and on():
        base["noise"] = float(rng.uniform(0.0, S["noise"]))
        base["noise_seed"] = int(rng.integers(0, 2 ** 31 - 1))
    return AugmentParams(**base)


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
    if params.blur > 0.05:
        crop = cv2.GaussianBlur(crop, (0, 0), params.blur)
    if params.saturation != 1.0 or params.hue != 0.0:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + params.hue / 2.0) % 180.0          # OpenCV hue is 0..180
        hsv[..., 1] = np.clip(hsv[..., 1] * params.saturation, 0, 255)
        crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = crop.astype(np.float32)
    out = (out - 127.5) * params.contrast + 127.5 + params.brightness * 255.0
    if params.gain != 1.0:
        out = out * params.gain
    if params.noise > 0:
        out = out + np.random.default_rng(params.noise_seed).normal(0.0, params.noise * 255.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
