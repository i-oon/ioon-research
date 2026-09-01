"""Which half of the augmentation produces the noise, crop or photometric jitter?

Crop moves the robot inside the image, which is the same kind of change the motion itself
produces, so it is both the larger nuisance and the one that overlaps the signal. Jitter changes
no geometry. Splitting them says which one can be reduced.
"""
import os, sys
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT); sys.path.insert(0, SCRIPTS)
from vjepa2_encoder import VJEPA2FrameEncoder
from wm.data.dataset import load_clip
from wm.data.augment import AugmentParams, apply, sample_params

frames = load_clip(f'{ROOT}/data/allocentric/fwd_hex8body/c10f10t10_ep6.npz')['frames'][:32]
H, W = frames.shape[1:3]
rng = np.random.default_rng(0)

def crop_only(r, min_scale):
    size = int(round(min(H, W) * r.uniform(min_scale, 1.0)))
    return AugmentParams(int(r.integers(0, H - size + 1)), int(r.integers(0, W - size + 1)),
                         size, 0.0, 1.0)
def jitter_only(r):
    return AugmentParams(0, 0, min(H, W), float(r.uniform(-0.2, 0.2)), float(r.uniform(0.8, 1.2)))

enc = VJEPA2FrameEncoder(device='cpu', dtype=torch.float32)
def emb(imgs):
    return torch.cat([enc.encode(list(imgs[i:i+4])).float() for i in range(0, len(imgs), 4)])
def pair(fn):
    r1, r2 = np.random.default_rng(1), np.random.default_rng(2)
    v1 = emb(np.stack([apply(f, fn(r1)) for f in frames]))
    v2 = emb(np.stack([apply(f, fn(r2)) for f in frames]))
    return float(((v1 - v2) ** 2).mean())

clean = emb(frames)
signal = float(((clean[:-1] - clean[1:]) ** 2).mean())
rows = [
    ('crop 85-100% + jitter (current)', pair(lambda r: sample_params(r, H, W, 0.85))),
    ('crop 85-100% only',               pair(lambda r: crop_only(r, 0.85))),
    ('crop 95-100% only',               pair(lambda r: crop_only(r, 0.95))),
    ('jitter only, no crop',            pair(jitter_only)),
    ('crop 95-100% + jitter',           pair(lambda r: sample_params(r, H, W, 0.95))),
]
del enc
print(f'{"augmentation":<34}{"noise":>9}{"noise/signal":>14}')
for tag, n in rows:
    print(f'{tag:<34}{n:>9.3f}{n/signal:>14.2f}')
print(f'\nsignal, consecutive clean frames: {signal:.3f}')
