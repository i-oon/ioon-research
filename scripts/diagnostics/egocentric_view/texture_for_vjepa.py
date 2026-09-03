"""Which wall texture does V-JEPA2 actually read motion from?

    .venv/bin/python3 scripts/diagnostics/egocentric_view/texture_for_vjepa.py

**Asked because the egocentric room reused the floor's texture recipe on faith.** That recipe was
chosen for a *floor*, seen from a *third-person* camera, to stop the background drowning the robot
(`set_floor_texture.py`). **The egocentric wall has the opposite job**: it is not background, it is
the entire signal -- the only thing a head camera can tell the model is how the world slid past.

So the property to measure is not "does it look nice", it is **does embedding change track camera
motion**. A surface fails in either direction: a blank one gives no change when the camera moves,
and a high-frequency one gives change when it does not, because sub-pixel motion aliases.

**No simulator involved.** A large texture is panned past a 256x256 viewport, which is exactly what
a wall does when a robot walks along it, and the frames go through the same encoder the pipeline
uses. A still sequence gives the noise floor.

Reported per style:

    slope        embedding distance per pixel of true pan -- the signal
    r            correlation between true displacement and embedding displacement
    still        embedding drift with the camera stopped -- the floor
    signal/still the number that matters: motion has to move `e` more than standing does
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

N, VIEW, T = 1024, 256, 12


def value_noise(size, seed, octaves=((4, .5), (8, .25), (16, .15), (64, .07), (256, .03))):
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size))
    for o, w in octaves:
        small = rng.random((o, o))
        idx = np.linspace(0, o - 1, size)
        i0 = np.floor(idx).astype(int); i1 = np.minimum(i0 + 1, o - 1); f = idx - i0
        top = small[np.ix_(i0, i0)] * (1 - f) + small[np.ix_(i0, i1)] * f
        bot = small[np.ix_(i1, i0)] * (1 - f) + small[np.ix_(i1, i1)] * f
        acc += w * (top * (1 - f[:, None]) + bot * f[:, None])
    return (acc - acc.min()) / (acc.max() - acc.min())


def styles():
    out = {}
    out["blank"] = np.full((N, N), 0.5)
    out["white noise"] = np.random.default_rng(0).random((N, N))
    c = np.indices((N, N)).sum(0) // 32 % 2
    out["checkerboard"] = c.astype(float)
    out["value noise (floor recipe)"] = value_noise(N, 1)
    out["value noise, coarse only"] = value_noise(N, 1, ((4, .5), (8, .3), (16, .2)))
    out["value noise, fine heavy"] = value_noise(N, 1, ((4, .2), (16, .2), (64, .3), (256, .3)))
    g = value_noise(N, 2)
    out["value noise + stripes"] = np.clip(0.7 * g + 0.3 * (np.sin(np.arange(N) / 18.0)[None, :]
                                                            * 0.5 + 0.5), 0, 1)
    return {k: (np.stack([v] * 3, -1) * 255).astype(np.uint8) for k, v in out.items()}


def pan(tex, dx, n=T):
    return np.stack([tex[100:100 + VIEW, 100 + int(round(i * dx)):100 + int(round(i * dx)) + VIEW]
                     for i in range(n)])


def main():
    enc = VJEPA2FrameEncoder(dtype=torch.float32)
    speeds = [2, 4, 8]
    print(f"{VIEW}px viewport panned across a {N}px surface, {T} frames, "
          f"speeds {speeds} px/frame\n")
    print(f"  {'style':>30}{'slope':>10}{'r':>8}{'still':>10}{'signal/still':>14}   verdict")
    for name, tex in styles().items():
        d_true, d_emb = [], []
        for dx in speeds:
            e = encode_clip(enc, pan(tex, dx), 2).float()
            for t in range(1, T):
                d_true.append(t * dx)
                d_emb.append(float(torch.norm(e[t].flatten() - e[0].flatten())))
        still = encode_clip(enc, pan(tex, 0), 2).float()
        floor = float(np.mean([float(torch.norm(still[t].flatten() - still[0].flatten()))
                               for t in range(1, T)]))
        d_true, d_emb = np.array(d_true, float), np.array(d_emb, float)
        slope = float(np.polyfit(d_true, d_emb, 1)[0])
        r = float(np.corrcoef(d_true, d_emb)[0, 1])
        ratio = float(np.mean(d_emb)) / max(floor, 1e-9)
        good = r > 0.8 and ratio > 3.0
        print(f"  {name:>30}{slope:>10.2f}{r:>8.3f}{floor:>10.1f}{ratio:>14.2f}   "
              + ("ok" if good else ("weak r" if r <= 0.8 else "low margin")))
    print("\n  **`r` is the property the egocentric view depends on.** A surface whose embedding "
          "distance\n  does not grow with real motion cannot tell the model how far it travelled, "
          "however\n  textured it looks.")


if __name__ == "__main__":
    main()
