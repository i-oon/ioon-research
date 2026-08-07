"""Replace the scene's checkerboard floor with a V-JEPA2-friendly surface.

WHY THIS EXISTS — this is not cosmetic. Two empirical findings drive it
(PROGRESS.md §4, direction_plan.md "Step 0 Check 3 ABANDONED"):

  * CHECKERBOARD floors are actively harmful: high-contrast repeating edges
    alias under sub-pixel motion, so pixels change where nothing actually moved.
    Measured: correlation(pixel motion, embedding change) r = -0.16, p = 7.6e-24.
    The stock Medauroidea scene ships with exactly this floor.
  * BLANK / FLAT floors are also harmful, for the opposite reason: featureless
    patches carry no information, and ViTs repurpose such tokens as internal
    scratch space, so their embeddings fluctuate MORE than the moving robot's.
    Measured: r = -0.20, p = 4.7e-37.

So the target is the middle: matte, mildly textured, non-repeating at the scale
the camera actually sees. This script generates that procedurally (fixed seed →
byte-identical across every morphology variant, which is what the Step 1.5
render-lock requires) and applies it to /Floor.

Usage (CoppeliaSim must be running):
  python sim/set_floor_texture.py --scene sim/env/medauroidea_stick_insect.ttt --preview
  python sim/set_floor_texture.py --all
"""
import argparse
import os

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_DIR = os.path.join(ROOT, "sim", "env")
VARIANTS = [
    "medauroidea_stick_insect.ttt",
    "medauroidea_stick_insect_medium.ttt",
    "medauroidea_stick_insect_short.ttt",
]
TEXTURE_PATH = os.path.join(ENV_DIR, "floor_texture.png")
TEXTURE_SIZE = 1024
SEED = 0

# Floor appearance: mid-grey, matte, low contrast.
BASE_GREY = 130
CONTRAST = 26      # +/- range of the noise. Low = matte, no hard edges to alias.
UV_SCALE = 6.0     # metres per texture tile. Large => no visible repeat in frame.


def make_texture(size=TEXTURE_SIZE, seed=SEED):
    """Multi-octave value noise -> concrete-ish matte surface.

    Deliberately: no sharp edges (nothing to alias), no flat regions (nothing
    for ViT register tokens to colonise), low contrast (won't dominate the
    robot), fixed seed (identical every run and every morphology).
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), dtype=np.float64)
    # octaves: coarse blotches -> fine grain
    for octave, weight in [(4, 0.5), (8, 0.25), (16, 0.15), (64, 0.07), (256, 0.03)]:
        small = rng.random((octave, octave))
        # bilinear upsample to full size
        ys = np.linspace(0, octave - 1, size)
        xs = np.linspace(0, octave - 1, size)
        y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, octave - 1)
        x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, octave - 1)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        top = small[np.ix_(y0, x0)] * (1 - fx) + small[np.ix_(y0, x1)] * fx
        bot = small[np.ix_(y1, x0)] * (1 - fx) + small[np.ix_(y1, x1)] * fx
        acc += weight * (top * (1 - fy) + bot * fy)

    acc = (acc - acc.min()) / (acc.max() - acc.min())      # -> [0,1]
    img = BASE_GREY + (acc - 0.5) * 2 * CONTRAST
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([img] * 3, axis=-1)                     # greyscale -> RGB


def apply_to_floor(sim, scene_path, preview_dir=None):
    scene_path = os.path.abspath(scene_path)
    sim.loadScene(scene_path)

    floor = sim.getObject("/Floor")

    # /Floor is usually a compound; texture must go on its renderable child shapes
    targets = [floor]
    try:
        kids = sim.getObjectsInTree(floor, sim.object_shape_type)
        if kids:
            targets = kids
    except Exception:
        pass

    shape, tex_id, res = sim.createTexture(TEXTURE_PATH, 0)
    print(f"  texture id={tex_id} res={res}")

    for t in targets:
        # mappingMode 4 = cubic; options bit2(4)+bit3(8) = repeat along U and V
        sim.setShapeTexture(t, tex_id, sim.texturemap_cube, 4 | 8, [UV_SCALE, UV_SCALE])
    print(f"  applied to {len(targets)} floor shape(s)")

    sim.removeObjects([shape])  # createTexture makes a temp carrier plane; drop it

    sim.saveScene(scene_path)
    print(f"  saved: {scene_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-dir", type=str, default="/tmp")
    ap.add_argument("--texture-only", action="store_true", help="just write the PNG, touch no scene")
    args = ap.parse_args()

    tex = make_texture()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(TEXTURE_PATH, tex)
    print(f"texture written: {TEXTURE_PATH}  ({tex.shape}, mean={tex.mean():.1f}, "
          f"std={tex.std():.1f}, min={tex.min()}, max={tex.max()})")
    if args.texture_only:
        return

    client = RemoteAPIClient("localhost", port=23000)
    sim = client.require("sim")

    scenes = [os.path.join(ENV_DIR, v) for v in VARIANTS] if args.all else [args.scene]
    if scenes == [None]:
        ap.error("give --scene or --all")

    for s in scenes:
        print(f"\n=== {os.path.basename(s)} ===")
        apply_to_floor(sim, s, args.preview_dir if args.preview else None)


if __name__ == "__main__":
    main()
