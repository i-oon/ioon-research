"""Re-render the B1 egocentric datasets with rendering matched to the hexapod's (2026-09-30, F275 addendum).

The B1 ego clips were rendered with an older room-texture recipe and a floor stretched by a fixed x3,
while the hexapod collector stretches the floor with the room (`ego_camera.scale_floor`). The frames of
the two pipelines differ in texture statistics (V-JEPA2 separates them from one frame at 0.998, even
from the ceiling alone). This re-renders every B1 ego clip from its stored MuJoCo states (kinematic
replay: deterministic, re-rendering a clip twice gives identical frames) with
    render_b1_replay.py --ego --match_floor --ground_uv_mult 1.0
i.e. pure scaling with the room. A static test (both scenes rendered from the same camera poses, no body;
static_scene_render_check.py) gives a hexapod-vs-B1 frame probe of 0.552 with x1.0 and 0.952 with x1.5:
x1.0 is the matched setting. (v2 used x1.5, tuned against walking data whose differences are camera motion.)

Only `frames` changes: every other field is copied from the original clip. Output mirrors the input
under data/egocentric_v3/, split directories as relative symlinks, originals untouched.

    .venv/bin/python3 scripts/dataset/rerender_b1_ego_matched.py
Needs CoppeliaSim on port 23000.
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = ("beh24_b1_ego_flat", "beh12_b1_ego_flat")
SPLITS = ("beh24_b1_ego_flat_cleantrain", "beh24_b1_ego_flat_cleanval", "beh24_b1_ego_flat_cleanheldout",
          "beh12_b1_ego_flat_cleantrain")
FLAGS = ["--ego", "--match_floor", "--ground_uv_mult", "1.0"]


def main():
    tmp = tempfile.mkdtemp(prefix="b1rr_")
    for d in SRC:
        out_dir = os.path.join(ROOT, "data/egocentric_v3", d)
        os.makedirs(out_dir, exist_ok=True)
        paths = sorted(p for p in glob.glob(os.path.join(ROOT, "data/egocentric", d, "*.npz")) if not os.path.islink(p))
        for i, p in enumerate(paths):
            dst = os.path.join(out_dir, os.path.basename(p))
            if os.path.exists(dst):
                continue
            work = os.path.join(tmp, f"{d}_{i}")
            subprocess.run([sys.executable, os.path.join(ROOT, "sim/render/render_b1_replay.py"),
                            "--scene", "sim/env/b1_flat.ttt", "--traj", p, "--out", work,
                            "--ego_seed", str(i)] + FLAGS, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            new = np.load(glob.glob(os.path.join(work, "*.npz"))[0], allow_pickle=True)["frames"]
            with np.load(p, allow_pickle=True) as f:
                data = {k: f[k] for k in f.files}
            if len(new) != len(data["frames"]):
                raise SystemExit(f"{p}: re-render gave {len(new)} frames, original {len(data['frames'])}")
            data["frames"] = new
            data["render_note"] = np.array("rerender_b1_ego_matched: match_floor, ground_uv x1.0, seed %d" % i)
            np.savez_compressed(dst, **data)
            print(f"{d} {i + 1}/{len(paths)} {os.path.basename(p)}", flush=True)
    for sd in SPLITS:
        src = os.path.join(ROOT, "data/egocentric", sd)
        out = os.path.join(ROOT, "data/egocentric_v3", sd)
        os.makedirs(out, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
            real = os.path.realpath(p)
            target = os.path.join("..", os.path.basename(os.path.dirname(real)), os.path.basename(real))
            link = os.path.join(out, os.path.basename(p))
            if not os.path.lexists(link):
                os.symlink(target, link)
        print(f"split {sd}: {len(os.listdir(out))} links")
    print("RERENDER_DONE")


if __name__ == "__main__":
    main()
