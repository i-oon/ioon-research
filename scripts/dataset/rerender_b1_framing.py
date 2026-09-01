"""Re-render `beh12_b1_flat` with the camera that keeps the whole robot in shot.

**Two flags, both defects, both measured before this script existed (F113).** At the scene's
authored 15-degree angle the B1 touches an image edge in 62% of frames averaged over the twelve
conditions and in 100% of the sideways ones, while the insect never does in any of its 48 clips; at
25 degrees it is 0% everywhere. And the B1's camera was never pinned to a fixed world point, so its
background differs by 2.8-4.4 grey levels from clip to clip where the insect's is identical to 0.00
-- `--spawn` exists for exactly that and was not used.

**Three flags, because widening the view exposed the next thing.** At 25 degrees the camera reaches
the far edge of the scene's 15 m floor, which draws a straight band across the upper third of every
frame -- worst edge 21.3 grey levels against the insect's 4.4, and the thing that made the wide shot
look nothing like the insect's. Raising the lights does not touch it; only more floor does.
`--floor_scale 3` brings the background to sd 3.60 and worst edge 3.45 against the insect's 3.77 and
4.40, which is *closer to the insect than the original B1 render was*.

**MuJoCo does not run again.** Every clip already stores `base_pos`, `base_quat` and `joint_pos`,
so the physics is replayed from the file and only CoppeliaSim's camera changes.

**The renderer drops the behaviour labels**, keeping only proprioception and pose, so they are
copied back from the source clip here. Without this the re-rendered set has frames and actions and
no idea which condition each clip is.

  .venv/bin/python3 scripts/dataset/rerender_b1_framing.py --out data/allocentric/beh12_b1_fov25
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARRY = ("condition", "behaviour", "level", "expert_episode", "embodiment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/allocentric/beh12_b1_flat")
    ap.add_argument("--out", default="data/allocentric/beh12_b1_fov25")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--cam_fov", type=float, default=24.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--floor_scale", type=float, default=3.0)
    ap.add_argument("--ego", action="store_true",
                    help="head camera and a randomised room instead of the fixed third-person shot")
    ap.add_argument("--ego_box", type=float, default=0.0)
    ap.add_argument("--ego_offset", type=float, nargs=3, default=None, metavar=("R", "U", "F"))
    ap.add_argument("--align_yaw", action="store_true",
                    help="rotate each clip to start facing +x, so a slot pairs with the insect's")
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    clips = sorted(glob.glob(os.path.join(src, "*.npz")))
    if args.ego and args.cam_fov == 24.0:
        # **90 degrees, not the third-person 24.** An egocentric clip shot through the allocentric
        # lens is not the experiment, and the default here is the allocentric one.
        args.cam_fov = 90.0
    print(f"{len(clips)} clips, view {'egocentric' if args.ego else 'allocentric'}, "
          f"fov {args.cam_fov} deg, spawn {tuple(args.spawn)}, floor x{args.floor_scale}", flush=True)

    for i, clip in enumerate(clips):
        tag = os.path.basename(clip)
        cmd = [os.path.join(ROOT, ".venv/bin/python3"),
               os.path.join(ROOT, "sim/render/render_b1_replay.py"),
               "--scene", args.scene, "--traj", clip, "--out", out,
               "--cam_fov", str(args.cam_fov),
               "--spawn", str(args.spawn[0]), str(args.spawn[1]),
               "--floor_scale", str(args.floor_scale)]
        if args.ego:
            # **The room is the clip's index within its condition, and that is the whole pairing
            # scheme.** `expert_episode` is `axis*1000 + cond*100 + clip`, so `% 100` is the clip
            # index -- 0 to 3 -- and the insect derives the same number from its repeat counter.
            # Same slot on either robot therefore means the same room, and Q2 differs by body alone.
            with np.load(clip, allow_pickle=True) as z:
                room = int(z["expert_episode"]) % 100 if "expert_episode" in z.files else i % 4
            cmd += ["--ego", "--ego_box", str(args.ego_box), "--ego_seed", str(room)]
            if args.align_yaw:
                cmd += ["--align_yaw"]
            if args.ego_offset:
                cmd += ["--ego_offset"] + [str(v) for v in args.ego_offset]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  FAILED {tag}\n{r.stdout[-400:]}{r.stderr[-800:]}", flush=True)
            sys.exit(1)
        # the labels the renderer does not carry
        with np.load(clip, allow_pickle=True) as a, np.load(os.path.join(out, tag), allow_pickle=True) as b:
            merged = {k: b[k] for k in b.files}
            merged.update({k: a[k] for k in CARRY if k in a.files})
        np.savez_compressed(os.path.join(out, tag), **merged)
        if (i + 1) % 6 == 0 or i + 1 == len(clips):
            print(f"  {i + 1}/{len(clips)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
