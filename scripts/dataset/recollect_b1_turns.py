"""Re-collect the B1's four turn levels so they match the insect in sign and in strength.

**Two defects, one rollout.** `turn_wz0.00` is the forward clip under another name -- +0.1259
forward and +0.0008 yaw, identical to `speed_vx0.30` in both channels (F114) -- and the whole family
turns the *opposite way* from the insect, which F75 diagnosed on 2026-08-22 and which was recorded
as fixed without being applied (F115). Every cross-embodiment turning result compares a left turn
with a right turn.

**Matched on what the robots achieve, not on what they are told.** F72 paired the two sides on
commanded turn rate and reported agreement within 3%; measured, only the third of four pairs
matches. Sweeping `--wz` at `--vx 0.30` gives a linear response, and these are the commands that
land on the insect's achieved yaw:

    insect turn_s0.05  -0.0072  ->  --wz -0.064
    insect turn_s0.15  -0.0241  ->  --wz -0.153
    insect turn_s0.29  -0.0372  ->  --wz -0.223
    insect turn_s0.56  -0.0878  ->  --wz -0.491

**The weakest level stays weak on purpose.** It is nearly straight walking, and so is the insect's
`turn_s0.05`; matching the insect matters more than being easy to tell apart from forward, because
the pretrained model already learned the insect's version.

**Four clips per condition come from four windows of one rollout**, because MuJoCo starts every run
at the same state and is deterministic, so four runs of one command are the same clip four times.
The windows do not overlap. The insect's four come from four separate CoppeliaSim episodes, which
differ because that engine does not repeat (F105); this is the closest available equivalent.

  .venv/bin/python3 scripts/dataset/recollect_b1_turns.py --out data/beh12_b1_turns
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# name, commanded wz, level, first episode id -- ids continue the existing scheme (F114's table)
# **The condition is a shared target, and each policy needs its own command to hit it.** F80 keeps
# both B1 policies, two clips each per condition, because every clip of one policy is the same limit
# cycle at a different phase -- four clips of one policy say one thing four times. The two need
# different commands for the same result: at the insect's weakest turn `gait3` needs -0.061 and
# `sym` -0.023, a factor of two apart. Conditions are therefore named by the **achieved** rate both
# robots share, not by either command.
#
# name, level, first episode id, {policy: commanded wz}
SYM = "sim/assets/b1_policy/base_1.7hz_sym/model_600.pt"
# **Named for what the B1 achieves, not for the insect level it is paired with.** Three of the four
# land within 1% of their target; the strongest does not. Pushing `--wz` from -0.475 to -0.664 moved
# yaw only -0.063 to -0.075 while forward Froude fell 0.117 to 0.097, so the quadruped buys rotation
# with forward speed and cannot reach the insect's 0.0878 while still walking at 0.13. **The
# insect's hardest turn has no true B1 counterpart**, and naming the condition 0.088 would have
# hidden that.
#
# Calibrated on the finished clips, not on the rollout: the sweep measures yaw at the rollout's
# 50 Hz and the clips are stored at 20 Hz, and the median of the downsampled series is not the
# median of the full one -- fitting on the wrong rate left the strongest level 27% short.
LEVELS = (("turn_w0.008", 0, 1000, {"sym": -0.023, "gait3": -0.081}),
          ("turn_w0.024", 1, 1100, {"sym": -0.115, "gait3": -0.169}),
          ("turn_w0.037", 2, 1200, {"sym": -0.191, "gait3": -0.252}),
          ("turn_w0.075", 3, 1300, {"sym": -0.587, "gait3": -0.664}))
# two clips per policy, in the order the existing set uses: sym first, then gait3 (F80)
POLICIES = (("sym", SYM, 1.7), ("gait3", "", 2.0))
CLIPS_PER_POLICY, FRAMES, FPS = 2, 66, 20.0


def _face_forward(pos, quat):
    """Rotate a path about its own first point so it starts at yaw 0."""
    pos, quat = np.asarray(pos, float).copy(), np.asarray(quat, float).copy()
    w, x, y, z = quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = pos[:, :2] - pos[0, :2]
    pos[:, 0], pos[:, 1] = pos[0, 0] + c * d[:, 0] - s * d[:, 1], pos[0, 1] + s * d[:, 0] + c * d[:, 1]
    rw, rz = np.cos(-yaw / 2), np.sin(-yaw / 2)          # rotation about world z, as (w,x,y,z)
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    quat[:, 0] = rw * qw - rz * qz
    quat[:, 1] = rw * qx - rz * qy
    quat[:, 2] = rw * qy + rz * qx
    quat[:, 3] = rw * qz + rz * qw
    return pos, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/beh12_b1_turns")
    ap.add_argument("--vx", type=float, default=0.30)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--cam_fov", type=float, default=24.0)
    ap.add_argument("--floor_scale", type=float, default=3.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_traj")
    os.makedirs(tmp, exist_ok=True)
    py = os.path.join(ROOT, ".venv/bin/python3")
    # 66 frames at 20 Hz is 3.3 s; four back-to-back windows need 13.2 s, and the rollout runs at
    # 50 Hz, so 660 steps plus a margin for the policy warm-up the script does before logging
    steps = int(CLIPS_PER_POLICY * FRAMES * (50.0 / FPS)) + 60

    for name, level, ep0, cmds in LEVELS:
        k = 0
        for pol, ckpt, gait in POLICIES:
            traj = os.path.join(tmp, f"{name}_{pol}.npz")
            cmd = [py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"),
                   "--vx", str(args.vx), "--wz", str(cmds[pol]),
                   "--steps", str(steps), "--gait_freq", str(gait), "--out", traj]
            if ckpt:
                cmd += ["--checkpoint", os.path.join(ROOT, ckpt)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"  rollout FAILED {name}/{pol}\n{r.stdout[-300:]}{r.stderr[-800:]}"); sys.exit(1)
            with np.load(traj, allow_pickle=True) as T:
                data = {kk: T[kk] for kk in T.files}
            n = len(data["base_pos"])
            per = int(FRAMES * (50.0 / FPS))
            for c in range(CLIPS_PER_POLICY):
                sl = slice(c * per, (c + 1) * per)
                piece = os.path.join(tmp, f"{name}_{pol}_c{c}.npz")
                cut = {kk: (v[sl] if getattr(v, "ndim", 0) and len(v) == n else v)
                       for kk, v in data.items()}
                cut["base_pos"], cut["base_quat"] = _face_forward(cut["base_pos"], cut["base_quat"])
                np.savez_compressed(piece, **cut)
                tag = f"b1_ep{ep0 + k}"
                r = subprocess.run([py, os.path.join(ROOT, "sim/render/render_b1_replay.py"),
                                    "--scene", args.scene, "--traj", piece, "--out", out,
                                    "--fps", str(FPS), "--cam_fov", str(args.cam_fov),
                                    "--floor_scale", str(args.floor_scale),
                                    "--spawn", str(args.spawn[0]), str(args.spawn[1])],
                                   capture_output=True, text=True, timeout=1800)
                if r.returncode != 0:
                    print(f"  render FAILED {tag}\n{r.stdout[-300:]}{r.stderr[-800:]}"); sys.exit(1)
                src = os.path.join(out, os.path.basename(piece))
                with np.load(src, allow_pickle=True) as b:
                    merged = {kk: b[kk] for kk in b.files}
                merged.update(condition=np.array(name), behaviour=np.array("turn"),
                              level=np.array(level), expert_episode=np.array(ep0 + k),
                              embodiment=np.array("b1"), policy=np.array(pol))
                np.savez_compressed(os.path.join(out, tag + ".npz"), **merged)
                os.remove(src)
                print(f"  {tag}  {name}  {pol}  wz {cmds[pol]:+.3f}", flush=True)
                k += 1
    print("done", flush=True)


if __name__ == "__main__":
    main()
