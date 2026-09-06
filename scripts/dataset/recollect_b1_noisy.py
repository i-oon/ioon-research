"""More B1 clips for the conditions duration-extension broke, via noise instead of length.

    .venv/bin/python3 scripts/dataset/recollect_b1_noisy.py --port 23002

**Why this script and not more of `recollect_b1_more.py`.** That script got 5x more clips for
7 of 12 conditions safely (longer rollout, more windows) but `turn`x4 and `side_R_lvl1` went
unstable partway through the longer duration -- confirmed by height dropping well below the
walking threshold, not a judgment call. This collects those 5 conditions differently: many SHORT
rollouts (the original, proven-safe 2-window duration, `steps=390`), each seeded with different
correlated noise (`rollout_b1_mujoco.py --cmd_noise`, added post-action-scale to the joint target,
same shape and injection point as the hexapod's own `--cmd_noise`) so repeated short rollouts of
the same nominal command are no longer bit-identical.

**The noise level, 0.0137, and why it is not 0.02 (hexapod's own number).** Matched by proportion,
not copied: hexapod's proven level is 12.4% of its own joint-target std (0.161); B1's joint-target
std is 0.110 (0.25 x the raw policy output's 0.441 -- ACTION_SCALE, not the raw output itself,
which is the mistake an earlier pass in this session made and corrected before using it). Verified
before trusting it: `turn`x4 stays upright at both policies at this level; `side_R_lvl1`'s `gait3`
policy does not (height collapses to 0.094, well under the 0.35 fall threshold already built into
`rollout_b1_mujoco.py`) even though it is not walking any LONGER than its already-proven-safe
duration -- that policy was already the most fragile of the ten combinations (0/10 safe windows in
the duration test with zero added noise), and this is not a level to push it at. `side_R_lvl1` is
therefore collected on `sym` alone here, not both policies.

Each short rollout gives `WINDOWS_PER_ROLLOUT` clips; `N_ROLLOUTS` different-seeded rollouts are run
per condition/policy to reach the target count, each checked against the same fall threshold before
its clips are kept -- a rollout that falls contributes nothing, silently, rather than corrupting
the set the way the long-duration approach did.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYM = "sim/assets/b1_policy/base_1.7hz_sym/model_600.pt"
FRAMES, FPS = 66, 20.0
WINDOWS_PER_ROLLOUT = 2                 # the original, proven-safe duration
STEPS = int(WINDOWS_PER_ROLLOUT * FRAMES * (50.0 / FPS)) + 60     # 390, matches recollect_b1_more.py
FALL_Z = 0.35
CMD_NOISE = 0.0137

# name, level, first_episode_id, vx, vy, {policy: wz}; policy list per condition (side_R_lvl1: sym only)
CONDITIONS = (
    ("turn_w0.008", 0, 4000, 0.300, 0.0, {"sym": 0.023, "gait3": 0.081}, ("sym", "gait3")),
    ("turn_w0.024", 1, 4200, 0.300, 0.0, {"sym": 0.115, "gait3": 0.169}, ("sym", "gait3")),
    ("turn_w0.037", 2, 4400, 0.300, 0.0, {"sym": 0.191, "gait3": 0.252}, ("sym", "gait3")),
    ("turn_w0.075", 3, 4600, 0.300, 0.0, {"sym": 0.587, "gait3": 0.664}, ("sym", "gait3")),
    ("side_R_lvl1", 3, 4800, 0.0, -0.525, {"sym": 0.0}, ("sym",)),
)
POLICY_INFO = {"sym": (SYM, 1.7), "gait3": ("", 2.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/egocentric/beh12_b1_noisy_ego_flat")
    ap.add_argument("--port", type=int, default=23002)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--cam_fov", type=float, default=24.0)
    ap.add_argument("--floor_scale", type=float, default=3.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--target", type=int, default=20, help="clips wanted per condition, total "
                    "across its policies")
    ap.add_argument("--max_rollouts", type=int, default=15, help="give up on a policy after this "
                    "many falls in a row, rather than looping forever")
    args = ap.parse_args()

    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_traj")
    os.makedirs(tmp, exist_ok=True)
    py = os.path.join(ROOT, ".venv/bin/python3")

    for name, level, ep0, vx, vy, wz_by_policy, policies in CONDITIONS:
        per_policy_target = args.target // len(policies)
        k = 0
        for pol in policies:
            ckpt, gait = POLICY_INFO[pol]
            got, seed, attempts = 0, 0, 0
            while got < per_policy_target and attempts < args.max_rollouts:
                traj = os.path.join(tmp, f"{name}_{pol}_s{seed}.npz")
                cmd = [py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"),
                      "--vx", str(vx), "--vy", str(vy), "--wz", str(wz_by_policy[pol]),
                      "--steps", str(STEPS), "--gait_freq", str(gait),
                      "--cmd_noise", str(CMD_NOISE), "--noise_seed", str(seed), "--out", traj]
                if ckpt:
                    cmd += ["--checkpoint", os.path.join(ROOT, ckpt)]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                attempts += 1; seed += 1
                if r.returncode != 0:
                    print(f"  rollout FAILED {name}/{pol} seed{seed}\n{r.stderr[-400:]}")
                    continue
                # NOT the script's own WALKS/FELL print: that also requires x_travel > 0.5m, which
                # a legitimately successful sharp turn can fail without ever falling (confirmed:
                # turn_w0.075 read "FELL" with z never leaving 0.542-0.578, just y_drift +0.90m
                # from curving hard). Height alone, checked per-window below, is the real test.
                with np.load(traj, allow_pickle=True) as T:
                    data = {kk: T[kk] for kk in T.files}
                n = len(data["base_pos"])
                per = int(FRAMES * (50.0 / FPS))
                for c in range(WINDOWS_PER_ROLLOUT):
                    if got >= per_policy_target:
                        break
                    sl = slice(c * per, (c + 1) * per)
                    piece = os.path.join(tmp, f"{name}_{pol}_s{seed-1}_c{c}.npz")
                    cut = {kk: (v[sl] if getattr(v, "ndim", 0) and len(v) == n else v)
                          for kk, v in data.items()}
                    if cut["base_pos"][-1, 2] < FALL_Z:
                        print(f"  {name}/{pol} seed{seed-1} window{c}: height {cut['base_pos'][-1,2]:.3f} "
                             f"< {FALL_Z}, discarded")
                        continue
                    w, x, y, z = cut["base_quat"][0]
                    yaw0 = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    cc, ss = np.cos(-yaw0), np.sin(-yaw0)
                    d = cut["base_pos"][:, :2] - cut["base_pos"][0, :2]
                    cut["base_pos"] = cut["base_pos"].copy()
                    cut["base_pos"][:, 0] = cut["base_pos"][0, 0] + cc * d[:, 0] - ss * d[:, 1]
                    cut["base_pos"][:, 1] = cut["base_pos"][0, 1] + ss * d[:, 0] + cc * d[:, 1]
                    rw, rz = np.cos(-yaw0 / 2), np.sin(-yaw0 / 2)
                    qw, qx, qy, qz = cut["base_quat"][:, 0], cut["base_quat"][:, 1], \
                        cut["base_quat"][:, 2], cut["base_quat"][:, 3]
                    cut["base_quat"] = cut["base_quat"].copy()
                    cut["base_quat"][:, 0] = rw * qw - rz * qz
                    cut["base_quat"][:, 1] = rw * qx - rz * qy
                    cut["base_quat"][:, 2] = rw * qy + rz * qx
                    cut["base_quat"][:, 3] = rw * qz + rz * qw
                    np.savez_compressed(piece, **cut)
                    tag = f"b1_ep{ep0 + k}"
                    render_cmd = [py, os.path.join(ROOT, "sim/render/render_b1_replay.py"),
                                 "--port", str(args.port), "--scene", args.scene, "--traj", piece,
                                 "--out", out, "--fps", str(FPS), "--cam_fov", str(args.cam_fov),
                                 "--floor_scale", str(args.floor_scale), "--ego",
                                 "--spawn", str(args.spawn[0]), str(args.spawn[1])]
                    rr = subprocess.run(render_cmd, capture_output=True, text=True, timeout=1800)
                    if rr.returncode != 0:
                        print(f"  render FAILED {tag}\n{rr.stdout[-300:]}{rr.stderr[-800:]}")
                        continue
                    src = os.path.join(out, os.path.basename(piece))
                    with np.load(src, allow_pickle=True) as b:
                        merged = {kk: b[kk] for kk in b.files}
                    merged.update(condition=np.array(name), behaviour=np.array(
                        "turn" if name.startswith("turn") else "side"),
                        level=np.array(level), expert_episode=np.array(ep0 + k),
                        embodiment=np.array("b1"), policy=np.array(pol))
                    np.savez_compressed(os.path.join(out, tag + ".npz"), **merged)
                    os.remove(src)
                    print(f"  {tag}  {name}  {pol}  seed{seed-1}  ({got+1}/{per_policy_target})",
                         flush=True)
                    got += 1; k += 1
            if got < per_policy_target:
                print(f"  ! {name}/{pol}: only got {got}/{per_policy_target} after "
                     f"{attempts} rollouts -- giving up, not padding with bad data")
    print("done", flush=True)


if __name__ == "__main__":
    main()
