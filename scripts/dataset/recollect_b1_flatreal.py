"""Re-collect B1 expert clips on `b1_flat_real.xml` instead of `b1_flat.xml` -- same policies, same
conditions, same windowing as `recollect_b1_more.py`, but pointed at the physics the student is
actually evaluated in, and with the CoppeliaSim rendering step dropped entirely.

**Why this exists.** `beh12_b1_ego_flat` (the only B1 dataset that has existed so far) was collected
on `b1_flat.xml` (uniform placeholder joint damping/friction), while `B1MuJoCoEnv`'s eval default is
`b1_flat_real.xml` (system-identified). A whole session of B1 imitation-learning work (F216) trained
and evaluated across that mismatch without ever checking the cheap alternative first. A sanity check
confirmed the SAME trained policy checkpoints, unmodified, already walk cleanly on `b1_flat_real.xml`
(no retraining needed) -- so re-collecting the training data on the matched physics is
straightforward, not a research problem.

**No CoppeliaSim, no `frames` field.** `clone_b1.py`'s `body_state()`/`body_goal()` read only
`joint_pos`/`joint_vel`/`base_pos`/`base_quat`/`actions`/`dt` -- proprioception and commands, never
video. The original collection rendered through CoppeliaSim only because a DIFFERENT use of this
data (F210's cross-embodiment vision test) needs frames; this run doesn't, so it's skipped, which
also removes the only reason a CoppeliaSim instance was ever needed for B1 data collection.

    .venv/bin/python3 scripts/dataset/recollect_b1_flatreal.py --pilot
    .venv/bin/python3 scripts/dataset/recollect_b1_flatreal.py --behaviours speed \\
        --out data/proprioceptive/beh12_b1_flatreal
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "dataset"))
from recollect_b1_more import CONDITIONS, FRAMES, FPS, POLICIES, _face_forward  # noqa: E402

DEFAULT_MODEL = "sim/assets/b1_mujoco/b1_flat_real.xml"


def collect_recovery(args):
    """One clip per (yaw0, policy): spawn rotated `yaw0` degrees off the held heading target, no
    `_face_forward` (see `--yaw0_list`'s own help for why).

    `--wz 0.0` (default): `speed_vx0.30`'s own command -- the heading target stays fixed while the
    robot corrects a pure offset. `--wz` nonzero: a TURN recovery clip -- `heading_target` still
    starts at 0 (not at the perturbed spawn heading) but then keeps advancing at the commanded
    turn rate exactly as an ordinary turn condition's does, so the robot has to close the initial
    offset WHILE the target keeps moving, not chase a fixed point. `condition`/`behaviour` are
    labelled to match whichever this is (`speed` or `turn`), so `--forward_only`/behaviour
    filtering picks these up alongside the ordinary clips of the same kind -- they ARE that
    behaviour, just ones that also demonstrate recovery instead of starting already on target."""
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    py = os.path.join(ROOT, ".venv/bin/python3")
    tmp_dir = os.path.join(out, "_traj_recovery")
    os.makedirs(tmp_dir, exist_ok=True)
    yaw0s = [float(v) for v in args.yaw0_list.split(",")]
    vx, vy, wz = 0.300, 0.0, float(args.wz)
    behaviour = "speed" if wz == 0.0 else "turn"
    tag_prefix = "recover" if wz == 0.0 else "recoverturn"
    # **Native-rate steps, not 20 Hz frames** -- `args.steps` in rollout_b1_mujoco.py counts
    # steps at the policy's own 50 Hz decimated rate, same as `recollect_b1_more.py`'s own
    # `int(FRAMES * (50.0/FPS)) + buffer` convention. Using `FRAMES + 10` directly here the first
    # time produced only 31 of the needed 66 frames after 20 Hz subsampling -- caught before
    # training on it, the same class of rate mistake as the earlier 165-vs-66-frame bug.
    steps = int(FRAMES * (50.0 / FPS)) + 15
    ep0 = int(args.recovery_ep0)
    # **"sym" (base_1.7hz_sym) falls at every yaw0 tested here (-30..+30), even +-10 degrees --**
    # consistent with it already being the fragile policy (it also fell on every turn/side
    # condition in the main 12-condition collection). A fallen clip demonstrates "the robot
    # collapses," not "the robot recovers" -- including it would teach the wrong lesson, not a
    # harder version of the right one. "gait3" walked cleanly at every offset tested, so only it
    # is used here.
    recovery_policies = [(pol, ckpt, gait) for pol, ckpt, gait in POLICIES if pol == "gait3"]
    print(f"{len(yaw0s)} yaw0 offsets x {len(recovery_policies)} policy (gait3 only -- see docstring), "
         f"behaviour={behaviour} wz={wz}, physics = {args.model}, "
         f"no _face_forward (heading offset preserved)", flush=True)
    k = 0
    for yaw0 in yaw0s:
        for pol, ckpt, gait in recovery_policies:
            traj = os.path.join(tmp_dir, f"{tag_prefix}{yaw0:+.0f}_{pol}.npz")
            cmd = [py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"),
                  "--model", os.path.join(ROOT, args.model),
                  "--vx", str(vx), "--vy", str(vy), "--wz", str(wz), "--yaw0", str(yaw0),
                  "--steps", str(steps), "--gait_freq", str(gait), "--out", traj]
            if ckpt:
                cmd += ["--checkpoint", os.path.join(ROOT, ckpt)]
            print(f"  rollout {tag_prefix}{yaw0:+.0f}/{pol}: steps={steps} vx={vx} wz={wz} yaw0={yaw0}",
                 flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"  rollout FAILED {tag_prefix}{yaw0:+.0f}/{pol}\n{r.stdout[-300:]}{r.stderr[-800:]}")
                sys.exit(1)
            print(f"    {[ln for ln in r.stdout.splitlines() if 'WALKS' in ln or 'FELL' in ln][-1]}",
                 flush=True)
            with np.load(traj, allow_pickle=True) as T:
                data = {kk: T[kk] for kk in T.files}
            dt_native = float(data["dt"])
            n = len(data["base_pos"])
            step = 1.0 / (FPS * dt_native)
            idx = np.unique(np.round(np.arange(0, n, step)).astype(int))
            idx = idx[idx < n][:FRAMES]                    # exactly one 66-frame window
            cut = {kk: (v[idx] if getattr(v, "ndim", 0) and len(v) == n else v)
                  for kk, v in data.items()}
            cut["dt"] = np.float32(1.0 / FPS)
            cut["fps"] = np.float32(FPS)
            # base_pos/base_quat left untouched -- the true starting offset IS the data
            tag = f"b1_ep{ep0 + k}"
            cut.update(condition=np.array(f"{tag_prefix}_yaw{yaw0:+.0f}"), behaviour=np.array(behaviour),
                      level=np.array(0), expert_episode=np.array(ep0 + k),
                      embodiment=np.array("b1"), policy=np.array(pol), model=np.array(args.model),
                      yaw0=np.float32(yaw0), wz=np.float32(wz))
            np.savez_compressed(os.path.join(out, tag + ".npz"), **cut)
            print(f"  {tag}  {tag_prefix}_yaw{yaw0:+.0f}  {pol}  ({len(idx)} frames @ {FPS} Hz)",
                 flush=True)
            k += 1
            os.remove(traj)
    os.rmdir(tmp_dir)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/proprioceptive/beh12_b1_flatreal")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="the physics model to collect ON -- the whole point of this script is "
                         "that this differs from b1_flat.xml, what beh12_b1_ego_flat used")
    ap.add_argument("--behaviours", nargs="+", default=None, metavar="NAME",
                    help="restrict to conditions whose behaviour is in this list (speed/turn/side). "
                         "Default: all 12 conditions.")
    ap.add_argument("--clips_per_policy", type=int, default=2,
                    help="2 -> 4 clips/condition (2 policies), matching beh12_b1_ego_flat's own "
                         "count exactly -- this is a controlled re-collection, not a data-volume fix")
    ap.add_argument("--pilot", action="store_true",
                    help="one condition only (speed_vx0.30), to check the mechanism before the "
                         "full run")
    ap.add_argument("--yaw0_list", type=str, default="",
                    help="comma-separated starting heading offsets in degrees, e.g. '-30,-15,15,30' "
                         "-- collects RECOVERY clips instead of the usual conditions: the robot "
                         "spawns rotated this far off the held heading target and the same "
                         "heading-hold correction (on by default) has to pull it back over the "
                         "episode. Every existing clip starts at zero heading error by "
                         "construction (the default spawn always faces the target already), so "
                         "none of them ever demonstrated the correction responding to a real "
                         "offset -- this exists to generate exactly that. Uses speed_vx0.30's own "
                         "command (straight ahead, no side/turn); `--_face_forward` is skipped for "
                         "these clips since it would erase the very offset being collected (it "
                         "normalizes every clip to start at heading 0, which would flip a "
                         "'start at +20, correct to 0' recovery into a 'start at 0, drift to -20' "
                         "one when read back).")
    ap.add_argument("--wz", type=float, default=0.0,
                    help="only with --yaw0_list: 0.0 (default) collects speed-recovery clips "
                         "(fixed heading target). Nonzero collects TURN-recovery clips instead: "
                         "heading_target still starts at 0, not the perturbed spawn heading, but "
                         "then advances at this commanded rate exactly like an ordinary turn "
                         "condition -- the robot has to close the initial offset WHILE the target "
                         "keeps moving, not chase a fixed point. Pass a turn condition's own gait3 "
                         "wz (e.g. 0.169 for turn_w0.024) to match real turn dynamics.")
    ap.add_argument("--recovery_ep0", type=int, default=4000,
                    help="only with --yaw0_list: starting expert_episode id. Use a different range "
                         "per call (e.g. 4000 for speed-recovery, 4100 for turn-recovery) so runs "
                         "don't overwrite each other's clips.")
    args = ap.parse_args()

    if args.yaw0_list:
        collect_recovery(args)
        return

    conditions = CONDITIONS
    if args.pilot:
        conditions = [c for c in CONDITIONS if c[0] == "speed_vx0.30"]
    elif args.behaviours:
        conditions = [c for c in CONDITIONS if c[1] in args.behaviours]
    if not conditions:
        raise SystemExit(f"no conditions match --behaviours {args.behaviours}")

    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_traj")
    os.makedirs(tmp, exist_ok=True)
    py = os.path.join(ROOT, ".venv/bin/python3")
    per = int(FRAMES * (50.0 / FPS))
    steps = int(args.clips_per_policy * FRAMES * (50.0 / FPS)) + 60

    print(f"{len(conditions)} conditions x {len(POLICIES)} policies x {args.clips_per_policy} "
         f"window-clips, physics = {args.model}", flush=True)

    for name, behaviour, level, ep0, vx, vy, wz_by_policy in conditions:
        k = 0
        for pol, ckpt, gait in POLICIES:
            traj = os.path.join(tmp, f"{name}_{pol}.npz")
            cmd = [py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"),
                  "--model", os.path.join(ROOT, args.model),
                  "--vx", str(vx), "--vy", str(vy), "--wz", str(wz_by_policy[pol]),
                  "--steps", str(steps), "--gait_freq", str(gait), "--out", traj]
            if ckpt:
                cmd += ["--checkpoint", os.path.join(ROOT, ckpt)]
            print(f"  rollout {name}/{pol}: steps={steps}  vx={vx} vy={vy} wz={wz_by_policy[pol]}",
                 flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"  rollout FAILED {name}/{pol}\n{r.stdout[-300:]}{r.stderr[-800:]}")
                sys.exit(1)
            print(f"    {[ln for ln in r.stdout.splitlines() if 'WALKS' in ln or 'FELL' in ln][-1]}",
                 flush=True)
            with np.load(traj, allow_pickle=True) as T:
                data = {kk: T[kk] for kk in T.files}
            dt_native = float(data["dt"])
            n = len(data["base_pos"])
            for c in range(args.clips_per_policy):
                sl = slice(c * per, (c + 1) * per)
                cut = {kk: (v[sl] if getattr(v, "ndim", 0) and len(v) == n else v)
                      for kk, v in data.items()}
                # **Subsample native-rate frames down to 20 Hz, matching `render_b1_replay.py`'s own
                # formula exactly.** That script normally does this at render time; skipping it here
                # (no CoppeliaSim, no frames) left an earlier pilot run at the native ~50 Hz window
                # (165 frames) instead of the 66-frame/20 Hz clips clone_b1's STEPS/D_real/eval-bar
                # convention all assume -- caught before the full collection ran.
                cn = len(cut["base_pos"])
                step = 1.0 / (FPS * dt_native)
                keep = np.unique(np.round(np.arange(0, cn, step)).astype(int))
                keep = keep[keep < cn]
                cut = {kk: (v[keep] if getattr(v, "ndim", 0) and len(v) == cn else v)
                      for kk, v in cut.items()}
                cut["dt"] = np.float32(1.0 / FPS)
                cut["fps"] = np.float32(FPS)
                cut["base_pos"], cut["base_quat"] = _face_forward(cut["base_pos"], cut["base_quat"])
                tag = f"b1_ep{ep0 + k}"
                cut.update(condition=np.array(name), behaviour=np.array(behaviour),
                          level=np.array(level), expert_episode=np.array(ep0 + k),
                          embodiment=np.array("b1"), policy=np.array(pol),
                          model=np.array(args.model))
                np.savez_compressed(os.path.join(out, tag + ".npz"), **cut)
                print(f"  {tag}  {name}  {pol}  vx {vx:+.3f} vy {vy:+.3f} wz {wz_by_policy[pol]:+.3f}"
                     f"  ({len(keep)} frames @ {FPS} Hz)", flush=True)
                k += 1
            os.remove(traj)
    os.rmdir(tmp)
    print("done", flush=True)


if __name__ == "__main__":
    main()
