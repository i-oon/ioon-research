"""More B1 clips per condition via longer rollouts, more non-overlapping windows -- no noise yet.

    .venv/bin/python3 scripts/dataset/recollect_b1_more.py --pilot
    .venv/bin/python3 scripts/dataset/recollect_b1_more.py --out data/egocentric/beh12_b1_more_raw

**Why this exists.** Hexapod's data-sparsity fix (5x more clips, same 12 behaviours) does not
transfer to B1 by re-running collection more times -- MuJoCo is deterministic, so the same command
gives the same clip back (see memory `b1-mujoco-deterministic`). The existing 4 clips per condition
are 4 non-overlapping WINDOWS of one longer rollout (`recollect_b1_turns.py`'s own mechanism,
generalised here to all 12 conditions, not just turn). This is the free lever: run the rollout
longer, cut more windows. It adds training examples (different gait phase, different position) but
NOT genuine repeat-to-repeat outcome variance the way real physics noise or independent episodes
would -- see the noise-injection plan (separate, not this script) for that.

**Commands, and where each came from.**

  turn   per-policy wz, copied verbatim from `recollect_b1_turns.py`'s own LEVELS table --
         already verified against the insect reference (F108/F109) and NOT re-derived here.
  speed  vx read directly off the existing `beh12_b1_ego_flat` clips' own stored `command` field
         (mean per condition), same value for both policies -- no per-policy speed split is
         recorded in the existing data, so none is assumed here.
  side   vy read the same way. wz is left at 0 for both speed and side; the small residual wz the
         existing data shows (0.028-0.047) comes from the rollout's own built-in heading-correction
         trim, not a manual command, and letting the same trim run here reproduces it rather than
         fighting it.

**Pilot mode first, on purpose.** `--pilot` reproduces the EXISTING clip count (2 per policy, 4
total) for a couple of conditions only, so the output can be checked against the real
`beh12_b1_ego_flat` numbers before spending the time to render all 12 conditions at 5x. Do not skip
this -- F174 (hexapod's turn-sign bug) was exactly the failure mode a small check like this catches
before it costs a full collection run.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYM = "sim/assets/b1_policy/base_1.7hz_sym/model_600.pt"
POLICIES = (("sym", SYM, 1.7), ("gait3", "", 2.0))
FRAMES, FPS = 66, 20.0

# name, behaviour, level, first_episode_id, vx, vy, {policy: wz}  -- wz per-policy only matters
# for turn; speed/side pass the same (0.0) for both and let the rollout's own heading trim produce
# the small residual the existing data already shows.
CONDITIONS = (
    ("speed_vx0.30", "speed", 0, 2000, 0.300, 0.0, {"sym": 0.0, "gait3": 0.0}),
    ("speed_vx0.38", "speed", 1, 2100, 0.381, 0.0, {"sym": 0.0, "gait3": 0.0}),
    ("speed_vx0.40", "speed", 2, 2200, 0.416, 0.0, {"sym": 0.0, "gait3": 0.0}),
    ("speed_vx0.50", "speed", 3, 2300, 0.495, 0.0, {"sym": 0.0, "gait3": 0.0}),
    ("turn_w0.008", "turn", 0, 2400, 0.300, 0.0, {"sym": 0.023, "gait3": 0.081}),
    ("turn_w0.024", "turn", 1, 2500, 0.300, 0.0, {"sym": 0.115, "gait3": 0.169}),
    ("turn_w0.037", "turn", 2, 2600, 0.300, 0.0, {"sym": 0.191, "gait3": 0.252}),
    ("turn_w0.075", "turn", 3, 2700, 0.300, 0.0, {"sym": 0.587, "gait3": 0.664}),
    # Recalibrated (superseding the original 0.220/0.520/-0.330/-0.525, which gave lateral speeds
    # 2-5x smaller than the matching hex condition -- see the cross-embodiment Froude-match audit).
    # Target |lat| schedule, shared with hex's own recalibrated side table: 0.03/0.08/0.13/0.14.
    # sym and gait3 need different vy for the same target -- gait3 tracks strafe ~2x more
    # sensitively than sym, verified by direct rollout measurement, not assumed.
    ("side_L_lvl0", "side", 0, 2800, 0.0, {"sym": 0.12, "gait3": 0.12}, {"sym": 0.0, "gait3": 0.0}),
    ("side_L_lvl1", "side", 1, 2900, 0.0, {"sym": 0.33, "gait3": 0.24}, {"sym": 0.0, "gait3": 0.0}),
    ("side_R_lvl0", "side", 2, 3000, 0.0, {"sym": -0.11, "gait3": -0.034}, {"sym": 0.0, "gait3": 0.0}),
    ("side_R_lvl1", "side", 3, 3100, 0.0, {"sym": -0.29, "gait3": -0.18}, {"sym": 0.0, "gait3": 0.0}),
)

# Mirror conditions (F218): the table above is forward-only speed, one-direction-only turn, and
# side is the only axis with real bidirectional coverage. These 12 add the missing half -- signs
# just flipped on the existing vx/wz, except SIDE_EXTRA which extrapolates two more magnitude
# levels per direction (no reference data for these, unlike speed/turn which reuse verified
# magnitudes with a flipped sign). Episode ids 3200+ so they never collide with the base table.
SPEED_BWD = tuple(
    (name.replace("speed_vx", "speed_vx-").replace("--", "-"), "speed", lvl + 4, ep0 + 1200,
     -vx, vy, wz)
    for lvl, (name, _, _, ep0, vx, vy, wz) in enumerate(c for c in CONDITIONS if c[1] == "speed")
)
TURN_NEG = tuple(
    (name + "_neg", "turn", lvl + 4, ep0 + 1200, vx, vy,
     {p: -w for p, w in wz.items()})
    for lvl, (name, _, _, ep0, vx, vy, wz) in enumerate(c for c in CONDITIONS if c[1] == "turn")
)
# Recalibrated (superseding the first-guess extrapolation 0.720/0.920/-0.725/-0.925, which was
# tuned for hex's own gait and, applied raw to B1, pushed `gait3` past its real control limit --
# it tracked fine up to about vy=-0.5/-0.6 and collapsed beyond that, near-zero or wrong-sign
# lateral velocity). Same shared target schedule as lvl0/1 above: 0.03/0.08/0.13/0.14 -> here 0.13/0.14.
SIDE_EXTRA = (
    ("side_L_lvl2", "side", 4, 4000, 0.0, {"sym": 0.64, "gait3": 0.37}, {"sym": 0.0, "gait3": 0.0}),
    ("side_L_lvl3", "side", 5, 4100, 0.0, {"sym": 0.71, "gait3": 0.40}, {"sym": 0.0, "gait3": 0.0}),
    ("side_R_lvl2", "side", 6, 4200, 0.0, {"sym": -0.51, "gait3": -0.29}, {"sym": 0.0, "gait3": 0.0}),
    ("side_R_lvl3", "side", 7, 4300, 0.0, {"sym": -0.58, "gait3": -0.33}, {"sym": 0.0, "gait3": 0.0}),
)
CONDITIONS = CONDITIONS + SPEED_BWD + TURN_NEG + SIDE_EXTRA


def _face_forward(pos, quat):
    """Rotate a path about its own first point so it starts at yaw 0.

    Copied verbatim from `recollect_b1_turns.py`, not reconstructed -- B1's quaternion is MuJoCo's
    (w, x, y, z), and a hand-rewritten version of this that assumed (x, y, z, w) was caught in
    review before it ran: wrong component order plus translating to the origin instead of rotating
    about the first point in place would have silently mis-oriented every window.
    """
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
    ap.add_argument("--out", default="data/egocentric/beh12_b1_more_raw")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    # 0 -> render_b1_replay.py's own default (24 deg third-person / 90 deg --ego). Used to be
    # hardcoded 24 here regardless of --ego, silently overriding that script's ego-aware default
    # and shooting every egocentric B1 clip through a third-person-tuned telephoto lens -- a
    # corner never entered frame even across 75 degrees of real yaw. Fixed alongside that script.
    ap.add_argument("--cam_fov", type=float, default=0.0)
    ap.add_argument("--floor_scale", type=float, default=3.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--ego", action="store_true", help="egocentric camera, matching the ego dataset")
    ap.add_argument("--clips_per_policy", type=int, default=10,
                    help="10 -> 20 clips/condition (2 policies), 5x the existing 4. --pilot "
                         "overrides this to 2 regardless.")
    ap.add_argument("--pilot", action="store_true",
                    help="2 clips/policy (matches the existing count exactly) on the first 2 "
                         "conditions only -- run this and verify against beh12_b1_ego_flat's own "
                         "numbers before the real run")
    ap.add_argument("--only", nargs="*", default=[], help="condition names, for a partial re-run")
    args = ap.parse_args()

    # one of each behaviour, not just the first two (both speed) -- turn's per-policy wz split is
    # the trickier mechanism to get right and deserves its own check before the full run
    pilot_set = [c for c in CONDITIONS if c[0] in ("speed_vx0.30", "turn_w0.008", "side_L_lvl0")]
    conditions = pilot_set if args.pilot else CONDITIONS
    if args.only:
        conditions = [c for c in conditions if c[0] in args.only]
    clips_per_policy = 2 if args.pilot else args.clips_per_policy

    out = os.path.join(ROOT, args.out)
    tmp = os.path.join(out, "_traj")
    os.makedirs(tmp, exist_ok=True)
    py = os.path.join(ROOT, ".venv/bin/python3")
    per = int(FRAMES * (50.0 / FPS))
    steps = int(clips_per_policy * FRAMES * (50.0 / FPS)) + 60

    for name, behaviour, level, ep0, vx, vy, wz_by_policy in conditions:
        k = 0
        for pol, ckpt, gait in POLICIES:
            # vy may be a flat number (speed/turn: same for both policies) or a {policy: vy} dict
            # (side, post-calibration: sym and gait3 need different magnitudes to hit the same
            # target lateral speed -- gait3 tracks strafe roughly 2x more sensitively than sym).
            vy_pol = vy[pol] if isinstance(vy, dict) else vy
            traj = os.path.join(tmp, f"{name}_{pol}.npz")
            cmd = [py, os.path.join(ROOT, "sim/collect/rollout_b1_mujoco.py"),
                  "--vx", str(vx), "--vy", str(vy_pol), "--wz", str(wz_by_policy[pol]),
                  "--steps", str(steps), "--gait_freq", str(gait), "--out", traj]
            if ckpt:
                cmd += ["--checkpoint", os.path.join(ROOT, ckpt)]
            print(f"  rollout {name}/{pol}: steps={steps}  vx={vx} vy={vy_pol} wz={wz_by_policy[pol]}",
                 flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                print(f"  rollout FAILED {name}/{pol}\n{r.stdout[-300:]}{r.stderr[-800:]}")
                sys.exit(1)
            with np.load(traj, allow_pickle=True) as T:
                data = {kk: T[kk] for kk in T.files}
            n = len(data["base_pos"])
            for c in range(clips_per_policy):
                sl = slice(c * per, (c + 1) * per)
                piece = os.path.join(tmp, f"{name}_{pol}_c{c}.npz")
                cut = {kk: (v[sl] if getattr(v, "ndim", 0) and len(v) == n else v)
                      for kk, v in data.items()}
                cut["base_pos"], cut["base_quat"] = _face_forward(cut["base_pos"], cut["base_quat"])
                np.savez_compressed(piece, **cut)
                tag = f"b1_ep{ep0 + k}"
                render_cmd = [py, os.path.join(ROOT, "sim/render/render_b1_replay.py"),
                             "--port", str(args.port),
                             "--scene", args.scene, "--traj", piece, "--out", out,
                             "--fps", str(FPS), "--cam_fov", str(args.cam_fov),
                             "--floor_scale", str(args.floor_scale),
                             "--spawn", str(args.spawn[0]), str(args.spawn[1])]
                if args.ego:
                    # room = clip index within the condition (0..3), matching hex's "room = repeat
                    # index" convention (step1_egocentric.sh) -- without this every clip of a
                    # condition got the identical room (fixed default ego_seed=0), killing the
                    # per-clip room variety the original recipe relies on.
                    render_cmd += ["--ego", "--ego_seed", str(k)]
                r = subprocess.run(render_cmd, capture_output=True, text=True, timeout=1800)
                if r.returncode != 0:
                    print(f"  render FAILED {tag}\n{r.stdout[-300:]}{r.stderr[-800:]}")
                    sys.exit(1)
                src = os.path.join(out, os.path.basename(piece))
                with np.load(src, allow_pickle=True) as b:
                    merged = {kk: b[kk] for kk in b.files}
                merged.update(condition=np.array(name), behaviour=np.array(behaviour),
                             level=np.array(level), expert_episode=np.array(ep0 + k),
                             embodiment=np.array("b1"), policy=np.array(pol))
                np.savez_compressed(os.path.join(out, tag + ".npz"), **merged)
                os.remove(src)
                print(f"  {tag}  {name}  {pol}  vx {vx:+.3f} vy {vy_pol:+.3f} wz {wz_by_policy[pol]:+.3f}",
                     flush=True)
                k += 1
    print("done", flush=True)


if __name__ == "__main__":
    main()
