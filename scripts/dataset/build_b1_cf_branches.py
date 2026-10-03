"""B1 counterfactual-branch training clips: one exact MuJoCo start state, every command (v3 matched rendering).

Each source clip of beh24_b1_ego_flat (TRAIN split for the training set, VAL split for the val set) is a
165-policy-step window of a long `rollout_b1_mujoco.py` run (recollect_b1_more.py / recollect_b1_turns.py:
policy warmup 45 steps, 50 Hz policy, frames kept at round(arange(0, 165, 2.5)) -> 66 frames at 20 Hz,
window rotated by `_face_forward`). The source rollout is re-run here with the identical loop (heading PI
controller included), its policy / command / heading gains / window start are located by exact match of
the clip's stored joint_pos (all 72 train+val sources reproduce to <1e-6 rad; the older beh12 speed clips
were rolled out with F69's PI gains kp 2.5 / ki 1.0 and cut at recorded step 0 or 50, every other source
with kp 0.5 / ki 0 at multiples of 165). The state at source frame t is snapshotted (mjSTATE_INTEGRATION
+ sensordata + last action + gait-clock step_i + heading_target + yaw_int + the rollout SETTINGS: heading
kp / ki / ki_clip, policy checkpoint + gait clock, model xml). From the snapshot, each of the 24 condition
commands (recollect_b1_more.CONDITIONS, the source's own policy's values) is executed by the same loop
for BRANCH frames. **The command is the only thing that changes at the branch point**: every setting is
the one `locate` found reproducing the source (never a default); a source whose settings cannot be
determined raises. (Before 2026-10-01 every branch was run with kp 0.5 / ki 0, so the branches of the
kp 2.5 / ki 1 beh12 speed sources switched controller at the branch point: own-command branch off its
source by up to 0.023 rad joint / 0.015 Froude. Regenerated.) The settings are stored per branch file
(`heading_kp`, `heading_ki`, `heading_ki_clip`, `policy`, `policy_checkpoint`, `gait_freq`, `model`).

Output clip = the source's own last PREFIX frames up to t (copied: identical to a re-render, checked)
+ the shared start frame t + BRANCH branch frames, in exactly the beh24 npz format, every field filled
the way the collector fills it; poses are put in the source window's `_face_forward` frame and rendered
in the source's v3 room (render_b1_replay.py --ego --match_floor --ground_uv_mult 1.0, seed = source
clip's room seed). Why a prefix and BRANCH 20 (not 11 frames): `body_motion` / yaw is smoothed over a
1 s (20-frame) 'same' convolution -- an 11-frame clip returns a 20-long body_motion and zero-padded
labels; with 10 frames of real context on each side, the labels of the pairs starting at the branch
point equal the in-context definition. Manifest records `branch_index` (= PREFIX) per file.

    .venv/bin/python3 scripts/dataset/build_b1_cf_branches.py --check          # sanity checks
    .venv/bin/python3 scripts/dataset/build_b1_cf_branches.py --split train
    .venv/bin/python3 scripts/dataset/build_b1_cf_branches.py --split val
    .venv/bin/python3 scripts/dataset/build_b1_cf_branches.py --split heldout   (heldout sources, t 15 35)
CPU only. Needs CoppeliaSim on --port.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import mujoco
import numpy as np
import torch

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts/dataset", "sim/collect", "sim/render", "sim/scene",
          "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))
from recollect_b1_more import CONDITIONS, POLICIES, _face_forward  # noqa: E402
from wm.data.com import b1_com  # noqa: E402
from rollout_b1_mujoco import (ACTION_SCALE, DECIMATION, DEFAULT_IL, FOOT_FORCE_THRESH, MODEL,  # noqa: E402
                               SPAWN_Z, _TOUCH_SDK_TO_IL_LEG, il_to_sdk, load_actor, quat_to_R, sdk_to_il)

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
PER = 165                                     # policy steps per source window (66 frames at 20 Hz)
KEEP = np.unique(np.round(np.arange(0, PER, 2.5)).astype(int))   # render_b1_replay --fps 20, n=165
POLICY_WARMUP, SETTLE = 45, 25
HEAD_KI_CLIP = 2.0                          # rollout_b1_mujoco --head_ki_clip default; every B1 collector
GAINS = ((0.5, 0.0), (2.5, 1.0))           # rollout default; F69's PI (--head_kp 2.5 --head_ki 1.0)
SETTING_KEYS = ("heading_kp", "heading_ki", "heading_ki_clip", "policy", "policy_checkpoint", "gait_freq",
                "model")
FIELDS = ("base_pos", "base_quat", "joint_pos", "joint_vel", "action", "command", "foot_contact")
COND = {c[0]: c for c in CONDITIONS}                     # name -> (name, beh, level, ep0, vx, vy, wz)
NAMES = [c[0] for c in CONDITIONS]                        # 24, fixed order = command index
TURN_EP0 = {1000: "turn_w0.008", 1100: "turn_w0.024", 1200: "turn_w0.037", 1300: "turn_w0.075"}


def cond_cmd(name, pol):
    _, _, _, _, vx, vy, wz = COND[name]
    return (float(vx), float(vy[pol] if isinstance(vy, dict) else vy), float(wz[pol]))


class Roll:
    """rollout_b1_mujoco.py's loop, step for step, as an object with an exact full-state snapshot."""

    def __init__(self, pol, gains, model):
        # no defaults: every caller passes the settings `locate` found for the clip it continues
        self.kp, self.ki = (float(g) for g in gains)
        self.ki_clip = HEAD_KI_CLIP
        ckpt = dict((p, (c, g)) for p, c, g in POLICIES)[pol]
        self.pol = pol
        self.ckpt = ckpt[0] or "sim/assets/b1_policy/base_gait3/model_600.pt"
        self.actor = load_actor(os.path.join(ROOT, self.ckpt))
        self.use_clock = self.actor[0].in_features == 60
        self.gait = ckpt[1]
        self.model = model
        m = self.m = mujoco.MjModel.from_xml_path(model)
        d = self.d = mujoco.MjData(m)
        self.adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): m.sensor_adr[i]
                    for i in range(m.nsensor)}
        d.qpos[0:3] = [0, 0, SPAWN_Z]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qpos[7:19] = il_to_sdk(DEFAULT_IL); d.ctrl[:] = il_to_sdk(DEFAULT_IL)
        mujoco.mj_forward(m, d)
        for _ in range(SETTLE * DECIMATION):
            mujoco.mj_step(m, d)
        self.last = np.zeros(12, np.float32); self.step_i = 0
        self.heading_target = None; self.yaw_int = 0.0

    def step(self, vx, vy, wz):
        m, d, adr = self.m, self.d, self.adr
        R = quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"] + 4])
        cur_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if self.heading_target is None:
            self.heading_target = cur_yaw
        self.heading_target += wz * (DECIMATION * m.opt.timestep)
        err = float(np.arctan2(np.sin(self.heading_target - cur_yaw), np.cos(self.heading_target - cur_yaw)))
        self.yaw_int += err * (DECIMATION * m.opt.timestep)
        self.yaw_int = float(np.clip(self.yaw_int, -self.ki_clip, self.ki_clip))
        cmd = np.array([vx, vy, float(np.clip(self.kp * err + self.ki * self.yaw_int, -1, 1))], np.float32)
        lin = d.sensordata[adr["base_linvel"]:adr["base_linvel"] + 3]
        ang = d.sensordata[adr["base_angvel"]:adr["base_angvel"] + 3]
        grav = quat_to_R(d.sensordata[adr["base_quat"]:adr["base_quat"] + 4]).T @ np.array([0., 0., -1.])
        jpos = sdk_to_il(d.qpos[7:19]) - DEFAULT_IL
        jvel = sdk_to_il(d.qvel[6:18])
        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"] + 4])[_TOUCH_SDK_TO_IL_LEG]
        foot = (touch > FOOT_FORCE_THRESH).astype(np.float32)
        obs = np.concatenate([lin, ang, grav, cmd, jpos, jvel, self.last, foot]).astype(np.float32)
        if self.use_clock:
            t = self.step_i * DECIMATION * m.opt.timestep
            phi = 2 * np.pi * self.gait * t + np.array([0., np.pi, np.pi, 0.])
            obs = np.concatenate([obs, np.sin(phi), np.cos(phi)]).astype(np.float32)
        with torch.no_grad():
            action = self.actor(torch.from_numpy(obs)).numpy()
        self.last = action
        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        self.step_i += 1
        return {"base_pos": d.qpos[0:3].copy(), "base_quat": d.qpos[3:7].copy(),
                "joint_pos": d.qpos[7:19].copy(), "joint_vel": d.qvel[6:18].copy(),
                "action": action.copy(), "command": cmd.copy(), "foot_contact": foot.copy()}

    def settings(self):
        return {"heading_kp": self.kp, "heading_ki": self.ki, "heading_ki_clip": self.ki_clip, "policy": self.pol,
                "policy_checkpoint": self.ckpt, "gait_freq": float(self.gait),
                "model": os.path.relpath(self.model, ROOT)}

    def snapshot(self):
        s = np.empty(mujoco.mj_stateSize(self.m, STATE))
        mujoco.mj_getState(self.m, self.d, s, STATE)
        return (s, self.last.copy(), self.step_i, self.heading_target, self.yaw_int, self.d.sensordata.copy(),
                self.settings())

    def restore(self, snap):
        # **sensordata is restored too, not recomputed.** After mj_step the sensors hold the values
        # computed at the start of that step (pre-integration), and the policy reads them; mj_forward
        # after mj_setState recomputes them for the post-step state, and the branch then differs
        # from the uninterrupted run by ~1e-3 rad within 20 frames.
        s, last, self.step_i, self.heading_target, self.yaw_int, sens, settings = snap
        if settings != self.settings():
            # a snapshot carries the rollout's settings; continuing it under others is exactly the bug
            # this guards against (branch-point controller switch, 2026-10-01)
            raise RuntimeError(f"restore under different rollout settings: {settings} vs {self.settings()}")
        mujoco.mj_setState(self.m, self.d, s, STATE)
        mujoco.mj_forward(self.m, self.d)
        self.d.sensordata[:] = sens
        self.last = last.copy()

    def upright(self):
        w, x, y, z = self.d.qpos[3:7]
        return float(1 - 2 * (x * x + y * y))


def source_candidates(d, ep):
    """(policy, (vx, vy, wz)) hypotheses for how a source clip was rolled out."""
    cond = str(d["condition"])
    pols = [str(d["policy"])] if "policy" in d.files else ["sym", "gait3"]
    gains_list = GAINS
    if "rollout_head_kp" in d.files:
        # clips collected since 2026-10-01 record their rollout settings: those are the only hypothesis
        # (still verified by the exact replay in `locate`); settings Roll does not implement -> refuse
        unsupported = {"rollout_cmd_noise": 0.0, "rollout_yaw0": 0.0, "rollout_warmup": SETTLE,
                       "rollout_policy_warmup": POLICY_WARMUP, "rollout_head_ki_clip": HEAD_KI_CLIP,
                       "rollout_load_state": ""}
        for k, v in unsupported.items():
            if k in d.files and str(d[k]) != str(v) and not (isinstance(v, float) and float(d[k]) == v):
                raise RuntimeError(f"{k} = {d[k]}: Roll reproduces only {k} = {v!r}")
        gains_list = ((float(d["rollout_head_kp"]), float(d["rollout_head_ki"])),)
        by_ckpt = {(c or "sim/assets/b1_policy/base_gait3/model_600.pt"): p for p, c, g in POLICIES}
        pols = [by_ckpt[str(d["rollout_checkpoint"])]]
    vx0, vy0 = float(d["command"][0, 0]), float(d["command"][0, 1])
    out = []
    for gains in gains_list:
      for pol in pols:
        if cond.startswith("turn") and ep < 2000:
            out.append((pol, cond_cmd(TURN_EP0[ep // 100 * 100], pol), gains))
        elif cond in COND:
            vx, vy, wz = cond_cmd(cond, pol)
            out.append((pol, (vx, vy, wz), gains))
            if abs(vx - vx0) > 1e-6 or abs(vy - vy0) > 1e-6:      # old beh12 speed clips: own vx
                out.append((pol, (vx0, vy0, wz), gains))
        else:                                                         # unknown name: clip's own vx, vy
            out.append((pol, (vx0, vy0, 0.0), gains))
    return out


def locate(path, max_windows=10):
    """Re-run the source's rollout and find its window. Returns dict or raises."""
    d = np.load(path, allow_pickle=True)
    ep = int(d["expert_episode"])
    j = d["joint_pos"].astype(np.float64)
    # the model xml: the clip's own record if it has one, else the only one any B1 collector used (MODEL);
    # either way it is confirmed by the exact replay below (joint_pos < 1e-5 over the whole window)
    model = os.path.join(ROOT, str(d["rollout_model"] if "rollout_model" in d.files else d["model"])) \
        if ("rollout_model" in d.files or "model" in d.files) else MODEL
    zc, jv = d["base_pos"][:, 2].astype(np.float64), d["joint_vel"].astype(np.float64)
    Pc, Qc = d["base_pos"].astype(np.float64), d["base_quat"].astype(np.float64)
    best = None
    cands = []          # (cmd, policy, gains, best score) per hypothesis that reproduces the clip
    for pol, cmd, gains in source_candidates(d, ep):
        hyp_best = None
        r = Roll(pol, gains, model)
        for _ in range(POLICY_WARMUP):
            r.step(*cmd)
        R_ = [r.step(*cmd) for _ in range(max_windows * PER)]
        J = np.asarray([x["joint_pos"] for x in R_], np.float32)
        # window start: any recorded step (recollect_* cut at multiples of 165; the older beh12 speed
        # clips were cut elsewhere, e.g. at 50). **The best match over every offset and hypothesis, not
        # the first under threshold**: the gait is a limit cycle, so straight walking matches joint_pos
        # to ~2e-6 one or more periods away from the true window (F293 fix run, beh12_b1_more ep2009).
        # Score = joint_pos, joint_vel/100, height and the planar path's rigid-fit residual.
        for off in range(len(J) - PER + 1):
            if np.abs(J[off] - j[0]).max() > 1e-5:
                continue
            err = float(np.abs(J[off + KEEP] - j).max())
            if err > 1e-5:
                continue
            P = np.asarray([R_[off + k]["base_pos"] for k in KEEP])
            Q = np.asarray([R_[off + k]["base_quat"] for k in KEEP])
            V = np.asarray([R_[off + k]["joint_vel"] for k in KEEP])
            _, res_pos = fit_planar(P, Q, Pc, Qc, pos_only=True)
            score = max(err, float(np.abs(P[:, 2] - zc).max()), float(np.abs(V - jv).max()) / 100, res_pos)
            if score > 1e-5:
                continue
            hyp_best = score if hyp_best is None else min(hyp_best, score)
            # the limit cycle repeats to ~1e-7 every gait period, so windows one or more periods apart
            # tie on every relative quantity; the face-forward keeps the window's first point where the
            # rollout put it, so the absolute start position breaks the tie
            dist = float(np.abs(P[0, :2] - Pc[0, :2]).max())
            if best is None or dist < best[0]:
                best = (dist, pol, cmd, gains, off, err, P, Q, score)
        if hyp_best is not None:
            cands.append((cmd, pol, gains, hyp_best))
    if best is not None:
        _, pol, cmd, gains, off, err, P, Q, score = best
        # stored poses = the (F293-correct) `_face_forward` of the window, or (older beh12 speed / side
        # clips) another rigid rotation, fitted
        ff, _ = face_forward_transform(P[0], Q[0])
        P2, Q2 = ff(P, Q)
        res_ff = max(float(np.abs(P2 - Pc).max()), float(np.abs(Q2 - Qc).max()))
        tf, res = fit_planar(P, Q, Pc, Qc)
        if res_ff <= res:
            tf, res = ff, res_ff
        # the settings that reproduce the source, nothing else: a tie between hypotheses that differ in a
        # setting (gains / policy / model) would leave the setting undetermined -> refuse
        ties = [h for h in cands if h[3] <= max(10 * score, 1e-6) and h[1:3] != (pol, gains)]
        if ties:
            raise RuntimeError(f"{path}: rollout settings ambiguous: best {(pol, gains)} score {score:.1e}, "
                               f"also reproduced by {[(h[1], h[2], f'{h[3]:.1e}') for h in ties]}")
        loc = {"policy": pol, "cmd": cmd, "gains": gains, "off": off, "err": err, "score": score, "tf": tf,
               "tf_res": res, "model": model}
        loc["settings"] = Roll(pol, gains, model).settings()
        return loc
    raise RuntimeError(f"{path}: no rollout hypothesis reproduces the clip")


def rollout_to(loc, step_rec):
    """Fresh rollout of the source, stopped right after recorded step `step_rec`; also returns its window."""
    r = Roll(loc["policy"], loc["gains"], loc["model"])
    if r.settings() != loc["settings"]:
        raise RuntimeError(f"rollout settings {r.settings()} != located {loc['settings']}")
    for _ in range(POLICY_WARMUP):
        r.step(*loc["cmd"])
    rec = []
    for _ in range(step_rec + 1):
        rec.append(r.step(*loc["cmd"]))
    return r, rec


def fit_planar(P, Q, Pc, Qc, pos_only=False):
    """The rigid xy rotation + translation that maps the raw MuJoCo window onto the stored clip.

    Fitted (Kabsch, 2-D) rather than assumed: the recollect_* clips went through `_face_forward`, the
    older beh12 speed clips through another rotation. Returns tf(pos, quat) and the max residual over
    the whole window of positions and quaternions (sign-aligned)."""
    a, b = P[:, :2] - P[:, :2].mean(0), Pc[:, :2] - Pc[:, :2].mean(0)
    H = a.T @ b
    yaw = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rm = np.array([[c, -s_], [s_, c]])
    tr = Pc[:, :2].mean(0) - Rm @ P[:, :2].mean(0)
    qr = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    def tf(pos, quat):
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        out = pos.copy()
        out[:, :2] = pos[:, :2] @ Rm.T + tr
        w1, x1, y1, z1 = qr
        w2, x2, y2, z2 = quat.T
        q = np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)
        return out, q
    P2, Q2 = tf(P, Q)
    sgn = np.sign((Q2 * Qc).sum(1, keepdims=True))
    res = float(np.abs(P2 - Pc).max())
    if not pos_only:
        res = max(res, float(np.abs(Q2 * sgn - Qc).max()))
    return tf, res


def face_forward_transform(win0_pos, win0_quat):
    """The rigid xy rotation `_face_forward` applies to a window, as a function on (pos, quat) arrays."""
    w, x, y, z = win0_quat
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def f(pos, quat):
        P = np.concatenate([np.asarray(win0_pos, float)[None], np.asarray(pos, float)])
        Q = np.concatenate([np.asarray(win0_quat, float)[None], np.asarray(quat, float)])
        P2, Q2 = _face_forward(P, Q)
        return P2[1:], Q2[1:]
    return f, yaw


def run_branch(r, snap, cmd, offs):
    """Restore, run the command; record at policy-step offsets `offs` (1-based after the snapshot).

    Only the command changes: gains, integral state, policy and model are the snapshot's (= the source's)."""
    r.restore(snap)
    out, fell = [], False
    for k in range(1, offs[-1] + 1):
        rec = r.step(*cmd)
        if r.upright() < 0.5 or rec["base_pos"][2] < 0.3:
            fell = True
        if k in offs:
            out.append(rec)
    return {f: np.asarray([o[f] for o in out]) for f in FIELDS}, fell


def store_settings(split):
    """Add the rollout settings to branch files generated before they were stored (idempotent).

    Those files were run with kp 0.5 / ki 0 at the branch, whatever the source used; a file is only
    tagged if `locate` finds its source rolled out with exactly those gains, its policy and the model
    -- otherwise it was generated under a controller switch and must be regenerated (raises)."""
    import csv as _csv
    out = os.path.join(ROOT, f"data/egocentric_v3/b1_cf_branches_{split}")
    src_dir = os.path.join(ROOT, "data/egocentric_v3", f"beh24_b1_ego_flat_clean{split}")
    rows = [r for r in _csv.DictReader(open(os.path.join(out, "manifest.csv"))) if r["file"]]
    n = 0
    for src in sorted({r["source"] for r in rows}):
        todo = []
        for r in rows:
            if r["source"] != src:
                continue
            with np.load(os.path.join(out, r["file"]), allow_pickle=True) as f:
                if "heading_kp" not in f.files:
                    todo.append(r["file"])
        if not todo:
            continue
        loc = locate(os.path.join(src_dir, src))
        st = loc["settings"]
        if (st["heading_kp"], st["heading_ki"]) != (0.5, 0.0) or st["model"] != os.path.relpath(MODEL, ROOT):
            raise RuntimeError(f"{split}/{src}: source settings {st} differ from the pre-2026-10-01 branch "
                               f"generator's (kp 0.5, ki 0, {MODEL}); regenerate its branches, do not tag them")
        for fn in todo:
            path = os.path.join(out, fn)
            with np.load(path, allow_pickle=True) as f:
                data = {k: f[k] for k in f.files}
            if str(data["policy"]) != st["policy"]:
                raise RuntimeError(f"{fn}: policy {data['policy']} != source's {st['policy']}")
            data.update({k: np.array(v) for k, v in st.items() if k != "policy"})
            tmp = path[:-4] + ".settmp.npz"
            np.savez_compressed(tmp, **data)
            os.replace(tmp, path)
            n += 1
        print(f"  {split}/{src}: settings stored in {len(todo)} files {st}", flush=True)
    print(f"{split}: stored settings in {n} files", flush=True)


def seed_of(d):
    return int(re.search(r"seed (\d+)", str(d["render_note"])).group(1))


def code_of(ep, t, ci):
    return 100000000 + ep * 10000 + t * 100 + ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "heldout"), default="train")
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="start frames t (default 10 25 40 / val, heldout 15 35)")
    ap.add_argument("--prefix", type=int, default=10)
    ap.add_argument("--branch", type=int, default=20)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--check", action="store_true", help="1 source x 1 t x 3 commands, sanity checks, no output dir")
    ap.add_argument("--sources", nargs="*", default=None, help="restrict to these source basenames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--store_settings", action="store_true",
                    help="only add the rollout settings to existing branch files of --split (no rendering)")
    args = ap.parse_args()
    if args.store_settings:
        return store_settings(args.split)
    src_dir = os.path.join(ROOT, "data/egocentric_v3",
                           f"beh24_b1_ego_flat_clean{args.split}")
    steps = args.steps or ([10, 25, 40] if args.split == "train" else [15, 35])
    out = os.path.join(ROOT, args.out or f"data/egocentric_v3/b1_cf_branches_{args.split}")
    paths = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
    if args.sources:
        paths = [p for p in paths if os.path.basename(p) in args.sources]
    cmd_idx = list(range(len(NAMES)))
    if args.check:
        paths = [p for p in paths if os.path.basename(p) == "b1_ep2801.npz"] or paths[:1]
        steps = steps[:1]
        own = str(np.load(paths[0], allow_pickle=True)["condition"])
        cmd_idx = [NAMES.index(own), NAMES.index("speed_vx0.50"), NAMES.index("turn_w0.075_neg")]
        out = os.path.join(ROOT, "results/wm/cache/b1_cf_check")
    os.makedirs(out, exist_ok=True)
    for t in steps:
        if t < args.prefix or t + args.branch > 65:
            raise SystemExit(f"t={t} needs {args.prefix} frames before and {args.branch} after within 66")

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from counterfactual_readout_b1 import build_b1v3_scene
    from wm.data.embodiment import REGISTRY, heading, load
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    man_path = os.path.join(out, "manifest.csv")
    new_man = not os.path.exists(man_path)
    man = open(man_path, "a", newline="")
    mw = csv.writer(man)
    if new_man:
        mw.writerow(["file", "source", "source_episode", "source_condition", "policy", "t", "command_index",
                     "command_name", "vx", "vy", "wz", "n_frames", "branch_index", "fell", "start_corr"])
    t0 = time.time()
    n_done = n_fell = 0
    try:
        for si, p in enumerate(paths):
            d = np.load(p, allow_pickle=True)
            ep = int(d["expert_episode"])
            todo = [(t, ci) for t in steps for ci in cmd_idx
                    if not os.path.exists(os.path.join(out, f"b1_ep{code_of(ep, t, ci)}.npz"))]
            if not todo:
                continue
            loc = locate(p)
            if loc["tf_res"] > 1e-4:
                raise SystemExit(f"{p}: planar frame fit residual {loc['tf_res']:.1e}")
            seed = seed_of(d)
            fr_src = d["frames"]
            cpos, cquat = d["base_pos"].astype(np.float64), d["base_quat"].astype(np.float64)
            pose = build_b1v3_scene(sim, args.scene, seed, float(heading(cquat[:1], "b1")[0]))
            spawn = cpos[0, :2].copy()
            render = lambda P, Q, J: pose(np.concatenate([P[:2] - spawn, P[2:]]), Q, J)  # noqa: E731
            print(f"[{si + 1}/{len(paths)}] {os.path.basename(p)} {d['condition']} pol={loc['policy']} "
                  f"window_start={loc['off']} gains={loc['gains']} model={loc['settings']['model']} reproduce max|dq|={loc['err']:.1e} pose-frame residual={loc['tf_res']:.1e} seed={seed}", flush=True)
            for t in steps:
                s_t = loc["off"] + KEEP[t]
                r, rec = rollout_to(loc, s_t)
                tf = loc["tf"]
                # the reproduced start state, in the clip's frame, against the stored clip
                P0, Q0 = tf(rec[s_t]["base_pos"][None], rec[s_t]["base_quat"][None])
                err_pos = float(np.abs(P0[0] - cpos[t]).max())
                snap = r.snapshot()
                img0 = render(P0[0], Q0[0], rec[s_t]["joint_pos"])
                corr0 = float(np.corrcoef(img0.ravel().astype(float), fr_src[t].ravel().astype(float))[0, 1])
                offs = [int(v) for v in np.round(2.5 * (t + np.arange(1, args.branch + 1))) - KEEP[t]]
                print(f"   t={t}: start pos err {err_pos:.1e}  start-frame pixel corr vs v3 {corr0:.4f}", flush=True)
                if corr0 < 0.99:
                    raise SystemExit("start frame does not match the source's v3 frame")
                for ci in [c for (tt, c) in todo if tt == t]:
                    name = NAMES[ci]
                    cmd = cond_cmd(name, loc["policy"])
                    br, fell = run_branch(r, snap, cmd, offs)
                    if fell:
                        n_fell += 1
                        mw.writerow(["", os.path.basename(p), ep, str(d["condition"]), loc["policy"], t, ci,
                                     name, *cmd, 0, args.prefix, 1, corr0]); man.flush()
                        print(f"     FELL {name}", flush=True)
                        continue
                    P, Q = tf(br["base_pos"], br["base_quat"])
                    frames = [render(P[i], Q[i], br["joint_pos"][i]) for i in range(len(P))]
                    a = t - args.prefix
                    data = {k: np.concatenate([d[k][a:t + 1], v]).astype(np.float32) for k, v in
                            (("base_pos", P), ("base_quat", Q), ("joint_pos", br["joint_pos"]),
                             ("joint_vel", br["joint_vel"]), ("action", br["action"]),
                             ("command", br["command"]), ("foot_contact", br["foot_contact"]))}
                    data["frames"] = np.concatenate([fr_src[a:t + 1], np.asarray(frames, np.uint8)])
                    # CoM per frame (F301; the loader's Froude reference): subtree CoM of the trunk from the
                    # stored (clip-frame) poses, prefix and branch alike
                    data["com_pos"] = b1_com(data["base_pos"], data["base_quat"], data["joint_pos"],
                                             os.path.join(ROOT, str(loc["settings"]["model"]))).astype(np.float32)
                    _, beh, lvl, _, _, _, _ = COND[name]
                    code = code_of(ep, t, ci)
                    fn = f"b1_ep{code}.npz"
                    np.savez_compressed(
                        os.path.join(out, fn), frames=data["frames"], joint_order_sdk=d["joint_order_sdk"],
                        dt=d["dt"], fps=d["fps"], **{k: data[k] for k in FIELDS}, com_pos=data["com_pos"],
                        condition=np.array(name), behaviour=np.array(beh), level=np.array(lvl),
                        expert_episode=np.array(code), embodiment=np.array("b1"), policy=np.array(loc["policy"]),
                        render_note=np.array(f"build_b1_cf_branches: v3 matched scene, seed {seed}"),
                        cf_source=np.array(os.path.basename(p)), cf_source_episode=np.array(ep),
                        cf_source_condition=np.array(str(d["condition"])), cf_t=np.array(t),
                        cf_branch_index=np.array(args.prefix), cf_command=np.array(cmd, np.float32),
                        **{k: np.array(v) for k, v in loc["settings"].items() if k != "policy"})
                    mw.writerow([fn, os.path.basename(p), ep, str(d["condition"]), loc["policy"], t, ci, name,
                                 *cmd, len(data["frames"]), args.prefix, 0, corr0]); man.flush()
                    n_done += 1
                el = time.time() - t0
                print(f"   t={t} done; {n_done} branches, {n_fell} fell, {el / 60:.1f} min", flush=True)
    finally:
        man.close()
        sim.stopSimulation()

    if args.check:
        check(out, paths[0], steps[0], cmd_idx, args.prefix, NAMES, REGISTRY, load)
    print(f"CF_DONE {n_done} branches, {n_fell} fell, {(time.time() - t0) / 60:.1f} min", flush=True)


def check(out, src, t, cmd_idx, prefix, names, REGISTRY, load):
    spec = REGISTRY["b1"]
    d = np.load(src, allow_pickle=True)
    ep = int(d["expert_episode"])
    files = [os.path.join(out, f"b1_ep{code_of(ep, t, ci)}.npz") for ci in cmd_idx]
    F = [np.load(f, allow_pickle=True) for f in files]
    f0 = [x["frames"][prefix] for x in F]
    print("\n=== check 1: start frame")
    print("  start frame identical across commands:", all(np.array_equal(f0[0], f) for f in f0[1:]))
    print(f"  start frame vs source v3 frame t: max |diff| {np.abs(f0[0].astype(int) - d['frames'][t].astype(int)).max()}"
          f" (prefix frames are copied; branch frame 0 is the copied v3 frame; rendered-state corr in manifest)")
    print("=== check 2: own-command branch vs source")
    own = F[0]
    n = len(own["frames"])
    a = t - prefix
    for k in ("joint_pos", "base_pos", "base_quat", "action", "command", "foot_contact"):
        print(f"  {k:<13} max|diff| {np.abs(own[k].astype(float) - d[k][a:a + n].astype(float)).max():.2e}")
    pix = [np.corrcoef(own["frames"][i].ravel().astype(float), d["frames"][a + i].ravel().astype(float))[0, 1]
           for i in range(n)]
    print(f"  rendered branch frames vs source v3 frames: pixel corr min {min(pix):.4f}")
    bm_own = load(files[0], spec)["body_motion"]
    bm_src = load(src, spec)["body_motion"][a:a + n]
    diff = np.abs(bm_own - bm_src)
    print(f"  body_motion (fwd, lat, yaw) max|diff| all {n} frames: {diff.max(0).round(5)}; "
          f"frames {prefix}..{prefix + 9} (pairs from the branch point): {diff[prefix:prefix + 10].max(0).round(5)}")
    print("=== check 3: loader + dataset pair")
    from wm.data.dataset import MultiEmbodimentPairs
    import inspect
    sig = inspect.signature(MultiEmbodimentPairs.__init__)
    print("  MultiEmbodimentPairs args:", list(sig.parameters)[:12])
    for f in files:
        c = load(f, spec)
        print(f"  {os.path.basename(f)}: frames {c['frames'].shape} actions {c['actions'].shape} "
              f"body_motion {c['body_motion'].shape} cond {np.load(f)['condition']}")
    try:
        ds = MultiEmbodimentPairs([(files, "b1")], frame_stride=5, action_chunk=5)
        s = ds[0]
        print(f"  dataset: {len(ds)} pairs; sample keys {sorted(s)}")
    except Exception as e:  # report, the caller adapts the call
        print("  dataset build failed:", repr(e))
    bm = np.stack([load(f, spec)["body_motion"][prefix:prefix + 5].mean(0) for f in files])
    print("  Froude over pair (branch, branch+5) per command:")
    for ci, b in zip(cmd_idx, bm):
        print(f"    {names[ci]:<16} {b.round(4)}")


if __name__ == "__main__":
    main()
