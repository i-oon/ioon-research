"""Generic CPG babble for B1 -- no trained policy, no B1-tuned constants.

    .venv/bin/python3 sim/collect/collect_b1_cpg_babble.py --freq 1.5 --out data/b1_babble/b1_1.npz
    .venv/bin/python3 sim/collect/collect_b1_cpg_babble.py --freq 1.5 --noise 0.05 --seed 1 \\
        --out data/b1_babble/b1_2.npz

**Why this file exists and `recollect_b1_noisy.py` cannot answer the same question.** Q21
(`doc/OPEN_QUESTION.md`) asks whether motor babble can replace the recorded-expert-clip candidate
library the closed loop currently scores (F188). `recollect_b1_noisy.py` adds noise to the TRAINED
PPO POLICY's own output -- it still needs a working controller, which is exactly what a genuinely
unseen body does not have. This file drives B1 with the same kind of open-loop, no-controller CPG
used for gecko (`collect_gecko_cpg.py`/`collect_gecko_dataset.py`), so the test is honest: can babble
alone, with no policy in the loop at all, produce usable candidates.

**"Generic" is a constraint on this file, not a description of it -- read before changing constants.**
The whole point of the test is "what would babbling a body we know nothing about produce", so this
CPG must not be tuned to B1's known dynamics the way gecko's was tuned to gecko's (gecko's FREQ=4.0,
DUTY_BIAS=-0.06 etc. are fixes for gecko-specific problems found by measuring gecko's own gait and
must NOT be copied here). What IS reused is the *functional form* gecko settled on -- a duty-cycle
swing/lift decomposition, diagonal-pair phasing -- because that shape (fast swing in the air, slow
push while planted) is a generic property of walking gaits, not a B1-specific fit. Frequency and
amplitude below are starting guesses sized only from the joint range (`ACTION_SCALE`) and a
diagonal trot's usual cadence, verified only for "does it walk without falling", never adjusted to
match B1's own measured Froude the way gecko's calibration did. If that temptation comes up, stop --
tuning to match B1 defeats the purpose of testing whether GENERIC babble is enough.

**A separate, NOT-contradictory rule (measured 2026-09-11, F194 follow-up): before sweeping ANY
parameter here, measure the GOAL-SOURCE body's own Froude range per family first.** This is not
"tuning to B1's own dynamics" (which stays forbidden, see above) -- it is calibrating collection
EFFORT toward the actual goal vocabulary this pool will be scored against (hexapod's
`beh12_c10f10t10`: forward in [0.12, 0.19], lateral in [-0.12, 0.07], no yaw-dominant example
exists there at all), which any real deployment would also need to know. Measured directly:
expert candidates have 0% chance of a cross-family nearest-neighbour in predicted-Froude space
(perfectly separated); this file's babble pool had 19% of candidates sitting at a ZERO-margin
family boundary (gap 0.0005, identical to the "safe" same-family gap) -- coverage of the right
RANGE is not sufficient on its own, margin AT the boundary between families matters just as much.
Concretely: sweep parameters to land candidates confidently inside a target family's known range,
and explicitly avoid the weak, near-zero-in-every-channel region that produces ambiguous,
barely-dominant motion -- that region is exactly where a small vision-read noise flips a
selection across a family boundary. After collecting, measure this directly (fraction of
candidates whose predicted-Froude nearest neighbour is a different family, and the gap there)
rather than assuming range coverage alone fixes it.

**Correction, v2 attempt, measured 2026-09-11: checking margin in TRUE Froude (`body_motion`) is
NOT sufficient, and a batch built on that check alone measurably did not help.** Pushed
`--strafe_amp` up specifically because the true-motion margin looked much better on a sample --
the actual cross-family-nearest-neighbour risk, measured through a FITTED checkpoint's PREDICTED
Froude (`body_head(proj(actions))`), came out identical to the unfixed batch (19% both times,
zero improvement), and one closed-loop cell regressed hard. The projector/body_head does not
preserve true-space distances linearly (this project's own F110: "direction right, extent
wrong"). **The real check has to go through a fitted model's prediction, not the raw recorded
motion** -- collect a sample, fit (or reuse) a checkpoint, then measure margin in predicted space
before trusting a parameter change.

**Frames are a placeholder, not a bug.** `wm/data/embodiment.py`'s `_b1` reader requires a `frames`
key to exist (it is unconditionally re-exposed in the returned dict), but `DirectFroudePlanner`
(the mechanism this candidate pool is built to feed, F188's mode A/D) scores candidates from
`actions` alone -- `body_head(proj(a))`, no vision, no `e_t`. Storing real rendered video for a
candidate pool that is never watched wastes CoppeliaSim time this test does not need; the array
below is shaped correctly (uint8, `(steps, 4, 4, 3)`) so nothing downstream breaks on `.shape`
access, but it is NOT real video and must never be treated as such (e.g. do not build a
three-panel comparison clip from this file's `frames`).

**Diagonal pairing and IL joint order, confirmed from `rollout_b1_mujoco.py`, not assumed.**
`IL_TO_SDK` there maps to SDK order `[FR_hip,FR_thigh,FR_calf, FL_..., RR_..., RL_...]` per its own
`joint_order_sdk` comment; tracing it back gives IL order (used for `action` and `DEFAULT_IL`) as
`[hip_FL,hip_FR,hip_RL,hip_RR, thigh_FL,thigh_FR,thigh_RL,thigh_RR, calf_FL,calf_FR,calf_RL,calf_RR]`
-- i.e. leg index 0=FL, 1=FR, 2=RL, 3=RR within each joint-type group of 4.

**The model file matters more than any CPG parameter, and this cost real debugging time to find.**
The first version of this script used `b1_flat.xml`, which has UNIFORM placeholder physics on
every joint (`armature=0.1 damping=0.5 frictionloss=0.2`, actuator gains flat `kp=300 kv=5` on
hip/thigh/calf alike, despite very different torque needs -- calf's range is +-140 N*m against
hip's +-91). The trained PPO policy walks fine on that file because RL adapted around whatever
dynamics it has; an untrained, hand-designed CPG has no such adaptation and is fully exposed to
them. Measured: on `b1_flat.xml`, EVERY sign/phase/frequency/duty combination tried (8+ configs)
walked backward, and even holding the default standing pose with NO gait command at all drifted
-0.08 m over 500 settle steps -- a real passive asymmetry, not a CPG bug. Switching to
`b1_flat_real.xml` (system-identified per-joint parameters from
`motor_param_real_30seeds_full.txt`, real gains `kp700/kd3` thigh, `kp970/kd3` calf) cut the same
passive drift to -0.017 m and, with the identical CPG logic, turned consistent backward walking
into **+1.17 m forward** over the same 100 steps. Use `b1_flat_real.xml` for anything without a
trained policy in the loop; `b1_flat.xml` is validated only together with the trained PPO policy.

**Swing sign and `--phase_lag` were swept empirically on `b1_flat_real.xml`, not derived.** `-pi/2`
is what walked forward (`+pi/2` walked backward, `-0.35m`); this is reading off a black-box
motor/geometry convention, exactly the kind of thing gecko's own `-(body y)` axis fix (F189)
needed empirical measurement for rather than a derivation.
"""
import argparse
import os

import numpy as np
import mujoco

MODEL = "/home/aria/Sim2Real-B1/b1_ws/src/b1_mujoco/model/b1_flat_real.xml"
DEFAULT_IL = np.array([0.061, -0.066, 0.058, -0.054,
                       1.064,  1.060, 1.077,  1.068,
                      -1.914, -1.935, -1.914, -1.913])
ACTION_SCALE = 0.25
DECIMATION = 4
SPAWN_Z = 0.50
FALL_HEIGHT = 0.35   # rollout_b1_mujoco.py's own "WALKS" criterion, reused verbatim

LEG_ORDER = ["FL", "FR", "RL", "RR"]
LEFT_LEGS = {"FL", "RL"}
FRONT_LEGS = {"FL", "FR"}
# diagonal trot, generic: (FL,RR) together, (FR,RL) half a cycle apart
PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
IL_TO_SDK = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8])


def il_to_sdk(v):
    out = np.empty(12)
    out[IL_TO_SDK] = np.asarray(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=66, help="matches this project's usual clip length")
    ap.add_argument("--dt_decimation", type=int, default=DECIMATION)
    ap.add_argument("--warmup", type=int, default=25, help="settle steps before the CPG starts")
    ap.add_argument("--freq", type=float, default=1.5, help="gait frequency, Hz -- a GENERIC "
                    "starting guess sized from the diagonal-trot literature range for a "
                    "quadruped this size, not fit to B1's own measured gait. Sweep this to find "
                    "a config that walks; do not tune it to match B1's known Froude.")
    ap.add_argument("--thigh_amp", type=float, default=1.0, help="swing amplitude, action units "
                    "(radians / ACTION_SCALE convention, +-1 matches a trained policy's tanh "
                    "output range). Verified to walk without falling at this value; not tuned to "
                    "match B1's own measured Froude.")
    ap.add_argument("--calf_amp", type=float, default=1.0, help="lift amplitude, action units")
    ap.add_argument("--phase_lag", type=float, default=-np.pi / 2, help="knee-vs-hip phase lag, "
                    "radians. Swept empirically on b1_flat_real.xml: -pi/2 walks forward "
                    "(+1.17 m/100 steps), +pi/2 walks backward (-0.35 m/100 steps) -- see module "
                    "docstring. Do not change sign without re-measuring x_travel.")
    ap.add_argument("--noise", type=float, default=0.0, help="per-step Gaussian noise on the "
                    "commanded action (babble mode)")
    ap.add_argument("--bias", type=float, default=0.0, help="per-rollout constant thigh-amplitude "
                    "offset, symmetric across left/right -- same role as gecko's --bias")
    ap.add_argument("--turn_bias", type=float, default=0.0, help="per-rollout asymmetric bias, "
                    "LEFT legs +turn_bias, RIGHT legs -turn_bias -- differential drive, same "
                    "mechanism as gecko's --turn_bias. Measured (F189-of-this-branch): this alone "
                    "never makes yaw the DOMINANT channel over forward, only shifts drift within a "
                    "forward-dominant clip -- use --pivot_amp for genuine turning.")
    ap.add_argument("--strafe_amp", type=float, default=0.0, help="hip (ab/adduction) oscillation "
                    "amplitude, action units. GENERIC STRAFING PRIMITIVE, added because forward-only "
                    "babble measured 100% forward-family with zero lateral coverage. Uses the SAME "
                    "duty-cycle swing/lift SHAPE as forward walking, applied to the hip channel "
                    "instead of thigh, same sign for every leg (not left/right differential -- "
                    "strafing pushes the whole body one lateral direction, unlike a turn). Sign "
                    "must be verified empirically the same way --phase_lag was (see module "
                    "docstring); do not assume positive means 'right'.")
    ap.add_argument("--pivot_amp", type=float, default=0.0, help="GENUINE turning, on the HIP "
                    "channel (ab/adduction), not thigh. A thigh-swing differential was tried first "
                    "(sign-flip LEFT legs, then a full pivot_sign*ph phase reversal) and BOTH are "
                    "mathematically degenerate for this diagonal trot, for the same underlying "
                    "reason: any two legs being split into two groups (LEFT/RIGHT, or FRONT/REAR) "
                    "always have base phases exactly pi apart in a 2-phase diagonal trot, so "
                    "negating one group's OWN-PHASE signal lands exactly on sin(ph+pi)=-sin(ph) --"
                    "i.e. relabels which physical leg sits in which diagonal group, zero net "
                    "asymmetry BY CONSTRUCTION (measured: yaw stayed ~0, then ~equal to forward and "
                    "noisy/oscillating -- +22deg over 6s but wobbling -2 to +33deg along the way, "
                    "not a clean turn -- because the underlying signal was still built from each "
                    "leg's own trot phase). Fixed properly (not patched again) by moving the "
                    "differential to hip, front-vs-rear, ONE side only (FL positive, RL negative, "
                    "RIGHT hip untouched) -- this creates a real yaw torque (front leg pushes one "
                    "lateral way, rear leg on the same side pushes the other way, moment arms add) "
                    "independent of the thigh/diagonal-trot machinery entirely, so thigh+calf keep "
                    "walking unmodified underneath. Still keyed to the GLOBAL clock, not each leg's "
                    "own phase -- see swing's docstring above, the same degeneracy applies to hip "
                    "too. Fraction of --thigh_amp used as the hip amplitude.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    adr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): m.sensor_adr[i]
          for i in range(m.nsensor)}

    d.qpos[0:3] = [0, 0, SPAWN_Z]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:19] = il_to_sdk(DEFAULT_IL)
    d.ctrl[:] = il_to_sdk(DEFAULT_IL)
    mujoco.mj_forward(m, d)
    for _ in range(args.warmup * args.dt_decimation):
        mujoco.mj_step(m, d)

    rng = np.random.default_rng(args.seed)
    base_pos, base_quat, joint_pos, actions, foot_contact = [], [], [], [], []
    phase_t = dict(PHASE)
    real_dt = args.dt_decimation * m.opt.timestep
    t_elapsed = 0.0

    for _ in range(args.steps):
        action = np.zeros(12, np.float32)
        global_theta = 2 * np.pi * args.freq * t_elapsed
        for li, leg in enumerate(LEG_ORDER):
            leg_bias = args.bias + (args.turn_bias if leg in LEFT_LEGS else -args.turn_bias)
            swing_amp = max(0.02, args.thigh_amp + leg_bias)
            # Standard hip/knee sine CPG: an elliptical foot trajectory from a 90-degree knee-vs-hip
            # phase lag, no duty-cycle parameter needed. Signs verified empirically -- see module
            # docstring and --phase_lag's help. Each leg keeps its OWN diagonal-trot phase here --
            # do not touch this for turning, see --pivot_amp's help for why that is degenerate.
            ph = 2 * np.pi * phase_t[leg]
            swing = -swing_amp * np.sin(ph)        # unmodified trot -- pivot no longer touches this
            lift = max(0.0, args.calf_amp * np.sin(ph + args.phase_lag))
            # strafe: same duty-cycle SHAPE as forward swing, on the hip channel, SAME sign every
            # leg (pushes the whole body one lateral direction rather than turning it)
            hip = -args.strafe_amp * np.sin(ph) if args.strafe_amp else 0.0
            if args.pivot_amp and leg in LEFT_LEGS:
                # genuine turning torque: hip channel, ONE side only, front vs rear opposite sign.
                # Front leg pushes the body laterally one way, rear leg on the SAME side pushes the
                # other way -- opposite forces at opposite fore-aft moment arms ADD into a yaw
                # torque instead of cancelling (unlike a thigh-swing differential, see --pivot_amp's
                # help for why that degenerates). Tied to the GLOBAL clock, not this leg's own trot
                # phase -- same reason as above: FL and RL's own phases are pi apart, so an own-
                # phase sign-flip here would degenerate exactly the same way.
                fr_sign = 1.0 if leg in FRONT_LEGS else -1.0
                hip += args.pivot_amp * args.thigh_amp * fr_sign * (-np.sin(global_theta))
            action[li] = hip                       # hip: strafe and/or pivot differential
            action[4 + li] = swing                 # thigh: fore-aft swing, always the plain trot
            action[8 + li] = lift                  # calf: lifts to clear the foot during swing
            if args.noise > 0.0:
                action[li] += rng.normal(0, args.noise)
                action[4 + li] += rng.normal(0, args.noise)
                action[8 + li] += rng.normal(0, args.noise)
        phase_t = {leg: phase_t[leg] + args.freq * real_dt for leg in LEG_ORDER}
        t_elapsed += real_dt

        target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * action)
        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(args.dt_decimation):
            mujoco.mj_step(m, d)

        touch = np.asarray(d.sensordata[adr["FR_touch"]:adr["FR_touch"] + 4])
        base_pos.append(d.qpos[0:3].copy())
        base_quat.append(d.qpos[3:7].copy())
        joint_pos.append(d.qpos[7:19].copy())
        actions.append(action.copy())
        foot_contact.append((touch > 1.0).astype(np.float32))

    bp = np.asarray(base_pos, np.float32)
    fell = bp[:, 2].min() < FALL_HEIGHT
    disp = bp[-1, :2] - bp[0, :2]
    print(f"steps={len(bp)}  x_travel={disp[0]:+.3f}m  y_drift={disp[1]:+.3f}m  "
         f"min_z={bp[:, 2].min():.3f}  {'FELL' if fell else 'walked'}")
    if fell:
        print("  FELL -- discarding, nothing saved")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = len(bp)
    placeholder_frames = np.zeros((n, 4, 4, 3), np.uint8)  # see module docstring: never real video
    np.savez_compressed(
        args.out,
        frames=placeholder_frames,
        base_pos=bp, base_quat=np.asarray(base_quat, np.float32),
        joint_pos=np.asarray(joint_pos, np.float32),
        action=np.asarray(actions, np.float32),
        foot_contact=np.asarray(foot_contact, np.float32),
        dt=np.float32(real_dt), expert_episode=np.int64(0),
        # same convention as rollout_b1_mujoco.py -- render_b1_replay.py's --out packaging reads
        # this key straight through
        joint_order_sdk=np.array([f"{l}_{s}_joint" for l in ("FR", "FL", "RR", "RL")
                                  for s in ("hip", "thigh", "calf")]),
        condition=f"cpgbabble_f{args.freq}_t{args.turn_bias}_st{args.strafe_amp}_pv{args.pivot_amp}_s{args.seed}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
