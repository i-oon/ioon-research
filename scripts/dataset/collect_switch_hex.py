"""Hexapod (c10f10t10) SWITCHING clips: the beh24 behaviour changes every 15-25 frames inside a clip.

**Why.** Every beh24 clip runs one condition for 66 frames, so the current state predicts the next
action. Here the same body state is followed by different commands -- the hexapod counterpart of
`data/egocentric_v3/b1_cf_branches_train` (counterfactual-style data, F252/F266).

**Mechanism.** One continuous physics run per clip, driven by `collect_ik.py --plan`: every CPG drive
(pace, spin, strafe, lead, a0-a2, ft_phase) is a per-frame array, the oscillator phase is integrated
(`cumsum(pace)`), so a switch changes the command from the CURRENT physical state -- no reset, no
teleport. Each switch is ramped linearly over `--ramp` frames centred on the boundary.
**`--schedule` is NOT used**: it retimes the recorded foot path and moved the CPG bias pose (F152 note).

**Each segment is the exact beh24 command of its condition**, fitted from the stored beh24 actions
(see BWD_CYCLES below; the recipe file is not what the canonical clips ran): cycles c -> pace c/8.8 on
a base of 8.8 cycles; turns and sideways 6 cycles; bwd = lead 0.75; sideways = a0 0, a2 0.3,
ft_phase 0.5 with its yaw-cancelling spin. beh24 walks on the raw
centre pose and walks sideways on the left-right symmetrised one (`--symmetric`), 0.134 rad apart;
the plan key `sym` (0 walk, 1 sideways, ramped) blends the two so each segment has its own recipe's
pose. Centre = the bias pose FITTED from a beh24 speed_c7.1 clip's commands (`centre_pose`; its plain
mean is off by up to 0.0035 rad because 7.1 cycles is not a whole number), handed to `--centre_from`
as a one-row npz. No expert CSV.

Stored per clip (flat dir, `hexapod_ep<N>.npz`): every collect_ik field (frames, actions, forces,
head, body_quat, foot_order, step_idx, morph, expert_episode, ...) + `plan_*` (per-frame drives) +
`segment` (int per frame, +1 at each switch; `wm.data.embodiment.smooth` never averages across it),
`seg_start`, `seg_len`, `seg_condition`, `schedule` ("cond@t0:t1 ..."), `frame_condition`,
`condition`="switch", `behaviour`="switch", `level`=-1, `embodiment`="hexapod", `ego_seed`.

    .venv/bin/python3 scripts/dataset/collect_switch_hex.py --clips 2 --out /tmp/x --pilot
    .venv/bin/python3 scripts/dataset/collect_switch_hex.py --clips 48 --seed 7 --ep0 20000 \\
        --out data/egocentric/beh24_c10f10t10_switch_train
"""
import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EP = 66
BASE = 8.8
CENTRE = "data/egocentric/beh24_c10f10t10_ego_flat/hexapod_ep100.npz"     # speed_c7.1, raw pose
WALK = dict(a0=0.25, a1=0.20, a2=0.20, ft_phase=0.0, strafe=0.0, sym=0.0)
SIDE = dict(a0=0.00, a1=0.20, a2=0.30, ft_phase=0.5, lead=0.25, sym=1.0)
KEYS = ("pace", "spin", "strafe", "lead", "a0", "a1", "a2", "ft_phase", "sym")
COMMON = ["--gait", "cpg", "--scale", "0.65", "--cam_dx", "-0.6", "--behavior", "switch",
          "--cycles", str(BASE), "--spin_amp", "0.25", "--view", "egocentric", "--travel", "0"]


# **Fitted from the stored beh24 commands, not taken from collect_beh24.py's recipe.** Every
# condition of data/egocentric/beh24_c10f10t10_ego_flat was fitted as bias + sinusoids of
# 2*pi*cycles*t/66 (residual 0.0000 rad on all 24); the canonical clips differ from the recipe file:
# turn_sX runs spin -X (and _neg +X), backward runs 3.49/3.90/4.97/5.05 cycles (not 5.8-8.8), and the
# recalibrated sideways strafes are L -0.48/-0.83/-1.30/-1.40, R +0.488/+0.73/+0.83/+0.87.
BWD_CYCLES = {5.8: 3.49, 7.1: 3.90, 8.15: 4.97, 8.8: 5.05}
SIDE_STRAFE = {"L": (-0.48, -0.83, -1.30, -1.40), "R": (0.488, 0.73, 0.83, 0.87)}


def conditions():
    """The 24 beh24 conditions as plan drives, with their family."""
    c = {}
    for cyc in (5.8, 7.1, 8.15, 8.8):
        c[f"speed_c{cyc:g}"] = ("fwd", dict(WALK, pace=cyc / BASE, lead=0.25, spin=0.0))
        c[f"speed_c{cyc:g}_bwd"] = ("bwd", dict(WALK, pace=BWD_CYCLES[cyc] / BASE, lead=0.75, spin=0.0))
    for v in (0.05, 0.15, 0.29, 0.56):
        n = f"turn_s{v:.2f}".rstrip("0").rstrip(".")
        c[n] = ("turn", dict(WALK, pace=6.0 / BASE, lead=0.25, spin=-v))
        c[n + "_neg"] = ("turn_neg", dict(WALK, pace=6.0 / BASE, lead=0.25, spin=v))
    for lvl in range(4):
        c[f"side_L_lvl{lvl}"] = ("sideL", dict(SIDE, pace=6.0 / BASE, strafe=SIDE_STRAFE["L"][lvl], spin=0.19))
        c[f"side_R_lvl{lvl}"] = ("sideR", dict(SIDE, pace=6.0 / BASE, strafe=SIDE_STRAFE["R"][lvl], spin=-0.24))
    return c


COND = conditions()


def centre_pose(work):
    """bias + sinusoids fit of CENTRE's commands (exact, residual ~0); writes a --centre_from npz."""
    with np.load(os.path.join(ROOT, CENTRE), allow_pickle=True) as d:
        a = d["actions"].astype(np.float64)
    t = np.arange(len(a))
    ph = 2 * np.pi * 7.1 * t / EP
    M = np.stack([np.ones(len(a)), np.sin(ph), np.cos(ph)], 1)
    coef = np.linalg.lstsq(M, a, rcond=None)[0]
    assert np.abs(M @ coef - a).max() < 1e-4, "centre clip is not a 7.1-cycle CPG clip"
    os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "centre_pose.npz")
    # **written atomically**: with several --ports, worker threads call this concurrently, and a plain
    # np.savez rewrite let collect_ik read a half-written file (BadZipFile, 2026-10-01 train extension)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    np.savez(tmp, actions=coef[0][None, :])
    os.replace(tmp, path)
    return path


def lengths(rng):
    n = int(rng.choice([3, 4]))
    while True:
        L = rng.integers(15, 26, size=n)
        if L.sum() == EP:
            return [int(x) for x in L]


def make_schedules(n_clips, seed):
    """Segment lengths + conditions for every clip, balancing condition counts and ordered family pairs."""
    rng = np.random.default_rng(seed)
    use = {k: 0 for k in COND}
    pairs = {}
    out = []
    for _ in range(n_clips):
        L, seq = lengths(rng), []
        for j in range(len(L)):
            cands = [k for k in COND if not seq or COND[k][0] != COND[seq[-1]][0]]
            def cost(k):
                pc = pairs.get((COND[seq[-1]][0], COND[k][0]), 0) if seq else 0
                return (use[k], pc, rng.random())
            k = min(cands, key=cost)
            if seq:
                fp = (COND[seq[-1]][0], COND[k][0]); pairs[fp] = pairs.get(fp, 0) + 1
            use[k] += 1
            seq.append(k)
        out.append((L, seq))
    return out, use, pairs


def make_plan(L, seq, ramp):
    plan = {k: np.zeros(EP) for k in KEYS}
    seg = np.zeros(EP, np.int64)
    t = 0
    for i, (n, c) in enumerate(zip(L, seq)):
        for k in KEYS:
            plan[k][t:t + n] = COND[c][1][k]
        seg[t:t + n] = i
        t += n
    starts = np.cumsum([0] + L[:-1])
    for t0 in starts[1:]:                                      # linear ramp centred on each switch
        lo, hi = t0 - ramp // 2, t0 + ramp // 2
        for k in KEYS:
            a, b = plan[k][lo - 1], plan[k][hi]
            plan[k][lo:hi] = np.linspace(a, b, hi - lo + 2)[1:-1]
    return plan, seg, starts


def collect(i, L, seq, args, ep, port):
    plan, seg, starts = make_plan(L, seq, args.ramp)
    work = os.path.join(args.work, f"clip{ep}")
    os.makedirs(work, exist_ok=True)
    pf = os.path.join(work, "plan.json")
    with open(pf, "w") as fh:
        json.dump({k: v.tolist() for k, v in plan.items()}, fh)
    ego_seed = args.ego_seed0 + i
    cmd = [sys.executable, os.path.join(ROOT, "sim", "collect", "collect_ik.py"),
           "--port", str(port), "--morphs", "c10f10t10=medauroidea_c10f10t10.ttt",
           "--episodes", str(ep), "--ego_seed", str(ego_seed), "--plan", pf, "--out", work,
           "--centre_from", centre_pose(args.work)] + COMMON
    with open(os.path.join(work, "collect.log"), "w") as log:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    src = glob.glob(os.path.join(work, "c10f10t10_*.npz"))[0]
    with np.load(src, allow_pickle=True) as d:
        rec = {k: d[k] for k in d.files}
    if len(rec["frames"]) != EP:
        raise RuntimeError(f"clip {ep}: {len(rec['frames'])} frames, want {EP}")
    ends = np.r_[starts[1:], EP]
    rec.update(segment=seg, seg_start=np.asarray(starts, np.int64), seg_len=np.asarray(L, np.int64),
               seg_condition=np.array(seq), seg_family=np.array([COND[c][0] for c in seq]),
               frame_condition=np.array([seq[s] for s in seg]),
               schedule=" ".join(f"{c}@{a}:{b}" for c, a, b in zip(seq, starts, ends)),
               condition="switch", behaviour="switch", level=-1, embodiment="hexapod",
               ego_seed=ego_seed, ramp=args.ramp, centre_from=CENTRE)
    dst = os.path.join(ROOT, args.out, f"hexapod_ep{ep}.npz")
    np.savez_compressed(dst, **rec)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7, help="schedule seed")
    ap.add_argument("--ep0", type=int, default=20000, help="expert_episode (= loader group) of clip 0")
    ap.add_argument("--ego_seed0", type=int, default=7000, help="room randomisation seed of clip 0")
    ap.add_argument("--ramp", type=int, default=4)
    ap.add_argument("--ports", type=int, nargs="+", default=[23000],
                    help="one CoppeliaSim per port; clips are spread over them")
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="clip indices to (re)collect")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    args.work = args.work or os.path.join(ROOT, args.out + "_work")
    sched, use, pairs = make_schedules(args.clips, args.seed)
    fams = sorted({f for f, _ in COND.values()})
    print(f"{args.clips} clips, {sum(len(L) for L, _ in sched)} segments; condition counts "
          f"{min(use.values())}-{max(use.values())}; ordered family pairs covered "
          f"{len(pairs)}/{len(fams) * (len(fams) - 1)}", flush=True)
    print("family pairs: " + " ".join(f"{a}>{b}:{n}" for (a, b), n in sorted(pairs.items())), flush=True)
    todo = [i for i in range(args.clips) if args.only is None or i in args.only]
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def job(slot_i):
        slot, i = slot_i
        L, seq = sched[i]
        ts = time.time()
        p = collect(i, L, seq, args, args.ep0 + i, args.ports[slot])
        print(f"clip {i:3d} ep{args.ep0 + i} {time.time() - ts:5.0f}s  "
              + " ".join(f"{c}[{n}]" for c, n in zip(seq, L)), flush=True)
        return p

    # one worker per port, each walking its own share of the clips
    shares = [[(s, i) for i in todo[s::len(args.ports)]] for s in range(len(args.ports))]
    with ThreadPoolExecutor(len(args.ports)) as ex:
        list(ex.map(lambda sh: [job(x) for x in sh], shares))
    print(f"done {len(todo)} clips in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
