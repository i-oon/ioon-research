"""Can the teacher rank actions, and at which scale? The characterisation of F144's failure.

    .venv/bin/python3 scripts/diagnostics/teacher_label_quality.py

**F144 is not reopened by this.** The bar was pre-registered and failed; this asks only *why*, so
the negative result can be written with a mechanism instead of a guess.

Two questions, deliberately at different scales:

  **coarse**  candidates are the recorded actions of the twelve conditions. Does the teacher pick
              one from the goal's own behaviour family? F143 measured 85-90% inside `adapt3`; this
              repeats it through the *labelling* path so the two numbers are comparable.

  **local**   candidates are Gaussian perturbations of the student's own action -- what the F144
              teacher actually ranked. **Judged in the simulator, not by the model**: the picked
              action and the student's own are each executed from the same state and the body motion
              each produces is compared against the goal. Anything else would let the teacher grade
              its own homework.

**The reading.** Teacher's pick reliably closer than the student's own means the labels were good
and F144 failed elsewhere. No better than the student's own means the teacher cannot order small
perturbations, which is the precise mechanism. **Coarse working while local does not** means it
ranks behaviours and not amounts -- the same shape as F111 and F140's within-clip `/mean-z` 0.951 --
and that tells any future distillation scheme it must choose among behaviours, never refine within
one.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from teacher_student_insect import Student, body_goal, load_teacher, pooled  # noqa: E402
from wm.data.embodiment import REGISTRY, body_velocity, load, yaw_rate  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

DATA = "data/beh12_c10f10t10_flat"


def family(cond):
    return "side" if cond.startswith("side") else cond.split("_")[0]


def channels_of(pos, quat, dt="0.05"):
    dt = float(dt)
    v = body_velocity(pos, quat, dt, "hexapod")
    w = yaw_rate(quat, dt, "hexapod", float(np.median(pos[:, 2])))
    return np.concatenate([v, np.asarray(w).reshape(len(v), 1)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_hex-b1_body3/stage3_hex_nce_s0.pt")
    ap.add_argument("--student", default="wm/runs/students/insect_taught.pt")
    ap.add_argument("--goal_clip", default=f"{DATA}/hexapod_ep100.npz")
    ap.add_argument("--cache", default="results/wm/cache/hex_c10.pt")
    ap.add_argument("--states", type=int, default=15, help="branch points for the local test")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--coarse_only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    mean_s = np.asarray(ck["body_stats"][0]).ravel()[:len(channels)]
    std_s = np.asarray(ck["body_stats"][1]).ravel()[:len(channels)]
    st = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(st["token_dim"], st["goal_dim"], st["action_dim"]).to(device).eval()
    student.load_state_dict(st["student"])

    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", channels)
    goal_std = torch.tensor((goal - mean_s) / std_s, dtype=torch.float32, device=device)
    with np.load(os.path.join(ROOT, args.goal_clip), allow_pickle=True) as z:
        goal_cond = str(z["condition"])
    print(f"goal {goal_cond} {np.round(goal, 4)} from {os.path.basename(args.goal_clip)}")

    cache = torch.load(os.path.join(ROOT, args.cache), map_location="cpu")

    # ---- coarse: the twelve recorded conditions as candidates ---------------------------------
    cand, seen = {}, set()
    for p_ in sorted(glob.glob(os.path.join(ROOT, DATA, "*.npz"))):
        with np.load(p_, allow_pickle=True) as z:
            c = str(z["condition"])
        if c not in seen:
            seen.add(c); cand[c] = p_
    conds = sorted(cand)
    acts = {c: torch.tensor(np.asarray(load(cand[c], REGISTRY["hexapod"])["actions"]),
                            dtype=torch.float32) for c in conds}
    hit = tot = 0
    with torch.no_grad():
        for p_ in sorted(glob.glob(os.path.join(ROOT, DATA, "*.npz")))[:24]:
            if p_ not in cache:
                continue
            e = cache[p_].float().to(device)
            for t in range(5, min(len(e) - args.horizon - 1, 50), 10):
                a = torch.stack([acts[c][min(t, len(acts[c]) - 1)] for c in conds]).to(device)
                z = proj(a, "hexapod")
                roll = e[t:t + 1].expand(len(conds), -1, -1)
                for _ in range(args.horizon):
                    roll = ftm(roll, z)
                motion = md.body(None, itm(e[t:t + 1].expand(len(conds), -1, -1), roll))
                if motion.dim() == 1:
                    motion = motion.unsqueeze(-1)
                k = min(motion.shape[-1], len(channels))
                err = (motion[:, :k] - goal_std[:k]).pow(2).mean(-1)
                hit += family(conds[int(err.argmin())]) == family(goal_cond)
                tot += 1
    print(f"\ncoarse -- twelve recorded conditions, does the pick share the goal's family")
    print(f"  {hit / max(tot, 1):.0%} of {tot} states   (chance 33% for `speed`; F143 read 95% "
          f"inside adapt3)")
    if args.coarse_only:
        return

    # ---- local: perturbations of the student's own action, judged in the simulator -------------
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])
    seed = np.asarray(clip["actions"], np.float32)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    branch = np.linspace(6, min(len(seed) - args.horizon - 1, 50), args.states).astype(int)

    def run_to(t_branch, tail):
        """Drive the recorded clip to `t_branch`, then hold `tail` for the horizon."""
        def policy(observation, t):
            return seed[t] if t < t_branch else tail
        return drive_and_record(sim, "medauroidea_stick_insect.ttt",
                                seed[:t_branch + args.horizon], 0.0, 20,
                                cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy)

    wins = ties = 0
    rows = []
    for bt in branch:
        frames, _a, _f, heads, oris = run_to(int(bt), seed[int(bt)])
        obs = frames[int(bt) - 1]
        with torch.no_grad():
            e_full = encode_clip(encoder, np.asarray(obs)[None], 1).float().to(device)
            g_t = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
            base = student.act(pooled(e_full), g_t)
            pert = base + (args.sigma * student.std.to(device)) * torch.randn(
                args.samples, base.shape[-1], device=device)
            allc = torch.cat([base, pert])
            z = proj(allc, "hexapod")
            roll = e_full.expand(len(allc), -1, -1)
            for _ in range(args.horizon):
                roll = ftm(roll, z)
            motion = md.body(None, itm(e_full.expand(len(allc), -1, -1), roll))
            if motion.dim() == 1:
                motion = motion.unsqueeze(-1)
            k = min(motion.shape[-1], len(channels))
            pick = int((motion[:, :k] - goal_std[:k]).pow(2).mean(-1).argmin())
        got = {}
        for tag, a in (("student", base[0]), ("teacher", allc[pick])):
            _fr, _ac, _fo, hh, oo = run_to(int(bt), a.cpu().numpy())
            hh, oo = np.asarray(hh, np.float64), np.asarray(oo, np.float64)
            got[tag] = channels_of(hh[-args.horizon - 1:], oo[-args.horizon - 1:])[:, channels].mean(0)
        d_s = float(np.linalg.norm(got["student"] - goal))
        d_t = float(np.linalg.norm(got["teacher"] - goal))
        wins += d_t < d_s
        ties += pick == 0
        rows.append((int(bt), d_s, d_t, pick))
        print(f"  t={bt:3d}  student {d_s:.4f}  teacher {d_t:.4f}  "
              f"{'teacher' if d_t < d_s else 'student'} closer"
              f"{'   (teacher kept the student action)' if pick == 0 else ''}", flush=True)

    n = len(rows)
    print(f"\nlocal -- the teacher's pick against the student's own, executed in the simulator")
    print(f"  teacher closer on {wins}/{n} states = {wins / max(n, 1):.0%}   "
          f"(a coin is 50%)")
    print(f"  the teacher kept the student's own action on {ties}/{n}")
    print(f"  mean distance to the goal: student {np.mean([r[1] for r in rows]):.4f}, "
          f"teacher {np.mean([r[2] for r in rows]):.4f}")


if __name__ == "__main__":
    main()
