"""Does the gradient through the world model's imagined rollout point the right way in the real world?

    .venv/bin/python3 scripts/diagnostics/dreamer_gradient.py \\
        --teacher wm/runs/beh12_ego/teacher_ego.pt --student wm/runs/students/insect_bc_ego.pt \\
        --data data/egocentric/beh12_c08f09t09_ego_flat \\
        --goal_clip data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz \\
        --scene medauroidea_c08f09t09.ttt --ego

**The de-risk before an actor-critic in imagination.** A Dreamer-style rebuild learns by
backpropagating an imagined return through the world model into the actor. The infrastructure is
standard; **the load-bearing assumption is not**. A model trained to *predict* is not automatically
a model whose *gradient* points anywhere useful, and this one is accurate for about three to five
steps (F138, F150). Nothing has tested the gradient.

The loop, per branch point and per horizon K:

  1. drive the recorded clip to the branch, take `e_0` from the frame there
  2. imagine K steps: `z_k = proj(a_k)`, `e_k+1 = FTM(e_k, z_k)`
  3. imagined reward = the F136 body coordinate of the rolled transition against the goal
  4. **backprop that reward into the action sequence** and take one normalised step
  5. **execute the stepped sequence in the simulator** and measure whether the real motion moved
     toward the goal, against executing the original

**Two controls, and the result is unreadable without either.**

**A random step of the same norm.** A gradient step changes the action; so does any step. If a
random step of equal size improves the real motion as often, the gradient carried nothing and the
improvement is the perturbation's.

**The recorded command range.** F181 measured the ranking collapse when actions leave the range the
model was fitted on -- 33% of joints outside it at sigma 2.0 and the teacher below a coin. A
gradient step that leaves that range has the same problem, so the fraction outside is reported
beside every number.

**One correction to the premise, stated because it changes what is being tested.** The frozen
V-JEPA2 encoder is **not** in this graph. Imagination runs in embedding space from a detached `e_0`,
so the backward path is `body head <- ITM <- FTM^K <- projector <- action` and nothing else. Whether
gradients vanish through the *encoder* is not this question and would only arise if the policy were
trained on pixels.
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
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_label_quality import EGO_CAM, channels_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="wm/runs/beh12_ego/teacher_ego.pt")
    ap.add_argument("--student", default="wm/runs/students/insect_bc_ego.pt")
    ap.add_argument("--data", default="data/egocentric/beh12_c08f09t09_ego_flat")
    ap.add_argument("--goal_clip", required=True)
    ap.add_argument("--scene", default="medauroidea_c08f09t09.ttt")
    ap.add_argument("--ego", action="store_true")
    ap.add_argument("--ego_seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--step", type=float, default=0.5,
                    help="size of the gradient step, in units of each joint's own command sd. "
                         "**The random control uses the same norm**, so the comparison is between "
                         "directions and not between step sizes.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    sck = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(sck["token_dim"], sck["goal_dim"], sck["action_dim"]).to(device).eval()
    student.load_state_dict(sck["student"])
    channels = list(sck["channels"])
    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", channels)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device)

    allacts = np.concatenate([np.asarray(load(p_, REGISTRY["hexapod"])["actions"], np.float64)
                              for p_ in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))])
    lo, hi = allacts.min(0), allacts.max(0)
    sd = torch.tensor(allacts.std(0), dtype=torch.float32, device=device)

    clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])
    seed = np.asarray(clip["actions"], np.float32)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    ego = dict(EGO_CAM, ego_seed=args.ego_seed) if args.ego else {}

    def run_seq(t_branch, tail_seq):
        """Drive the clip to the branch, then play `tail_seq` one command per step."""
        K = len(tail_seq)

        def policy(observation, t):
            return seed[t] if t < t_branch else tail_seq[min(t - t_branch, K - 1)]
        return drive_and_record(sim, args.scene, seed[:t_branch + K], 0.0, 20,
                                cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy, **ego)

    def reached(heads, oris, K):
        hh, oo = np.asarray(heads, np.float64), np.asarray(oris, np.float64)
        m = channels_of(hh[-K - 1:], oo[-K - 1:])[:, channels].mean(0)
        return float(np.linalg.norm(m - goal))

    branch = np.linspace(6, min(len(seed) - max(args.horizons) - 1, 50),
                         args.states).astype(int)

    print(f"{args.teacher}\ngoal {np.round(goal, 4)} from {os.path.basename(args.goal_clip)}")
    print(f"{args.states} branch points, horizons {args.horizons}, "
          f"step {args.step} sd, random control at the same norm\n")

    tally = {K: {"grad": 0, "rand": 0, "n": 0, "gnorm": [], "off": [], "d0": [],
                 "dg": [], "dr": []} for K in args.horizons}

    for bt in branch:
        frames, *_ = run_seq(int(bt), [seed[int(bt)]])
        with torch.no_grad():
            e0 = encode_clip(encoder, np.asarray(frames[int(bt) - 1])[None], 1).float().to(device)
        for K in args.horizons:
            # the sequence to be improved: the student's own action, repeated
            with torch.no_grad():
                base = student.act(pooled(e0), goal_t.unsqueeze(0))[0]
            a = base.detach().clone().unsqueeze(0).repeat(K, 1).requires_grad_(True)

            z = proj(a, "hexapod")
            roll = e0
            for k in range(K):
                roll = ftm(roll, z[k:k + 1])
            motion = md.body(None, itm(e0, roll))
            if motion.dim() == 1:
                motion = motion.unsqueeze(0)
            reward = -((motion[0, :len(channels)] - goal_t) ** 2).mean()
            g, = torch.autograd.grad(reward, a)

            gn = float(g.norm())
            tally[K]["gnorm"].append(gn)
            if not np.isfinite(gn) or gn == 0.0:
                # a dead or exploded gradient is the answer, not an error to route around
                tally[K]["n"] += 1
                continue

            unit = g / (g.norm(dim=-1, keepdim=True) + 1e-12)
            stepped = (a.detach() + args.step * sd * unit).cpu().numpy()
            rnd = torch.randn(g.shape, generator=torch.Generator(device=device).manual_seed(int(bt)),
                              device=device)
            rnd = rnd / (rnd.norm(dim=-1, keepdim=True) + 1e-12)
            control = (a.detach() + args.step * sd * rnd).cpu().numpy()

            d0 = reached(*run_seq(int(bt), [base.detach().cpu().numpy()] * K)[3:5], K)
            dg = reached(*run_seq(int(bt), list(stepped))[3:5], K)
            dr = reached(*run_seq(int(bt), list(control))[3:5], K)

            tally[K]["grad"] += dg < d0
            tally[K]["rand"] += dr < d0
            tally[K]["n"] += 1
            tally[K]["off"].append(float(np.mean((stepped < lo) | (stepped > hi))))
            tally[K]["d0"].append(d0); tally[K]["dg"].append(dg); tally[K]["dr"].append(dr)
            print(f"  t={bt:3d} K={K}  base {d0:.4f}  grad {dg:.4f}  random {dr:.4f}"
                  f"   |g| {gn:.2e}   off-range {tally[K]['off'][-1]:.0%}", flush=True)

    print(f"\n  {'K':>3}{'grad better':>13}{'random better':>15}{'mean base':>11}"
          f"{'mean grad':>11}{'mean rand':>11}{'|grad|':>11}{'off-range':>11}")
    for K in args.horizons:
        t = tally[K]
        n = max(t["n"], 1)
        print(f"  {K:>3}{t['grad']}/{t['n']:<11}{t['rand']}/{t['n']:<13}"
              f"{np.mean(t['d0']) if t['d0'] else float('nan'):>11.4f}"
              f"{np.mean(t['dg']) if t['dg'] else float('nan'):>11.4f}"
              f"{np.mean(t['dr']) if t['dr'] else float('nan'):>11.4f}"
              f"{np.mean(t['gnorm']):>11.2e}"
              f"{np.mean(t['off']) if t['off'] else float('nan'):>11.1%}")

    print("\n  **The gradient is only usable where it beats its own random control.** Any step")
    print("  changes the action, so `grad better` alone measures perturbation rather than")
    print("  direction. **And a step that leaves the recorded command range is one F181 already")
    print("  showed the model cannot predict**, so a win bought there is not a win.")
    print("\n  Gradients improving the real motion out to K=3-5 would give a Dreamer actor a")
    print("  foundation. Failing at K=1-2 means the model predicts and its gradient misleads,")
    print("  and an actor trained through it would be pushed the wrong way.")


if __name__ == "__main__":
    main()
