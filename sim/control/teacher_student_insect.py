"""Distil a walking policy for the insect from short imagined rollouts (F142).

    .venv/bin/python3 sim/control/teacher_student_insect.py bc      # bootstrap by cloning clips
    .venv/bin/python3 sim/control/teacher_student_insect.py improve # the world-model teacher
    .venv/bin/python3 sim/control/teacher_student_insect.py eval    # the pre-registered test

**Same robot on both sides. No cross-embodiment claim is made or measurable here.** This asks one
question: can short imagined rollouts train a policy that walks, on the easy case where the forward
model is trustworthy? If it cannot, teacher-student is dead and no transfer result can rescue it.

    teacher   from `e_t`, sample actions around the student's current output, project each through
              the action projector, roll the forward model **h <= 3** steps -- F140 measures the
              rollout worse than a frozen frame by five -- and read the body motion the rolled
              transition implies. The label is the sampled action whose imagined motion is nearest
              the goal.

    student   pi(e_t, goal) -> 18 joint targets. One forward pass at run time, no library, no
              rollout, no planner.

**The goal is a body-motion vector in the shared coordinate** (F136's three channels), which is why
this design is worth testing at all -- the same goal is readable from another robot's video. That
step is not taken here.

**Bootstrap is by cloning the insect's own recorded clips**, which is honest on this robot: it is
the body the project has data for, and F137 measured that a policy starting from noise never walks.
The `bc` stage alone is therefore also the control -- **if cloning already passes, the teacher has
added nothing**, and both numbers are reported.

**Pass or fail is decided by `eval` against a bar fixed before any training** (F142): upright for
the whole 3-second window **and** at least half of `D_real`, the replayed distance of a real walk.
Statues fail on distance, lurches fail on uprightness, and the render is diagnosis only -- it never
promotes a numeric fail.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

# **The base 1.0x body, and it is NOT the scene either behaviour dataset was collected from.**
# `beh12_c10f10t10_*` comes from `medauroidea_c10f10t10.ttt` and `beh12_c08f09t09_*` from
# `medauroidea_c08f09t09.ttt`; all three files are distinct on disk. Camera, lighting and floor are
# authored per scene and are part of the data, so a loop driving this file evaluates a policy in a
# world its training clips were not shot in. Override with `--scene`.
SCENE = "medauroidea_stick_insect.ttt"

# **The egocentric camera, and every value here is part of the data.** `--view egocentric` in the
# collector expands to fov 90 and an 8 m textured room, and neither is recorded in the npz -- so a
# loop that does not repeat them films a different world from the one the model was trained on and
# measures that difference. `ego_offset` and `ego_euler` stay at the collector's defaults (3 cm ahead
# of the head, 2 cm above, pitch-compensated for the walking posture) for the same reason.
#
# `ego_seed` is the appearance seed and advances by REPEAT in the dataset, so a loop reproducing a
# particular clip's world has to be given that clip's repeat index rather than 0.
EGO = dict(ego=True, cam_fov=90.0, ego_box=8.0)
DATA = "data/allocentric/beh12_c10f10t10_flat"
STEPS = 66


class Student(nn.Module):
    """`(pooled embedding, goal) -> standardised joint targets`.

    **Pooled, not the full token grid.** The policy has to run inside the simulator loop at 20 Hz;
    a 256x1408 input is a second model. Pooling loses where in the frame things are, which the
    scripts' own trap list warns about for readouts -- it is accepted here because the student is
    being asked for a gait, not for a spatial judgement.
    """

    def __init__(self, token_dim, goal_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim + goal_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim))
        self.register_buffer("mean", torch.zeros(action_dim))
        self.register_buffer("std", torch.ones(action_dim))

    def forward(self, e, goal):
        return self.net(torch.cat([e, goal], dim=-1))

    def act(self, e, goal):
        return self.forward(e, goal) * self.std + self.mean


def pooled(e):
    """One vector per frame. `e` is (T, tokens, dim) or (tokens, dim)."""
    return e.mean(-2)


def body_goal(clip_path, embodiment, channels):
    """The clip's own body motion, the quantity the student is asked to reproduce."""
    motion = np.asarray(load(clip_path, REGISTRY[embodiment])["body_motion"])[:, channels]
    return motion.mean(0)


def load_teacher(path, device):
    from wm.models.action_projector import ActionProjector, action_dims_from
    from wm.models.ftm import ForwardTransitionModel
    from wm.models.itm import InverseTransitionModel
    from wm.models.motion_decoder import MotionDecoder
    ck = torch.load(os.path.join(ROOT, path), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"hexapod": 18}).to(device).eval()
    md.load_state_dict(ck["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
    proj.load_state_dict(ck["projector"])
    for m in (itm, ftm, md, proj):
        for p in m.parameters():
            p.requires_grad_(False)
    return ck, cfg, itm, ftm, md, proj


def clone(args, device):
    """Bootstrap: fit the student on the insect's recorded frames and commands.

    **This is also the control.** If the cloned policy already clears the bar, the teacher stage has
    added nothing and F142 says so rather than crediting it.
    """
    ck, cfg, *_ = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    if args.forward_only:
        keep = []
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                if str(z["behaviour"]) == "speed":
                    keep.append(p)
        paths = keep
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    val_n = max(1, int(0.2 * len(paths)))
    val_paths = {paths[i] for i in order[:val_n]}
    print(f"cloning on {len(paths) - val_n} clips, {val_n} held out, from {args.data}")

    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    before = len(cache)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    X, G, Y, V = [], [], [], []
    for p in paths:
        clip = load(p, REGISTRY["hexapod"])
        if p not in cache:
            cache[p] = encode_clip(encoder, clip["frames"], 2).cpu().half()
        e = pooled(cache[p].float())
        a = torch.tensor(np.asarray(clip["actions"]), dtype=torch.float32)
        n = min(len(e), len(a))
        g = torch.tensor(body_goal(p, "hexapod", channels), dtype=torch.float32)
        X.append(e[:n]); Y.append(a[:n]); G.append(g.expand(n, -1))
        V.append(torch.full((n,), p in val_paths))
    if len(cache) > before:
        torch.save(cache, cache_path)
    del encoder
    torch.cuda.empty_cache()

    X = torch.cat(X).to(device); G = torch.cat(G).to(device)
    Y = torch.cat(Y).to(device); V = torch.cat(V).to(device)
    student = Student(X.shape[-1], G.shape[-1], Y.shape[-1]).to(device)
    student.mean.copy_(Y[~V].mean(0)); student.std.copy_(Y[~V].std(0).clamp_min(1e-6))
    target = (Y - student.mean) / student.std

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    # **Keep the best held-out weights, not the last.** Egocentrically the curve bottoms early and
    # then climbs -- 0.5629 at epoch 200 against 0.6202 at 2000 -- so saving the final state ships a
    # measurably worse policy than the run produced. Everything downstream perturbs around this
    # student, so the difference is not cosmetic.
    best = {"v": float("inf"), "epoch": 0, "state": None}
    for epoch in range(args.epochs):
        student.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(student(X[~V], G[~V]), target[~V])
        loss.backward(); opt.step()
        if (epoch + 1) % args.eval_every == 0:
            student.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(student(X[V], G[V]), target[V]).item()
            if v < best["v"]:
                best = {"v": v, "epoch": epoch + 1,
                        "state": {k: t.detach().clone() for k, t in student.state_dict().items()}}
            print(f"  epoch {epoch + 1:4d}  train {loss.item():.4f}  held out {v:.4f}"
                  + ("   <- best" if v == best["v"] else ""))
    if best["state"] is not None:
        student.load_state_dict(best["state"])
    # **The targets are standardised, so held-out MSE is 1 - R2 and the two are the same statement.**
    # Reported explicitly because the clone's bar comes from P3, which is quoted as R2.
    print(f"\n  best held out {best['v']:.4f} at epoch {best['epoch']}  =  R2 {1 - best['v']:+.3f}")
    print("  **Not comparable to P3's 0.263 unless the condition set matches** -- `--forward_only`")
    print("  clones on the walking clips alone, a much narrower target than the twelve conditions")
    print("  P3 measured, so a higher number here is not a better policy.")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "val_mse": best["v"], "val_epoch": best["epoch"],
                "forward_only": bool(args.forward_only), "token_dim": X.shape[-1],
                "goal_dim": G.shape[-1], "action_dim": Y.shape[-1],
                "channels": channels, "data": args.data,
                "val_paths": sorted(os.path.basename(p) for p in val_paths)}, out)
    print(f"-> {args.out}")


def run_in_sim(student, goal, device, port, steps, seed_cmds, encoder, ego=False, ego_seed=0,
               scene=SCENE):
    """Drive the insect with the student and return frames, head track and orientation.

    **`ego` must match how the student's training clips were shot.** A policy fitted on egocentric
    frames and evaluated through the fixed chase camera is being fed a distribution it never saw,
    and the run would measure the viewpoint swap rather than the policy.
    """
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)

    def policy(observation, t):
        with torch.no_grad():
            e = encode_clip(encoder, np.asarray(observation)[None], 1).float().to(device)
            return student.act(pooled(e), g)[0].cpu().numpy()

    sim = RemoteAPIClient("localhost", port=port).getObject("sim")
    return drive_and_record(sim, scene, seed_cmds[:steps], 0.0, 20,
                            cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy,
                            **(dict(EGO, ego_seed=ego_seed) if ego else {}))


def verdict(heads, d_real, fall_ratio=0.6):
    """The pre-registered test. Both conditions, over the whole window."""
    heads = np.asarray(heads, np.float64)
    d = float(np.linalg.norm(heads[-1, :2] - heads[0, :2]))
    z0 = float(np.median(heads[:5, 2]))
    upright = bool((heads[:, 2] >= fall_ratio * z0).all())
    return {"distance": d, "d_real": d_real, "fraction": d / max(d_real, 1e-9),
            "upright": upright, "min_z": float(heads[:, 2].min()), "z0": z0,
            "pass": bool(upright and d >= 0.5 * d_real)}


def evaluate(args, device):
    ck = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(ck["token_dim"], ck["goal_dim"], ck["action_dim"]).to(device).eval()
    student.load_state_dict(ck["student"])
    ref = np.load(os.path.join(ROOT, args.d_real), allow_pickle=True)
    d_real = float(ref["d_real"])
    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", list(ck["channels"]))
    seed = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])["actions"].astype(np.float32)
    print(f"D_real {d_real:.4f} m, bar {0.5 * d_real:.4f} m; goal {np.round(goal, 4)} "
          f"from {os.path.basename(args.goal_clip)}")

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    frames, actions, forces, heads, oris = run_in_sim(student, goal, device, args.port,
                                                      args.steps, seed, encoder,
                                                      ego=args.ego, ego_seed=args.ego_seed,
                                                      scene=args.scene)
    v = verdict(heads, d_real)
    print(f"\n  travelled {v['distance']:.4f} m = {v['fraction']:.0%} of D_real"
          f"   upright {v['upright']} (min head z {v['min_z']:.4f} against {v['z0']:.4f})")
    print(f"  **{'PASS' if v['pass'] else 'FAIL'}** -- both conditions are required and the render "
          f"does not overrule this")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, frames=np.asarray(frames, np.uint8),
                        actions=np.asarray(actions, np.float32),
                        head=np.asarray(heads, np.float32),
                        body_quat=np.asarray(oris, np.float32), dt=np.float32(0.05),
                        goal=np.asarray(goal, np.float32), **{k: v[k] for k in
                                                              ("distance", "d_real", "fraction",
                                                               "upright", "pass")})
    print(f"-> {args.out}")


def improve(args, device):
    """DAgger with the world model as the teacher: run the student, label what it saw, refit.

    **The candidate a step is labelled with is a constant command held for `h` steps**, which is
    what `--commit` does in the planner and keeps the sample space small enough to search at 20 Hz.
    Sampling a different action per step would search `h` times the dimensions for a horizon the
    forward model is only trustworthy over as a whole.

    **The cloning data stays in the buffer.** A teacher that only ever sees states the student
    reaches will happily walk it off the distribution its own labels were fitted on; keeping the
    recorded pairs is the standard guard and it is also what makes the comparison against the clone
    fair -- the teacher run has no data the clone did not.
    """
    ck, cfg, itm, ftm, md, proj = load_teacher(args.teacher, device)
    channels = [int(c) for c in cfg.body_channels]
    st = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(st["token_dim"], st["goal_dim"], st["action_dim"]).to(device)
    student.load_state_dict(st["student"])
    mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)],
                          dtype=torch.float32, device=device)
    std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)],
                         dtype=torch.float32, device=device)

    # goals come from the clips stage 3 trained on; the evaluation goal is held out from both
    train_names = set(os.popen(". scripts/run/hex_stage3_clips.sh; echo $HEX_CLIPS").read().split())
    goals = []
    for p_ in sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz"))):
        if os.path.basename(p_) not in train_names:
            continue
        with np.load(p_, allow_pickle=True) as z:
            if str(z["behaviour"]) != "speed":
                continue
        goals.append((p_, body_goal(p_, "hexapod", channels)))
    print(f"{len(goals)} forward goals from the stage-3 training clips; evaluation goal is not "
          f"among them")

    # the cloning pairs, kept in the buffer
    base = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    cache_path = os.path.join(ROOT, args.cache)
    cache = torch.load(cache_path, map_location="cpu") if os.path.exists(cache_path) else {}
    BX, BG, BY = [], [], []
    for p_, g in goals:
        clip = load(p_, REGISTRY["hexapod"])
        if p_ not in cache:
            cache[p_] = encode_clip(encoder, clip["frames"], 2).cpu().half()
        e = pooled(cache[p_].float())
        a = torch.tensor(np.asarray(clip["actions"]), dtype=torch.float32)
        n = min(len(e), len(a))
        BX.append(e[:n]); BY.append(a[:n])
        BG.append(torch.tensor(g, dtype=torch.float32).expand(n, -1))
    BX = torch.cat(BX).to(device); BG = torch.cat(BG).to(device); BY = torch.cat(BY).to(device)
    print(f"cloning buffer {len(BX)} pairs")

    seed = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])["actions"].astype(np.float32)
    sigma = (args.sigma * student.std).to(device)
    TX, TG, TY = [], [], []
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    for it in range(args.iters):
        gp, g = goals[int(rng.integers(len(goals)))]
        g_t = torch.tensor(g, dtype=torch.float32, device=device).unsqueeze(0)
        labelled = {"n": 0}

        def policy(observation, t):
            with torch.no_grad():
                e = pooled(encode_clip(encoder, np.asarray(observation)[None], 1).float().to(device))
                base_a = student.act(e, g_t)                       # 1 x action_dim
                cand = base_a + sigma * torch.randn(args.samples, base_a.shape[-1], device=device)
                cand = torch.cat([base_a, cand])                   # keep the student's own choice
                z = proj(cand, "hexapod")
                roll = e_full.expand(len(cand), -1, -1)
                for _ in range(args.horizon):
                    roll = ftm(roll, z)
                motion = md.body(None, itm(e_full.expand(len(cand), -1, -1), roll))
                if motion.dim() == 1:
                    motion = motion.unsqueeze(-1)
                k = min(motion.shape[-1], len(channels))
                err = (motion[:, :k] - ((g_t[:, :k] - mean_s[:k]) / std_s[:k])).pow(2).mean(-1)
                best = cand[int(err.argmin())]
                TX.append(e[0].cpu()); TG.append(g_t[0].cpu()); TY.append(best.cpu())
                labelled["n"] += 1
                return base_a[0].cpu().numpy()

        # the full token grid is needed by the ITM and the FDM; the pooled vector by the student
        e_full = None

        def policy_wrapper(observation, t):
            nonlocal e_full
            with torch.no_grad():
                e_full = encode_clip(encoder, np.asarray(observation)[None], 1).float().to(device)
            return policy(observation, t)

        frames, actions, forces, heads, oris = run_in_sim_raw(student, g_t, device, args.port,
                                                              args.steps, seed, policy_wrapper,
                                                              ego=args.ego,
                                                              ego_seed=args.ego_seed,
                                                              scene=args.scene)
        d = float(np.linalg.norm(np.asarray(heads)[-1, :2] - np.asarray(heads)[0, :2]))
        X = torch.cat([BX, torch.stack(TX).to(device)])
        G = torch.cat([BG, torch.stack(TG).to(device)])
        Y = torch.cat([BY, torch.stack(TY).to(device)])
        target = (Y - student.mean) / student.std
        for _ in range(args.refit):
            student.train(); opt.zero_grad()
            loss = nn.functional.mse_loss(student(X, G), target)
            loss.backward(); opt.step()
        student.eval()
        print(f"  iter {it + 1:2d}/{args.iters}  goal {os.path.basename(gp)}  "
              f"travelled {d:.4f} m  labelled {labelled['n']}  buffer {len(TX)}  "
              f"loss {loss.item():.4f}", flush=True)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({**base, "student": student.state_dict(),
                "teacher": args.teacher, "iters": args.iters, "horizon": args.horizon,
                "samples": args.samples, "sigma": args.sigma}, out)
    print(f"-> {args.out}")


def run_in_sim_raw(student, goal, device, port, steps, seed_cmds, policy, ego=False, ego_seed=0,
                   scene=SCENE):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    sim = RemoteAPIClient("localhost", port=port).getObject("sim")
    return drive_and_record(sim, scene, seed_cmds[:steps], 0.0, 20,
                            cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy,
                            **(dict(EGO, ego_seed=ego_seed) if ego else {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("bc", "improve", "eval"))
    ap.add_argument("--teacher", default="wm/runs/beh12_hex-b1_body3/stage3_hex_nce_s0.pt")
    ap.add_argument("--student", default="wm/runs/students/insect_bc.pt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--cache", default="results/wm/cache/hex_c10.pt")
    ap.add_argument("--goal_clip", default=f"{DATA}/hexapod_ep100.npz")
    ap.add_argument("--d_real", default="results/wm/closed_loop/f142_d_real.npz")
    ap.add_argument("--forward_only", action="store_true",
                    help="clone on the forward-walking clips alone. **The engine test is whether a "
                         "policy walks**; adding turns and strafes to the clone widens what it has "
                         "to fit before the question is even asked")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200,
                    help="held-out evaluations, which are also the checkpoint candidates: the best "
                         "of them is what gets saved.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--scene", default=SCENE,
                    help="**must be the scene the student's clips were collected from.** "
                         "`medauroidea_c10f10t10.ttt` for `beh12_c10f10t10_*`, "
                         "`medauroidea_c08f09t09.ttt` for `beh12_c08f09t09_*`. The default is the "
                         "base body and matches neither.")
    ap.add_argument("--ego", action="store_true",
                    help="**film the loop from the head camera, matching an egocentric training "
                         "set.** Expands to fov 90 and an 8 m textured room, which is what "
                         "`--view egocentric` expands to in the collector; neither value is stored "
                         "in the npz, so a loop that omits this films a different world from the "
                         "one the student was fitted on and measures the swap rather than the "
                         "policy.")
    ap.add_argument("--ego_seed", type=int, default=0,
                    help="the room's appearance seed. It advances by REPEAT in the datasets, so "
                         "reproducing a particular clip's world needs that clip's repeat index.")
    ap.add_argument("--iters", type=int, default=10, help="simulator episodes of teacher labels")
    ap.add_argument("--samples", type=int, default=32, help="candidates the teacher ranks per step")
    ap.add_argument("--sigma", type=float, default=0.5, help="candidate spread, in action sd")
    ap.add_argument("--horizon", type=int, default=3,
                    help="**never past 3.** F143 measures the teacher's state ratio crossing 0.8 "
                         "exactly there on this body")
    ap.add_argument("--refit", type=int, default=300, help="gradient steps after each episode")
    ap.add_argument("--out", default="wm/runs/students/insect_bc.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "bc":
        clone(args, device)
    elif args.stage == "eval":
        evaluate(args, device)
    else:
        improve(args, device)


if __name__ == "__main__":
    main()
