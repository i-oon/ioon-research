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

SCENE = "medauroidea_stick_insect.ttt"
DATA = "data/beh12_c10f10t10_flat"
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
    for epoch in range(args.epochs):
        student.train(); opt.zero_grad()
        loss = nn.functional.mse_loss(student(X[~V], G[~V]), target[~V])
        loss.backward(); opt.step()
        if (epoch + 1) % 200 == 0:
            student.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(student(X[V], G[V]), target[V]).item()
            print(f"  epoch {epoch + 1:4d}  train {loss.item():.4f}  held out {v:.4f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "token_dim": X.shape[-1],
                "goal_dim": G.shape[-1], "action_dim": Y.shape[-1],
                "channels": channels, "data": args.data,
                "val_paths": sorted(os.path.basename(p) for p in val_paths)}, out)
    print(f"-> {args.out}")


def run_in_sim(student, goal, device, port, steps, seed_cmds, encoder):
    """Drive the insect with the student and return frames, head track and orientation."""
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)

    def policy(observation, t):
        with torch.no_grad():
            e = encode_clip(encoder, np.asarray(observation)[None], 1).float().to(device)
            return student.act(pooled(e), g)[0].cpu().numpy()

    sim = RemoteAPIClient("localhost", port=port).getObject("sim")
    return drive_and_record(sim, SCENE, seed_cmds[:steps], 0.0, 20,
                            cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0), policy=policy)


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
                                                      args.steps, seed, encoder)
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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="wm/runs/students/insect_bc.pt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "bc":
        clone(args, device)
    elif args.stage == "eval":
        evaluate(args, device)
    else:
        raise SystemExit("`improve` needs the insect stage-3 teacher; it is not built until the "
                         "checkpoint exists and clears its gate (scripts/com7_stage3_hexapod.sh)")


if __name__ == "__main__":
    main()
