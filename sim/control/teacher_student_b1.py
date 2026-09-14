"""B1 teacher-student: proprioceptive behavior cloning, then DAgger against the world-model teacher.

**History of this file's name, kept here because it is informative, not just trivia.** First built
with only `bc`/`eval` -- no grading stage existed, so it was named `clone_b1.py` to say precisely
that. Once `improve()` was added below (this entry), "teacher-student" became accurate again and
the file reverted to this name -- the lesson being the name should track what stages actually exist,
checked at every change, not asserted once and left standing.

    .venv/bin/python3 sim/control/teacher_student_b1.py bc      --forward_only --out wm/runs/students/b1_bc_forward.pt
    .venv/bin/python3 sim/control/teacher_student_b1.py bc      --out wm/runs/students/b1_bc_all.pt
    .venv/bin/python3 sim/control/teacher_student_b1.py bc_aux  --forward_only --base_ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt \\
        --out wm/runs/students/b1_bc_aux_forward.pt
    .venv/bin/python3 sim/control/teacher_student_b1.py improve --base_ckpt wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt \\
        --student wm/runs/students/b1_bc_forward.pt --out wm/runs/students/b1_dagger_forward.pt
    .venv/bin/python3 sim/control/teacher_student_b1.py eval    --student wm/runs/students/b1_dagger_forward.pt

**Why proprioceptive, unlike the insect version.** `beh12_b1_ego_flat` already carries `joint_pos`/
`joint_vel`/`base_quat`/`base_pos` every frame -- the exact quantities `B1MuJoCoEnv._obs()` assembles
live. Cloning on that needs no VJEPA2 encoding at train or eval time and needs no CoppeliaSim: the
whole loop is MuJoCo only, reusing `B1MuJoCoEnv`'s already-validated reset/step/fall logic instead of
a second physics harness. Base linear+angular velocity isn't recorded directly (only position/
quaternion), so it's rebuilt by finite-differencing -- validated directly against `B1MuJoCoEnv`'s own
live `qvel` on a re-simulated trajectory before trusting it as training data: central-difference
linear velocity is accurate to 2.1% of its own std, angular to 18.1% (a real, stated imprecision --
angular finite-differencing is intrinsically noisier at this control rate, not a bug).

**Reuses `Student`/`body_goal`/`verdict` from `teacher_student_insect.py` unchanged** -- the goal is
each clip's OWN recorded body motion (self-supervised, same-embodiment), exactly the insect's
methodology: this is an engine test on B1 alone, no cross-embodiment claim (that path -- reading the
goal from another body's video -- is F210's, already validated, and deliberately not exercised here).

**A real train/eval physics gap, stated because it could explain a failure.** `beh12_b1_ego_flat` was
collected on `sim/assets/b1_mujoco/b1_flat.xml` (uniform placeholder joint physics --
`rollout_b1_mujoco.py:21`). `B1MuJoCoEnv` (used here for both `D_real` and `eval`) defaults to
`b1_flat_real.xml` (system-identified damping/friction -- F209 measured this is what makes an
open-loop gait work at all). So the student is cloned on state/action pairs generated under one
physics model and evaluated in a dynamically different one. `D_real` is computed under the SAME
model `eval` uses (not read off the clip's own recorded `base_pos`, which reflects the OTHER model),
so the bar is at least fair to the physics the student is actually judged in -- but the gap between
training and evaluation dynamics remains a live, undismissed hypothesis for any failure below.

**The grading/DAgger stage (`improve_b1`), added after `clone_b1`/`evaluate_b1` were already
validated.** F135/F136/F138 measured, on the insect, that local-perturbation ranking fails at
33% against a 50% coin flip -- real physics barely distinguishes nearby actions at fixed magnitude,
a task property the insect result argued was not insect-specific. That argument was an
extrapolation, never actually run on B1; `improve_b1` exists to check it directly rather than
assume it, on the one embodiment the extrapolation was never tested on.
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
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
from teacher_student_insect import Student, body_goal, verdict  # noqa: E402

from wm.config import from_checkpoint  # noqa: E402
from wm.data.embodiment import B1, load  # noqa: E402
from wm.policy.b1_mujoco_env import ACTION_SCALE, DEFAULT_IL, B1MuJoCoEnv, il_to_sdk  # noqa: E402

DATA = "data/egocentric/beh12_b1_ego_flat"
STEPS = 66  # matches the clips' own length


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def base_velocity_fd(base_pos, base_quat, dt):
    """Central-difference world-frame linear + body-frame angular velocity, matching MuJoCo's own
    free-joint `qvel[0:6]` convention. Validated against a live re-simulated trajectory before use
    (see module docstring): 2.1%/18.1% mean-abs-error against true `qvel`, central difference
    (t-1, t+1) -- a one-step forward difference measured WORSE (8.5%/39.8%), confirmed directly, not
    assumed, so central is what's used here despite the larger nominal step.
    """
    n = len(base_pos)
    vel = np.zeros((n, 6), dtype=np.float32)
    for t in range(n):
        t0, t1 = max(0, t - 1), min(n - 1, t + 1)
        denom = max(1, t1 - t0) * dt
        vel[t, 0:3] = (base_pos[t1] - base_pos[t0]) / denom
        dq = quat_mul(quat_conj(base_quat[t0]), base_quat[t1])
        dq = dq / np.linalg.norm(dq)
        vel[t, 3:6] = 2.0 * dq[1:4] / denom
    return vel


def body_state(clip_path):
    """34-d proprioceptive state per frame, in exactly `B1MuJoCoEnv._obs()`'s convention:
    12 joint pos + 12 joint vel + 4 base quat + 6 base lin/ang vel."""
    with np.load(clip_path, allow_pickle=True) as d:
        joint_pos = d["joint_pos"].astype(np.float32)
        joint_vel = d["joint_vel"].astype(np.float32)
        base_pos = d["base_pos"].astype(np.float64)
        base_quat = d["base_quat"].astype(np.float64)
        dt = float(d["dt"])
    base_vel = base_velocity_fd(base_pos, base_quat, dt)
    return np.concatenate([joint_pos, joint_vel, base_quat.astype(np.float32), base_vel], axis=1)


def clone_b1(args, device):
    """Pure imitation: fit the student on B1's own recorded frames and commands, no grading of any
    kind. Also the control -- if this alone clears the bar, nothing past it would have been needed."""
    ck = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
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

    X, G, Y, V = [], [], [], []
    for p in paths:
        clip = load(p, B1)
        s = body_state(p)
        a = np.asarray(clip["actions"], dtype=np.float32)
        n = min(len(s), len(a))
        g = body_goal(p, "b1", channels)
        X.append(torch.from_numpy(s[:n])); Y.append(torch.from_numpy(a[:n]))
        G.append(torch.from_numpy(g).float().expand(n, -1))
        V.append(torch.full((n,), p in val_paths))

    X = torch.cat(X).to(device); G = torch.cat(G).to(device)
    Y = torch.cat(Y).to(device); V = torch.cat(V).to(device)
    student = Student(X.shape[-1], G.shape[-1], Y.shape[-1]).to(device)
    student.mean.copy_(Y[~V].mean(0)); student.std.copy_(Y[~V].std(0).clamp_min(1e-6))
    target = (Y - student.mean) / student.std

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
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
    print(f"\n  best held out {best['v']:.4f} at epoch {best['epoch']}  =  R2 {1 - best['v']:+.3f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "val_mse": best["v"], "val_epoch": best["epoch"],
               "forward_only": bool(args.forward_only), "state_dim": X.shape[-1],
               "goal_dim": G.shape[-1], "action_dim": Y.shape[-1], "channels": channels,
               "data": args.data, "val_paths": sorted(os.path.basename(p) for p in val_paths)}, out)
    print(f"-> {args.out}")


def aux_bc_b1(args, device):
    """Option A: backprop through the frozen model at the student's own action, no sampling, no
    relabelling -- the one candidate mechanism F216 (fine local-perturbation grading) and F217
    (coarse-candidate distillation) didn't already rule out, because neither of those failures was
    about the model's *gradient*: F216 failed from comparing noisy discrete samples, F217 from
    pairing one clip's states with a different clip's actions. This never samples or relabels
    anything -- it adds one differentiable loss term to plain BC.

    **Pre-registered before running (verbatim, this session):**
      - A beats plain BC (`b1_bc_forward.pt`, 52% of D_real) and stays upright -> the model's
        gradient carries real information about the action even where its coarse discrete ranking
        does not -- a first positive on the acting side.
      - A ~= plain BC, or the auxiliary loss stays flat/uninformative during training -> fine-scale
        is dead at the level of an exact analytic gradient too, not just noisy discrete sampling --
        closes the acting-side question as a fully characterized negative, not just F216/F217's two
        specific mechanisms.

    **Also pre-registered as a live risk, stated because it changes how a negative should be read.**
    A uses the SAME frozen `body_head` whose local discrimination F135/F136/F138/F216 already
    measured as failing at fine scale. A flat, noisy function has a flat, noisy gradient regardless
    of how exactly it's computed -- autodiff doesn't invent sensitivity finite-difference sampling
    couldn't find. So a negative result here does not distinguish "the signal was measured too
    noisily before" from "the signal genuinely isn't there at this scale" -- it only rules out the
    first explanation if A is measurably different from noise, which the training curve itself
    (does the aux loss actually move, or sit flat from epoch 1) is what to look at first.

    Frames come from the dataset's OWN recorded video (`beh12_b1_ego_flat`'s `frames` field) --
    encoded once, offline, cached in memory for the run. No live CoppeliaSim rendering, no sampled
    candidates: `student(state, goal) -> action -> proj -> roll FTM h steps from that clip's own
    recorded frame at this timestep -> itm -> body_head -> compare to the same goal the BC loss
    already targets`.
    """
    from wm.evaluate import encode_clip, offset_for
    from wm.models.action_projector import ActionProjector, action_dims_from
    from wm.models.ftm import ForwardTransitionModel
    from wm.models.itm import InverseTransitionModel
    from wm.models.motion_decoder import MotionDecoder
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from vjepa2_encoder import VJEPA2FrameEncoder

    # **On an 11 GB card, run the frozen model and the training loop on CPU; only the one-time
    # video encoding uses the GPU.** The 1B-parameter VJEPA2 encoder's own resident weights and
    # activations leave no headroom afterward for a training batch through FTM's cross-attention
    # over the 256-token grids -- measured directly (OOM even at batch 48, even after freeing the
    # encoder and calling `empty_cache()`). Measured equally directly that the CPU fallback is not
    # actually cheap: 54 minutes and 21.5 GB RAM without finishing on this machine, killed rather
    # than let it keep going -- **use --cpu_models only as a last resort on an 11 GB card; on a
    # 16 GB+ card (BIAS-2/com7) leave it off and everything stays on GPU, which is what this
    # defaults to.**
    mdl_device = torch.device("cpu") if args.cpu_models else device
    ck = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    itm = InverseTransitionModel(cfg).to(mdl_device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(mdl_device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"b1": 12}).to(mdl_device).eval(); md.load_state_dict(ck["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(mdl_device).eval()
    proj.load_state_dict(ck["projector"])
    for m in (itm, ftm, md, proj):
        for p in m.parameters():
            p.requires_grad_(False)
    mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)],
                          dtype=torch.float32, device=mdl_device)
    std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)],
                         dtype=torch.float32, device=mdl_device)
    offset = offset_for(ck, "b1")
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

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
    print(f"aux-BC on {len(paths) - val_n} clips, {val_n} held out, from {args.data}", flush=True)

    X, G, Y, E, V = [], [], [], [], []
    for p in paths:
        clip = load(p, B1)
        s = body_state(p)
        a = np.asarray(clip["actions"], dtype=np.float32)
        with np.load(p, allow_pickle=True) as raw:
            frames = raw["frames"]
        with torch.no_grad():
            e = encode_clip(encoder, frames, 4).float().cpu()
            if offset is not None:
                e = e - offset.cpu()
        n = min(len(s), len(a), len(e))
        g = body_goal(p, "b1", channels)
        X.append(torch.from_numpy(s[:n])); Y.append(torch.from_numpy(a[:n]))
        G.append(torch.from_numpy(g).float().expand(n, -1))
        E.append(e[:n])
        V.append(torch.full((n,), p in val_paths))
        print(f"  encoded {os.path.basename(p)}", flush=True)

    del encoder
    torch.cuda.empty_cache()  # release the encoder's VRAM now that every clip is encoded

    X = torch.cat(X).to(mdl_device); G = torch.cat(G).to(mdl_device)
    Y = torch.cat(Y).to(mdl_device); V = torch.cat(V).to(mdl_device)
    E = torch.cat(E).to(mdl_device)  # ~1.5 GB for this dataset -- fine on a 16 GB card, kept off
                                     # GPU only when --cpu_models forces mdl_device to CPU

    student = Student(X.shape[-1], G.shape[-1], Y.shape[-1]).to(mdl_device)
    student.mean.copy_(Y[~V].mean(0)); student.std.copy_(Y[~V].std(0).clamp_min(1e-6))
    target = (Y - student.mean) / student.std
    train_idx = (~V).nonzero(as_tuple=True)[0]
    val_idx = V.nonzero(as_tuple=True)[0]

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    best = {"v": float("inf"), "epoch": 0, "state": None}
    rng2 = np.random.default_rng(args.seed)
    bsz = min(args.batch, len(train_idx))
    for epoch in range(args.epochs):
        student.train(); opt.zero_grad()
        batch = train_idx[torch.from_numpy(rng2.choice(len(train_idx), size=bsz, replace=False)).to(mdl_device)]
        bc_loss = nn.functional.mse_loss(student(X[batch], G[batch]), target[batch])

        e_t = E[batch]
        a_pred = student.act(X[batch], G[batch])
        z = proj(a_pred, "b1")
        roll = e_t
        for _ in range(args.horizon):
            roll = ftm(roll, z)
        motion_std = md.body(None, itm(e_t, roll))
        k = min(motion_std.shape[-1], len(channels))
        goal_std = (G[batch][:, :k] - mean_s[:k]) / std_s[:k]
        aux_loss = nn.functional.mse_loss(motion_std[:, :k], goal_std)

        loss = bc_loss + args.lambda_aux * aux_loss
        loss.backward(); opt.step()

        if (epoch + 1) % args.eval_every == 0:
            student.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(student(X[val_idx], G[val_idx]), target[val_idx]).item()
            if v < best["v"]:
                best = {"v": v, "epoch": epoch + 1,
                        "state": {kk: t.detach().clone() for kk, t in student.state_dict().items()}}
            print(f"  epoch {epoch + 1:4d}  bc {bc_loss.item():.4f}  aux {aux_loss.item():.4f}  "
                  f"held out {v:.4f}" + ("   <- best" if v == best["v"] else ""), flush=True)
    if best["state"] is not None:
        student.load_state_dict(best["state"])
    print(f"\n  best held out {best['v']:.4f} at epoch {best['epoch']}  =  R2 {1 - best['v']:+.3f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "val_mse": best["v"], "val_epoch": best["epoch"],
               "forward_only": bool(args.forward_only), "state_dim": X.shape[-1],
               "goal_dim": G.shape[-1], "action_dim": Y.shape[-1], "channels": channels,
               "data": args.data, "val_paths": sorted(os.path.basename(p) for p in val_paths),
               "lambda_aux": args.lambda_aux, "aux_horizon": args.horizon}, out)
    print(f"-> {args.out}")


def distill_b1(args, device):
    """Train the student on the model's OWN coarse candidate choice instead of the raw recorded
    expert action, in place of a "correct the student at its drift points" test that turned out
    not to be buildable: `DirectFroudePlanner` (F210's own mechanism, the only one proven to
    discriminate well -- 90%+ family accuracy) scores candidates purely as
    `body_head(proj(candidate_actions)) vs. goal`, with no notion of the live rollout's actual
    current state at all -- it cannot see or react to drift. The planner that DOES look at the
    live state (`LatentPlanner`, rolling the FTM forward) is the one Slide 24/F127 already showed
    prefers the WORST candidate. Neither combination gives a state-aware corrector that also
    discriminates, so this tests a different, honestly-buildable question instead: does relabeling
    training data with the model's own coarse pick change the resulting policy at all?

    **Expected outcome, stated before running rather than after.** Each training clip's goal is
    drawn from that SAME clip's own recorded motion, so the planner is very likely to pick that
    clip (or a near-duplicate of the same condition) as its own best match -- in which case this
    reduces to relabeling each clip with close to its own actions, and the result should look like
    plain BC. A materially different result, in either direction, is the informative case, and
    `self_match` below is printed precisely so that isn't discovered only after training.

    Uses the SAME held-out split as `--reference` (default `b1_bc_all.pt`, F212's all-behaviour
    clone) so the two are compared on identical held-out data, not just identical architecture.
    """
    from wm.policy.planner import DirectFroudePlanner

    planner = DirectFroudePlanner.from_checkpoint(
        os.path.join(ROOT, args.base_ckpt), os.path.join(ROOT, args.data), embodiment="b1",
        horizon=args.horizon, per_condition=1, device=str(device), free_offset=True)
    channels = planner.channels
    print(f"{len(planner.candidates)} coarse candidates: "
         f"{[c['condition'] for c in planner.candidates]}")

    ref = torch.load(os.path.join(ROOT, args.reference), map_location="cpu", weights_only=False)
    val_paths = set(ref["val_paths"])
    paths = sorted(glob.glob(os.path.join(ROOT, args.data, "*.npz")))
    print(f"distilling on {len(paths) - len(val_paths)} clips, {len(val_paths)} held out "
         f"(matching --reference's own split), from {args.data}")

    X, G, Y, V = [], [], [], []
    self_match = 0
    for p in paths:
        s = body_state(p)
        g = body_goal(p, "b1", channels)
        goal_std = planner.standardize(g)
        action_seq, i, scores, tau = planner.act(goal_std, 0)
        cand = planner.candidates[i]
        if os.path.abspath(cand["path"]) == os.path.abspath(p):
            self_match += 1
        n = min(len(s), len(cand["actions"]) - tau)
        a = cand["actions"][tau:tau + n]
        X.append(torch.from_numpy(s[:n])); Y.append(torch.from_numpy(a))
        G.append(torch.from_numpy(g).float().expand(n, -1))
        V.append(torch.full((n,), os.path.basename(p) in val_paths))
    print(f"planner picked the clip's OWN condition as the best match on {self_match}/{len(paths)} "
         f"training clips")

    X = torch.cat(X).to(device); G = torch.cat(G).to(device)
    Y = torch.cat(Y).to(device); V = torch.cat(V).to(device)
    student = Student(X.shape[-1], G.shape[-1], Y.shape[-1]).to(device)
    student.mean.copy_(Y[~V].mean(0)); student.std.copy_(Y[~V].std(0).clamp_min(1e-6))
    target = (Y - student.mean) / student.std

    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
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
    print(f"\n  best held out {best['v']:.4f} at epoch {best['epoch']}  =  R2 {1 - best['v']:+.3f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"student": student.state_dict(), "val_mse": best["v"], "val_epoch": best["epoch"],
               "forward_only": False, "state_dim": X.shape[-1],
               "goal_dim": G.shape[-1], "action_dim": Y.shape[-1], "channels": channels,
               "data": args.data, "val_paths": sorted(os.path.basename(p) for p in val_paths),
               "self_match": self_match, "n_train_clips": len(paths)}, out)
    print(f"-> {args.out}")


def seed_from_clip(env, clip_path):
    """Set MuJoCo state to a clip's own recorded frame 0, not the env's static default pose.

    **Why this matters, found by measuring, not assumed.** `rollout_b1_mujoco.py` runs
    `--policy_warmup 45` steps of real walking BEFORE recording starts (`_i < policy_warmup:
    continue`), so a clip's own `actions[0]` was applied to an already-moving body, mid-gait --
    not a freshly reset, standing one. Replaying it from `env.reset()`'s static pose measured a
    net displacement in the WRONG direction (-0.23 m against the clip's own recorded +1.25 m,
    same action sequence, same model file) -- not a sign bug, a mismatched initial condition.
    Base linear/angular velocity isn't recorded directly, so it's rebuilt with the same
    finite-difference formula `body_state()` uses (validated to 2.1%/18.1% mean-abs-error against
    live MuJoCo `qvel` elsewhere in this file).
    """
    with np.load(clip_path, allow_pickle=True) as d:
        base_pos = d["base_pos"].astype(np.float64)
        base_quat = d["base_quat"].astype(np.float64)
        joint_pos = d["joint_pos"].astype(np.float64)
        joint_vel = d["joint_vel"].astype(np.float64)
        dt = float(d["dt"])
    import mujoco
    mujoco.mj_resetData(env.m, env.d)
    env.d.qpos[0:3] = base_pos[0]
    env.d.qpos[3:7] = base_quat[0]
    env.d.qpos[7:19] = joint_pos[0]
    env.d.qvel[6:18] = joint_vel[0]
    base_vel0 = base_velocity_fd(base_pos[:3], base_quat[:3], dt)[0]  # fwd-diff from frames 0,1
    env.d.qvel[0:6] = base_vel0
    mujoco.mj_forward(env.m, env.d)
    env.air_time[:] = 0.0
    env.prev_contact[:] = False
    env.pos_hist = np.tile(env.d.qpos[0:3].copy(), (env.hist_len, 1))
    env.quat_hist = np.tile(env.d.qpos[3:7].copy(), (env.hist_len, 1))
    env.t = 0


def apply_action_unclipped(env, action):
    """Step physics exactly as `rollout_b1_mujoco.py` does -- clamp only the FINAL scaled target
    at the physical actuator range, never the raw action at +-1 first.

    **Why not `env.step()`.** `B1MuJoCoEnv.step()` clips the raw action to [-1, 1] before scaling,
    correct for a tanh-squashed RL actor but wrong here: this dataset's own recorded expert
    actions are unbounded (up to 3.5, 32% already exceed |1|, matching F203's already-documented
    "action-space corruption" pattern), and the student is trained to reproduce that same
    distribution (plain linear output head, no tanh). Feeding either through `env.step()`'s
    pre-clip truncates a third of the commanded targets and silently breaks the gait -- measured
    directly: replaying the clip's own recorded actions through `env.step()` gave -0.17 m to
    -0.50 m net displacement (wrong direction) against the clip's own recorded +1.25 m, on the
    SAME model file. This function reproduces instead.
    """
    import mujoco
    target = il_to_sdk(DEFAULT_IL + ACTION_SCALE * np.asarray(action, dtype=np.float64))
    env.d.ctrl[:] = np.clip(target, env.m.actuator_ctrlrange[:, 0], env.m.actuator_ctrlrange[:, 1])
    for _ in range(env.decimation):
        mujoco.mj_step(env.m, env.d)
    env.t += 1


def rollout_open_loop(env, actions, clip_path=None):
    """Replay a fixed action sequence through `env`'s own physics, return per-step head xyz.

    `clip_path` given: seed from that clip's own recorded frame-0 state (for `D_real` -- see
    `seed_from_clip`). Omitted: fresh `env.reset()` (for evaluating a student from a standing
    start, matching F133's insect precedent)."""
    if clip_path is not None:
        seed_from_clip(env, clip_path)
    else:
        env.reset()
    heads = [env.d.qpos[0:3].copy()]
    for a in actions:
        apply_action_unclipped(env, a)
        heads.append(env.d.qpos[0:3].copy())
    return np.asarray(heads)


def rollout_student(env, student, goal, device, steps, renderer=None, frames=None):
    obs = env.reset()
    g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    heads = [env.d.qpos[0:3].copy()]
    if renderer is not None:
        renderer.update_scene(env.d, camera=-1); frames.append(renderer.render())
    for _ in range(steps):
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = student.act(s, g)[0].cpu().numpy()
        apply_action_unclipped(env, action)  # not env.step() -- see its docstring
        obs = env._obs()
        heads.append(env.d.qpos[0:3].copy())
        if renderer is not None:
            renderer.update_scene(env.d, camera=-1); frames.append(renderer.render())
    return np.asarray(heads)


def improve_b1(args, device):
    """DAgger with the world model as teacher: run the student, grade what it saw against
    `body_head_b1_hex_v2.pt`, refit. The stage this file originally left out -- added because
    clone-only never tests whether the graded pipeline (F210's actual mechanism) beats plain BC.

    **Three departures from `improve()` in `teacher_student_insect.py`, each forced by B1's own
    pipeline, not a new design:**

    - Physics is MuJoCo, camera is CoppeliaSim, live, per step -- exactly the split
      `close_loop_b1_physics.py` already established (`render()` below is that file's closure,
      reused almost verbatim). B1 has no walking physics in CoppeliaSim at all.
    - The room is rendered **in the same egocentric convention `beh12_b1_ego_flat` itself was
      rendered in** -- a head camera parented to the base via `ego_camera.attach_ego`, the same
      room-size/pitch-compensation math (`room_for`, `WALK_PITCH["b1"]`). `body_head_b1_hex_v2.pt`
      was fit on those frames; grading through MuJoCo's own renderer instead would feed ITM/
      body_head visually out-of-domain frames and confound "does grading help" with "does a
      render-style shift break body_head" -- exactly the risk that deferred this stage until now.
    - The goal is B1's OWN recorded body motion (`body_goal(..., "b1", ...)`), matching
      `clone_b1`'s convention exactly, kept from its OWN training split (never the held-out eval
      clip) -- so this is a fair, same-goal-distribution comparison against the BC-only number,
      not a new cross-embodiment claim (that is F210's, kept separate).

    **The cloning buffer stays in memory and in the refit**, the same guard `improve()` uses: a
    teacher that only ever sees states the student reaches will happily walk it off the
    distribution its own labels were fitted on.
    """
    import mujoco
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    sys.path.insert(0, os.path.join(ROOT, "sim", "render"))
    sys.path.insert(0, os.path.join(ROOT, "sim", "scene"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from ego_camera import attach_ego, build_texture_box, randomise_ground, room_for, WALK_PITCH
    from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, SENSOR, capture, settle
    from vjepa2_encoder import VJEPA2FrameEncoder

    from wm.data.embodiment import heading  # noqa: E402
    from wm.evaluate import encode_clip, offset_for  # noqa: E402
    from wm.models.action_projector import ActionProjector, action_dims_from  # noqa: E402
    from wm.models.ftm import ForwardTransitionModel  # noqa: E402
    from wm.models.itm import InverseTransitionModel  # noqa: E402
    from wm.models.motion_decoder import MotionDecoder  # noqa: E402

    ck = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
    cfg = from_checkpoint(ck["config"])
    channels = [int(c) for c in cfg.body_channels]
    itm = InverseTransitionModel(cfg).to(device).eval(); itm.load_state_dict(ck["itm"])
    ftm = ForwardTransitionModel(cfg).to(device).eval(); ftm.load_state_dict(ck["ftm"])
    md = MotionDecoder(cfg, {"b1": 12}).to(device).eval(); md.load_state_dict(ck["md"], strict=False)
    proj = ActionProjector(cfg, action_dims_from(ck)).to(device).eval()
    proj.load_state_dict(ck["projector"])
    for m in (itm, ftm, md, proj):
        for p in m.parameters():
            p.requires_grad_(False)
    mean_s = torch.tensor(np.asarray(ck["body_stats"][0]).ravel()[:len(channels)],
                          dtype=torch.float32, device=device)
    std_s = torch.tensor(np.asarray(ck["body_stats"][1]).ravel()[:len(channels)],
                         dtype=torch.float32, device=device)
    offset = offset_for(ck, "b1")

    st = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(st["state_dim"], st["goal_dim"], st["action_dim"]).to(device)
    student.load_state_dict(st["student"])

    # the cloning buffer: clone_b1's own training split (its held-out eval clip is never in here)
    paths = sorted(glob.glob(os.path.join(ROOT, st["data"], "*.npz")))
    val_names = set(st["val_paths"])
    train_paths = [p for p in paths if os.path.basename(p) not in val_names]
    if st["forward_only"]:
        keep = []
        for p in train_paths:
            with np.load(p, allow_pickle=True) as z:
                if str(z["behaviour"]) == "speed":
                    keep.append(p)
        train_paths = keep
    BX, BG, BY, goals = [], [], [], []
    for p in train_paths:
        clip = load(p, B1)
        s = body_state(p)
        a = np.asarray(clip["actions"], dtype=np.float32)
        n = min(len(s), len(a))
        g = body_goal(p, "b1", channels)
        BX.append(torch.from_numpy(s[:n])); BY.append(torch.from_numpy(a[:n]))
        BG.append(torch.from_numpy(g).float().expand(n, -1))
        goals.append((p, g))
    BX = torch.cat(BX).to(device); BG = torch.cat(BG).to(device); BY = torch.cat(BY).to(device)
    print(f"{len(goals)} training goals (clone_b1's own split); cloning buffer {len(BX)} pairs")

    env = B1MuJoCoEnv(args.base_ckpt, goal_std=np.zeros(3, dtype=np.float32), horizon=args.steps + 5,
                      model_path=args.model_path)
    encoder = VJEPA2FrameEncoder(device=str(device), dtype=torch.float32)

    # --- CoppeliaSim: the camera, same egocentric geometry beh12_b1_ego_flat was rendered with ---
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    sim.loadScene(os.path.abspath(os.path.join(ROOT, args.scene)))
    settle(sim)
    jm = {sim.getObjectAlias(h): h
          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    joints = [jm[a] for a in JOINT_ALIASES_SDK]
    sm = {sim.getObjectAlias(h): h
          for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    root = sm[ROOT_ALIAS]
    cam = sim.getObject("/" + SENSOR)

    R = room_for(sim.getObjectPosition(root, sim.handle_world)[2])
    build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=args.ego_seed)
    randomise_ground(sim, seed=args.ego_seed, uv=R["ground_uv"])
    floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
              if sim.getObjectAlias(h, 1).startswith("/Floor")]
    if floors:
        top = sim.getObject("/Floor")

        def surface():
            q = sim.getObjectPosition(top, sim.handle_world)
            bb = sim.getShapeBB(top)
            return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
        before = surface()
        sim.scaleObjects(floors, float(args.floor_scale), False)
        drop = surface() - before
        for h in floors:
            q = sim.getObjectPosition(h, sim.handle_world)
            sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - drop])
    sim.setObjectFloatParam(cam, sim.visionfloatparam_perspective_angle, float(np.deg2rad(args.cam_fov)))

    obs = env.reset()
    psi0 = float(heading(env.d.qpos[3:7][None].copy(), "b1")[0])
    fwd = [float(np.cos(psi0)), float(np.sin(psi0)), 0.0]
    attach_ego(sim, cam, root, fwd, offset_frac=R["offset_frac"], pitch_comp=WALK_PITCH["b1"])
    print(f"ego camera attached, room {R['size']:.2f} m, forward heading {np.degrees(psi0):+.1f} deg")

    def render():
        p = env.d.qpos[0:3]
        w, x, y, z = env.d.qpos[3:7]
        sim.setObjectPosition(root, sim.handle_world, [float(p[0]), float(p[1]), float(p[2])])
        sim.setObjectQuaternion(root, sim.handle_world, [float(x), float(y), float(z), float(w)])
        for h, a in zip(joints, env.d.qpos[7:19]):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)

    sigma = (args.sigma * student.std).to(device)
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    TX, TG, TY = [], [], []

    for it in range(args.iters):
        gp, g = goals[int(rng.integers(len(goals)))]
        g_t = torch.tensor(g, dtype=torch.float32, device=device).unsqueeze(0)
        obs = env.reset()
        frame = render()
        heads = [env.d.qpos[0:3].copy()]
        it_TX, it_TG, it_TY = [], [], []
        for t in range(args.steps):
            with torch.no_grad():
                e_full = encode_clip(encoder, np.asarray(frame)[None], 1).float().to(device)
                if offset is not None:
                    e_full = e_full - offset.to(device)
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                base_a = student.act(s_t, g_t)
                cand = base_a + sigma * torch.randn(args.samples, base_a.shape[-1], device=device)
                cand = torch.cat([base_a, cand])          # keep the student's own choice
                z = proj(cand, "b1")
                roll = e_full.expand(len(cand), -1, -1)
                for _ in range(args.horizon):
                    roll = ftm(roll, z)
                motion = md.body(None, itm(e_full.expand(len(cand), -1, -1), roll))
                k = min(motion.shape[-1], len(channels))
                err = (motion[:, :k] - ((g_t[:, :k] - mean_s[:k]) / std_s[:k])).pow(2).mean(-1)
                best = cand[int(err.argmin())]
            it_TX.append(s_t[0].cpu()); it_TG.append(g_t[0].cpu()); it_TY.append(best.cpu())
            apply_action_unclipped(env, base_a[0].cpu().numpy())   # advance with the student's OWN
            obs = env._obs()                                       # action, not the label -- DAgger
            frame = render()
            heads.append(env.d.qpos[0:3].copy())
        TX += it_TX; TG += it_TG; TY += it_TY
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
              f"travelled {d:.4f} m  labelled {len(it_TX)}  buffer {len(TX)}  "
              f"loss {loss.item():.4f}", flush=True)

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({**st, "student": student.state_dict(), "base_ckpt": args.base_ckpt,
               "iters": args.iters, "horizon": args.horizon, "samples": args.samples,
               "sigma": args.sigma}, out)
    print(f"-> {args.out}")


def evaluate_b1(args, device):
    ck = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(ck["state_dim"], ck["goal_dim"], ck["action_dim"]).to(device).eval()
    student.load_state_dict(ck["student"])
    channels = list(ck["channels"])

    goal_path = os.path.join(ROOT, args.goal_clip)
    goal = body_goal(goal_path, "b1", channels)
    goal_clip = load(goal_path, B1)
    goal_actions = np.asarray(goal_clip["actions"], dtype=np.float32)

    # D_real: replay the goal clip's OWN recorded actions open-loop, through the SAME model/reset
    # `eval` uses (b1_flat_real.xml, via B1MuJoCoEnv) -- not the clip's own stored base_pos, which
    # reflects the different (uniform-physics) model it was originally collected on. See module
    # docstring: this is a real, stated train/eval physics gap.
    env = B1MuJoCoEnv(args.base_ckpt, goal_std=np.zeros(3, dtype=np.float32), horizon=len(goal_actions) + 5,
                      model_path=args.model_path)
    heads_real = rollout_open_loop(env, goal_actions, clip_path=goal_path)
    d_real = float(np.linalg.norm(heads_real[-1, :2] - heads_real[0, :2]))
    print(f"D_real (replayed under eval physics) = {d_real:.4f} m, bar = {0.5 * d_real:.4f} m, "
         f"goal {np.round(goal, 4)} from {os.path.basename(args.goal_clip)}")

    renderer, frames = None, []
    if args.video:
        import mujoco
        renderer = mujoco.Renderer(env.m, 480, 640)
    heads = rollout_student(env, student, goal, device, args.steps, renderer, frames)
    if frames:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.join(ROOT, args.video)) or ".", exist_ok=True)
        imageio.mimsave(os.path.join(ROOT, args.video), frames, fps=int(round(1 / env.dt)))
        print(f"video: {args.video}")
    v = verdict(heads, d_real)
    print(f"\n  travelled {v['distance']:.4f} m = {v['fraction']:.0%} of D_real"
         f"   upright {v['upright']} (min head z {v['min_z']:.4f} against {v['z0']:.4f})")
    print(f"  **{'PASS' if v['pass'] else 'FAIL'}**")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, head=heads.astype(np.float32), d_real=np.float32(d_real),
                        goal=goal.astype(np.float32),
                        **{k: v[k] for k in ("distance", "fraction", "upright", "pass")})
    print(f"-> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("bc", "bc_aux", "improve", "distill", "eval"))
    ap.add_argument("--base_ckpt", default="wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/teacher_b1.pt",
                    help="For `bc`/`eval`: only `cfg.body_channels` is read (or, for eval, used to "
                         "construct B1MuJoCoEnv with goal_std zeroed out) -- body_head's own fit "
                         "quality never matters there, so `teacher_b1.pt` (no `body_head_fit` "
                         "metadata, i.e. an unfit body_head) is fine. For `improve`, this checkpoint "
                         "IS the grading teacher (`body_head(ITM(.))` scores every candidate), so "
                         "pass `.../b1_adapt_hinge/body_head_b1_hex_v2.pt` (the one with a real "
                         "body_head_fit) explicitly -- F214 found this exact mixup in two other "
                         "scripts, checked here at startup rather than trusted.")
    ap.add_argument("--student", default="wm/runs/students/b1_bc_forward.pt")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--goal_clip", default=f"{DATA}/b1_ep100.npz")
    ap.add_argument("--forward_only", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="bc/improve: checkpoint to write, default wm/runs/students/b1_bc_forward.pt. "
                         "eval: results .npz to write, default derived from --student's own name -- "
                         "left shared with bc's default this silently overwrote wm/runs/students/"
                         "b1_bc_forward.pt.npz next to the real checkpoint of nearly the same name.")
    ap.add_argument("--video", default=None, help="optional .mp4 of the eval rollout")
    ap.add_argument("--model_path", default=None,
                    help="override B1MuJoCoEnv's default model (b1_flat_real.xml, system-"
                         "identified). Pass sim/assets/b1_mujoco/b1_flat.xml (uniform placeholder "
                         "physics, what beh12_b1_ego_flat was actually collected on) to check "
                         "whether that matches the student's cloned data better.")
    ap.add_argument("--port", type=int, default=23000, help="improve only: CoppeliaSim ZMQ port")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt", help="improve only")
    ap.add_argument("--ego_seed", type=int, default=0,
                    help="improve only: room appearance seed, fixed for the whole run -- this loop "
                         "is not reproducing a training clip's own room, just staying in the same "
                         "rendering domain body_head was fit on")
    ap.add_argument("--floor_scale", type=float, default=3.0, help="improve only, matches render_b1_replay.py")
    ap.add_argument("--cam_fov", type=float, default=90.0, help="improve only, matches the ego default")
    ap.add_argument("--iters", type=int, default=30, help="improve only: DAgger rounds")
    ap.add_argument("--samples", type=int, default=15, help="improve only: perturbations per step")
    ap.add_argument("--sigma", type=float, default=0.3, help="improve only: perturbation scale, x student.std")
    ap.add_argument("--horizon", type=int, default=3, help="improve only: FTM imagination steps (F131: <=3)")
    ap.add_argument("--refit", type=int, default=50, help="improve only: gradient steps per DAgger round")
    ap.add_argument("--reference", default="wm/runs/students/b1_bc_all.pt",
                    help="distill only: the checkpoint whose --val_paths split to reuse, so the "
                         "distilled student and the plain-BC one are compared on the same held-out "
                         "clips")
    ap.add_argument("--batch", type=int, default=48,
                    help="bc_aux only: examples per step -- the aux loss forwards through the "
                         "frozen ITM/FTM/body_head every step, so this trains on sampled minibatches "
                         "rather than the full dataset per step the way bc/distill do")
    ap.add_argument("--lambda_aux", type=float, default=1.0,
                    help="bc_aux only: weight on the auxiliary model-consistency loss relative to "
                         "the plain BC MSE. Not tuned -- a single principled first run, per the "
                         "pre-registered test this stage exists to run")
    ap.add_argument("--cpu_models", action="store_true",
                    help="bc_aux only: run ITM/FTM/proj/body_head and the whole training loop on "
                         "CPU (only the one-time video encoding stays on GPU). Last resort for an "
                         "11 GB card with no VRAM headroom left after the encoder -- measured to "
                         "cost 54 minutes and 21.5 GB RAM without finishing, so prefer running on "
                         "BIAS-2/com7 with this flag OFF (everything on GPU) instead of using it.")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("wm/runs/students/b1_bc_forward.pt" if args.stage not in ("eval", "bc_aux") else
                   (f"results/wm/{os.path.splitext(os.path.basename(args.student))[0]}_eval.npz"
                    if args.stage == "eval" else "wm/runs/students/b1_bc_aux_forward.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "bc":
        clone_b1(args, device)
    elif args.stage == "bc_aux":
        ck_peek = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
        assert "body_head_fit" in ck_peek and ck_peek["body_head_fit"].get("embodiment") == "b1", (
            f"--base_ckpt {args.base_ckpt} has no fitted body_head for b1 (F214) -- "
            f"pass body_head_b1_hex_v2.pt, not teacher_b1.pt")
        del ck_peek
        aux_bc_b1(args, device)
    elif args.stage in ("improve", "distill"):
        ck_peek = torch.load(os.path.join(ROOT, args.base_ckpt), map_location="cpu", weights_only=False)
        assert "body_head_fit" in ck_peek and ck_peek["body_head_fit"].get("embodiment") == "b1", (
            f"--base_ckpt {args.base_ckpt} has no fitted body_head for b1 (F214) -- "
            f"pass body_head_b1_hex_v2.pt, not teacher_b1.pt")
        del ck_peek
        (improve_b1 if args.stage == "improve" else distill_b1)(args, device)
    else:
        evaluate_b1(args, device)


if __name__ == "__main__":
    main()
