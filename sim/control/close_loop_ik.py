"""Close the loop: the world model chooses what the insect does, step by step, from vision alone.

    camera ──► V-JEPA2 ──► e_t ──► planner scores every candidate behaviour
                                   ──► executes the winner's command for this step
                                   ──► CoppeliaSim steps ──► new frame

**Nothing here computes `z` from two frames.** The latent reaches the forward model only through
the action projector, because at control time the next frame is the thing being decided. Every
other number in this project reads `z` off a recorded transition and is therefore reconstruction;
this is the one path that could actually run on a robot.

**The goal is a demonstration clip.** At step `t` the planner is asked to reach the demonstration's
frame `t + horizon`. Success is not "did it copy the demonstration's joint angles" -- it is
`scripts/diagnostics/score_closed_loop.py`'s three criteria: achieved Froude within 15% of the
demonstrated one, the right behaviour class, and the body still up.

**Why the driving loop is `collect_ik.drive_and_record` rather than a new one.** Scene loading,
camera placement, the settle, the force sensors, the abdomen-quaternion convention and the
`--cam_dx -0.6` framing fix are each a measured decision with a finding behind them; a second
implementation would reproduce the bugs they were written to fix. It takes a `policy` hook for
exactly this.

  .venv/bin/python3 sim/control/close_loop_ik.py --ckpt wm/runs/beh12_hexonly/best.pt --demo <clip.npz>
"""
import argparse
import os
import sys

import numpy as np
import torch
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from sim.collect.collect_ik import EP, LEGS, SEG, drive_and_record  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import LatentPlanner, condition_of  # noqa: E402


def encode_one(encoder, frame, offset, device):
    """One frame to one embedding, through the same path the training data was built with."""
    e = encode_clip(encoder, np.asarray(frame)[None], 1).float()
    if offset is not None:
        e = e - offset.to(e.device)
    return e[0:1].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--projector", default="")
    ap.add_argument("--demo", required=True, help="clip whose behaviour the planner must reproduce")
    ap.add_argument("--candidates_dir", default="data/beh12_hex_flat")
    ap.add_argument("--scene", default="medauroidea_stick_insect.ttt")
    ap.add_argument("--morph", default="",
                    help="body label written into the output, and the name collect_ik registers "
                         "the scene under. Defaults to the scene filename's body, so a run on a "
                         "held-out body is not recorded as the body it was held out from.")
    ap.add_argument("--embodiment", default="hexapod")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--warm_start", type=int, default=0,
                    help="execute the demonstration's own commands for this many steps before the "
                         "planner takes over. **The robot is standing still after warmup and there "
                         "is no motion to read**: measured, 80-100% of the first five picks are a "
                         "turn on a forward demonstration. This separates that cold start from the "
                         "covariate shift that follows it.")
    ap.add_argument("--commit", type=int, default=1,
                    help="hold a chosen behaviour for this many steps before reconsidering. "
                         "**1 is replan-every-step and it dithers**: measured at 9-15 changes "
                         "across 20 steps, which cuts the stride mid-cycle and costs 21-34% of "
                         "the commanded speed. A receding-horizon controller commits for the same "
                         "reason.")
    ap.add_argument("--steps", type=int, default=EP)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--travel", type=float, default=0.8)
    # Not cosmetic: with the authored offset alone the robot walks out of the right image edge and
    # 56-70% of frames are clipped, which silently changes what the encoder sees. See collect_ik.
    ap.add_argument("--cam_dx", type=float, default=-0.6)
    ap.add_argument("--cam_dy", type=float, default=0.0)
    ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"))
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the same demonstration N times, saved as _r0.._rN. **The planner is "
                         "deterministic** -- argmin over frozen forward passes -- so everything "
                         "that varies between repeats is the simulator. One run cannot separate a "
                         "change in the method from physics, and the first two runs disagreed by "
                         "14 points of speed error.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = os.path.join(ROOT, args.ckpt)
    planner = LatentPlanner.from_checkpoint(
        ckpt_path, os.path.join(ROOT, args.candidates_dir), args.embodiment,
        os.path.join(ROOT, args.projector) if args.projector else "",
        horizon=args.horizon, device=str(device))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    offset = offset_for(checkpoint, args.embodiment)

    # `medauroidea_c08f09t09.ttt` -> `c08f09t09`; falls back to the default body for the
    # unsuffixed scenes. **Hard-coding this was a real bug for a held-out-body run**: every clip
    # would have carried `morph=c10f10t10`, and `wm/data/dataset.py` reads the body from that.
    stem = os.path.splitext(os.path.basename(args.scene))[0]
    morph = args.morph or (stem.split("medauroidea_", 1)[-1]
                           if stem.startswith("medauroidea_") and "stick_insect" not in stem
                           else "c10f10t10")

    demo_path = args.demo if os.path.isabs(args.demo) else os.path.join(ROOT, args.demo)
    with np.load(demo_path, allow_pickle=True) as data:
        demo_frames = data["frames"]
    want = condition_of(demo_path)

    encoder = VJEPA2FrameEncoder(dtype=torch.float32)
    demo_e = encode_clip(encoder, demo_frames, 2).float()
    if offset is not None:
        demo_e = demo_e - offset.to(demo_e.device)

    steps = min(args.steps, len(demo_e) - args.horizon - planner.action_lag - 1)
    if steps < 2:
        raise SystemExit(f"demonstration is too short: {len(demo_e)} frames")
    print(f"demonstration {os.path.basename(demo_path)}  condition {want}")
    print(f"{len(planner.candidates)} candidates, horizon {args.horizon}, {steps} steps\n")

    chosen, scores_log = [], []
    held = {"index": None, "since": 0}

    def policy(frame, t):
        if t < args.warm_start:
            chosen.append(f"warm:{want}")
            scores_log.append(np.zeros(len(planner.candidates), np.float32))
            return seed_cmds[min(t, len(seed_cmds) - 1)]
        h = planner.horizon_at(t)
        # Within a commitment window the behaviour is fixed, so there is nothing to decide and
        # nothing to encode -- which also makes `--commit` proportionally cheaper to run.
        if held["index"] is not None and t - held["since"] < args.commit:
            i = held["index"]
            scores_log.append(scores_log[-1])
        else:
            e_t = encode_one(encoder, frame, offset, device)
            _, i, scores = planner.act(e_t, demo_e[min(t + h, len(demo_e) - 1)], t)
            scores_log.append(scores)
            if i != held["index"]:
                held["since"] = t
            held["index"] = i
        cand = planner.candidates[i]
        chosen.append(cand["condition"])
        if t % 10 == 0:
            print(f"  step {t:3d}  -> {cand['condition']}", flush=True)
        return cand["actions"][min(t, len(cand["actions"]) - 1)]

    # `cmds` supplies the clip length and the pose held during warmup; `policy` replaces every
    # value inside the loop. Seeded from the demonstration so the robot starts in its stance.
    with np.load(demo_path, allow_pickle=True) as data:
        seed_cmds = data["actions"].astype(np.float32)[:steps]

    client = RemoteAPIClient("localhost", port=args.port)
    sim = client.getObject("sim")
    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(demo_path))[0]

    for r in range(args.repeat):
        chosen.clear()
        scores_log.clear()
        held["index"], held["since"] = None, 0
        if args.repeat > 1:
            print(f"--- repeat {r + 1}/{args.repeat}", flush=True)
        frames, actions, forces, heads, oris = drive_and_record(
            sim, args.scene, seed_cmds, args.travel, args.warmup,
            cam_dx=args.cam_dx, cam_dy=args.cam_dy, spawn=tuple(args.spawn), policy=policy)

        suffix = f"_r{r}" if args.repeat > 1 else ""
        out = os.path.join(out_dir, f"closed_{stem}{suffix}.npz")
        np.savez_compressed(
            out, frames=np.asarray(frames, np.uint8), actions=np.asarray(actions, np.float32),
            forces=np.asarray(forces, np.float32), head=np.asarray(heads, np.float32),
            body_quat=np.asarray(oris, np.float32), dt=np.float32(0.05),
            morph=morph, embodiment=args.embodiment,
            condition=want, chosen=np.asarray(chosen),
            candidates=np.asarray([c["condition"] for c in planner.candidates]),
            scores=np.asarray(scores_log, np.float32), demo=os.path.basename(demo_path))
        planned = [c for c in chosen if not c.startswith("warm:")]
        picked, counts = np.unique(planned or chosen, return_counts=True)
        switches = int(np.sum(np.asarray(chosen[1:]) != np.asarray(chosen[:-1])))
        turn = float(np.mean([c.startswith("turn") for c in planned])) if planned else float("nan")
        print(f"held {picked[np.argmax(counts)]} for {counts.max()}/{len(planned or chosen)} "
              f"planned steps, {switches} switches, {turn:.0%} turn picks"
              f"   (demonstration was {want})")
        print(f"-> {os.path.relpath(out, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
