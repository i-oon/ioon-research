"""Physics closed loop on a hexapod body (e.g. the held-out c08f09t09), Froude-goal selection.

The hexapod twin of `close_loop_direct_froude.py`, with one difference that makes it stricter: the
insect is **simulated**, not posed. `collect_ik.drive_and_record` steps CoppeliaSim with joint
targets and calls `policy(frame, t)` for each command, so the planner picks a candidate from the
body's own library, the chosen candidate's recorded joint command at t is sent to the joints, and
the achieved Froude is measured from how the body actually moved (it can slip, drift or fall).

  every --replan_every steps   direct: planner.act(goal_t, t)
                               rollout: planner.act(encode(ego frame), goal_t, t)
  every step                   command = chosen candidate's actions[t]
  grading                      L2(achieved body_motion[t], goal body_motion[t]) at decision steps

Egocentric camera exactly as the collector's (`--view egocentric`, 90 deg set after
`startSimulation`, room around the spawn, `ego_seed`); the first frame must pass
`ego_camera.check_ego_view` against the candidate library's frames (F256).

    .venv/bin/python3 sim/control/close_loop_hexapod_froude.py --mechanism rollout --window 11 \\
        --ckpt wm/runs/beh24_stride5_cleansplit/c08_zeroshot/ckpt_lib_zeroshot.pt \\
        --goal data/egocentric/beh12_c10f10t10_ego_flat_cleanheldout/hexapod_ep302.npz \\
        --candidates_dir data/egocentric/beh12_c08f09t09_ego_flat --morph c08f09t09=medauroidea_c08f09t09.ttt
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in ("", "scripts", "sim/collect", "sim/scene", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
from collect_ik import drive_and_record  # noqa: E402
from ego_camera import EGO_FOV_DEG, check_ego_view  # noqa: E402
from final_2x2x2_test import build_planner  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip, offset_for  # noqa: E402
from wm.policy.planner import RolloutFroudePlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", choices=("direct", "rollout", "random"), default="rollout",
                    help="random: a uniformly random library candidate at every decision (seeded by "
                         "the goal), the physics reference a selector has to beat")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", required=True, help="goal clip (hexapod), read as recorded body_motion per step")
    ap.add_argument("--candidates_dir", required=True)
    ap.add_argument("--morph", required=True, help="NAME=SCENE of the controlled body")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--replan_every", type=int, default=2)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--phase_match", action="store_true",
                    help="execution only: on a switch, continue the new candidate from the frame whose "
                         "joint command is nearest the last one sent (its gait phase), not from index t. "
                         "Separates 'chose the right behaviour' from 'the switch broke the gait'.")
    ap.add_argument("--ego_seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--out", default="results/wm/closed_loop/hexapod_live")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    emb, dev = "hexapod", args.device
    ckpt = os.path.join(ROOT, args.ckpt)
    cdir = os.path.join(ROOT, args.candidates_dir)
    planner = build_planner(ckpt, cdir, emb, args.horizon, free_offset=False, device=dev, per_condition=999)
    planner.window = args.window
    rp, encoder, off = None, None, None
    if args.mechanism == "rollout":
        rp = RolloutFroudePlanner.from_checkpoint(ckpt, cdir, embodiment=emb, horizon=args.horizon,
                                                  per_condition=999, device=dev)
        rp.window = args.window
        from vjepa2_encoder import VJEPA2FrameEncoder
        encoder = VJEPA2FrameEncoder(dtype=torch.float32)
        off = offset_for(torch.load(ckpt, map_location="cpu", weights_only=False), emb)
    cands = planner.candidates
    goal_path = os.path.join(ROOT, args.goal)
    goal_bm = np.asarray(load(goal_path, REGISTRY[emb])["body_motion"])[:, :3]
    ref_frames = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
    steps = min(len(goal_bm), min(len(c["actions"]) for c in cands))
    seed_cmds = np.asarray(cands[0]["actions"], np.float32)[:steps]      # warmup pose + clip length

    held = {"i": 0, "off": 0, "last": None}
    rnd = np.random.default_rng(__import__("zlib").crc32(os.path.basename(goal_path).encode()))
    chosen, checked = [], {"done": False}

    def policy(frame, t):
        if not checked["done"]:
            r = check_ego_view(frame, ref_frames, what="hexapod closed-loop ego frame 0")
            print(f"  ego view check: row-profile corr {r:.3f}", flush=True)
            checked["done"] = True
        if t % args.replan_every == 0:
            g = torch.as_tensor(planner.standardize(goal_bm[min(t, len(goal_bm) - 1)]), dtype=torch.float32)
            with torch.no_grad():
                if args.mechanism == "random":
                    i = int(rnd.integers(len(cands)))
                elif rp is None:
                    _, i, _, _ = planner.act(g.numpy(), t)
                else:
                    e_t = encode_clip(encoder, np.asarray(frame)[None], 1).float()
                    if off is not None:
                        e_t = e_t - off.to(e_t.device)
                    _, i, _ = rp.act(e_t, g, t)
            if i != held["i"] and args.phase_match and held["last"] is not None:
                acts = np.asarray(cands[i]["actions"])
                tau = int(np.argmin(np.linalg.norm(acts[:-1] - held["last"], axis=1)))
                held["off"] = tau - t                  # continue the new clip from its matching phase
            elif i != held["i"]:
                held["off"] = 0
            held["i"] = i
        chosen.append(cands[held["i"]]["condition"])
        a = cands[held["i"]]["actions"]
        cmd = a[int(np.clip(t + held["off"], 0, len(a) - 1))]
        held["last"] = np.asarray(cmd)
        return cmd

    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    name, scene = args.morph.split("=", 1)
    frames, actions, forces, heads, oris = drive_and_record(
        sim, scene, seed_cmds, 0.0, args.warmup, spawn=(0.0, 0.0), ego=True, ego_seed=args.ego_seed,
        cam_fov=EGO_FOV_DEG, policy=policy)

    out_dir = os.path.join(ROOT, args.out, f"{args.mechanism}_w{args.window}" + ("_phase" if args.phase_match else ""))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(goal_path))[0]
    out = os.path.join(out_dir, f"{name}_{stem}.npz")
    np.savez_compressed(out, frames=frames, actions=actions, forces=forces, head=heads, body_quat=oris,
                        embodiment=emb, morph=name, chosen=np.asarray(chosen), goal=os.path.basename(goal_path),
                        mechanism=args.mechanism, window=args.window, dt=np.float32(0.05),
                        expert_episode=-1, condition="closed_loop", behaviour="closed_loop", level=-1)
    achieved = np.asarray(load(out, REGISTRY[emb])["body_motion"])[:, :3]
    n = min(len(achieved), len(goal_bm))
    dec = np.arange(0, n, args.replan_every)
    err = np.linalg.norm(achieved[dec] - goal_bm[dec], axis=1)
    with np.load(out, allow_pickle=True) as d:
        saved = {k: d[k] for k in d.files}
    np.savez_compressed(out, **saved, achieved_froude=achieved[:n].astype(np.float32),
                        goal_froude_t=goal_bm[:n].astype(np.float32))
    print(f"physics closed loop: mean L2 error {err.mean():.4f} at {len(dec)} decision steps -> "
          f"{os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
