"""Does the FTM predict the TRUE outcome of an action taken from a state it did not come from?

Every other check of `lambda_cycle` reads the FTM's prediction through the ITM, the same module the
cycle term trains against, so the FTM could in principle learn a code only that ITM decodes. This
check never touches the ITM. It renders real counterfactual outcomes in CoppeliaSim:

  state s at t   the B1 posed as library clip s at step t (same room, same 90-deg ego camera)
  e*(s, a)       from that state, apply candidate a's own one-step motion (planar delta and heading
                 change composed onto s, height/pitch/roll and joint angles from a at t+1 -- exactly
                 what the kinematic closed loop does when it executes a from s), render, encode
  pred(s, a)     FTM(e_s, proj(a's action at t+lag)), as selection computes it

and scores, per state and step, over all 24 actions:
  top1     pred(s, a) is nearer e*(s, a) than every other action's outcome e*(s, a')  (chance 1/24)
  rank     mean rank of the true outcome among the 24 (chance 12.5, best 1)
  gain     mean over a of  d(pred(s,a), e*(s,a')) - d(pred(s,a), e*(s,a)), a' != a, normalised by
           the spread of the outcomes: > 0 means predictions move toward the right outcome

    .venv/bin/python3 scripts/diagnostics/objective_experiments/counterfactual_truth_check.py \\
        --ckpt ctrl_fz=wm/runs/beh24_ft_ctrl_fz/b1_lora_c3/ckpt_lib_insample.pt \\
        --ckpt cycle_fz=wm/runs/beh24_ft_cycle_fz/b1_lora_c3/ckpt_lib_insample.pt

Needs CoppeliaSim on --port (headless is fine).
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in ("", "scripts", "sim/render", "sim/scene", "sim/control", "scripts/diagnostics/objective_experiments"):
    sys.path.insert(0, os.path.join(ROOT, p))

from coppeliasim_zmqremoteapi_client import RemoteAPIClient  # noqa: E402
from ego_camera import (EGO_FOV_DEG, WALK_PITCH, attach_ego, build_texture_box,  # noqa: E402
                        check_ego_view, randomise_ground, room_for)
from render_b1_replay import JOINT_ALIASES_SDK, ROOT_ALIAS, capture, settle  # noqa: E402
from close_loop_direct_froude import load_motion, qz, quat_mul  # noqa: E402
from rollout_state_action_anova import Models  # noqa: E402
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402
from wm.data.embodiment import heading  # noqa: E402
from wm.policy.planner import load_candidates  # noqa: E402


def build_scene(sim, scene):
    sim.loadScene(os.path.abspath(os.path.join(ROOT, scene)))
    settle(sim)
    jm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_joint_type)}
    sm = {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)}
    joints, root = [jm[a] for a in JOINT_ALIASES_SDK], sm[ROOT_ALIAS]
    root0 = np.array(sim.getObjectPosition(root, sim.handle_world))
    cam = sim.createVisionSensor(1 | 2 | 4, [256, 256, 0, 0],
                                 [0.01, 20.0, np.deg2rad(EGO_FOV_DEG), 0.05, 0, 0, 0, 0, 0, 0, 0])
    R = room_for(root0[2])
    build_texture_box(sim, size=R["size"], height=R["height"], tile=R["tile"], seed=0, centre=(0.0, 0.0))
    sim.setObjectPosition(root, sim.handle_world, [0.0, 0.0, float(root0[2])])
    sim.setObjectQuaternion(root, sim.handle_world, [0.0, 0.0, 0.0, 1.0])
    attach_ego(sim, cam, root, [1.0, 0.0, 0.0], (0, 0, 0), offset_frac=R["offset_frac"],
               pitch_comp=WALK_PITCH["b1"])
    floors = [h for h in sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type)
              if sim.getObjectAlias(h, 1).startswith("/Floor")]
    if floors:
        top = sim.getObject("/Floor")

        def surface():
            q = sim.getObjectPosition(top, sim.handle_world)
            bb = sim.getShapeBB(top)
            return q[2] + (bb[0] if isinstance(bb[0], list) else bb)[2] / 2
        before = surface()
        sim.scaleObjects(floors, 3.0, False)
        drop = surface() - before
        for h in floors:
            q = sim.getObjectPosition(h, sim.handle_world)
            sim.setObjectPosition(h, sim.handle_world, [q[0], q[1], q[2] - drop])
    randomise_ground(sim, seed=0, uv=R["ground_uv"])

    def pose(pos, quat, jangles):
        sim.setObjectPosition(root, sim.handle_world, [float(v) for v in pos])
        sim.setObjectQuaternion(root, sim.handle_world,
                                [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])])
        for h, a in zip(joints, jangles):
            sim.setJointPosition(h, float(a))
        return capture(sim, cam)
    return pose


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--candidates_dir", default="data/egocentric/beh12_b1_ego_flat_cleantrain")
    ap.add_argument("--scene", default="sim/env/b1_flat.ttt")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--states", type=int, default=8, help="how many library clips serve as states")
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--cache", default="results/wm/cache/counterfactual_truth.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cands = load_candidates(os.path.join(ROOT, args.candidates_dir), "b1", per_condition=999)
    mot = [load_motion(c["path"]) for c in cands]
    n = len(cands)
    S = np.linspace(0, n - 1, args.states).round().astype(int)

    cache = os.path.join(ROOT, args.cache)
    if os.path.exists(cache):
        E = torch.load(cache)
        print(f"rendered outcomes from cache: {cache}")
    else:
        sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
        pose = build_scene(sim, args.scene)
        enc = VJEPA2FrameEncoder(dtype=torch.float32)
        E = {}
        ref = [np.load(c["path"], allow_pickle=True)["frames"][0] for c in cands[:12]]
        for t in args.steps:
            for s in S:
                ms = mot[s]
                p_s = ms["pos"][t] - ms["pos"][0]
                p_s[2] = ms["pos"][t, 2]
                psi_s = float(ms["psi"][t])
                # the state, heading-normalised so every state looks down the same room axis
                q_s = quat_mul(qz(-float(ms["psi"][0])), ms["quat"][t])
                p_s[:2] = np.array([[np.cos(-ms["psi"][0]), -np.sin(-ms["psi"][0])],
                                    [np.sin(-ms["psi"][0]), np.cos(-ms["psi"][0])]]) @ p_s[:2]
                psi_s = float(heading(q_s[None], "b1")[0])
                img_s = pose(p_s, q_s, ms["jpos"][t])
                if t == args.steps[0] and s == S[0]:
                    print(f"view check vs library frame 0: corr {check_ego_view(img_s, ref, min_corr=0.9):.3f}")
                frames = [img_s]
                for a in range(n):
                    ma = mot[a]
                    d = ma["pos"][t + 1, :2] - ma["pos"][t, :2]
                    rot = psi_s - float(ma["psi"][t])
                    pos = np.array([p_s[0] + np.cos(rot) * d[0] - np.sin(rot) * d[1],
                                    p_s[1] + np.sin(rot) * d[0] + np.cos(rot) * d[1], ma["pos"][t + 1, 2]])
                    dpsi = float(ma["psi"][t + 1] - ma["psi"][t])
                    psi = psi_s + float(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
                    quat = quat_mul(qz(psi - float(ma["psi"][t + 1])), ma["quat"][t + 1])
                    frames.append(pose(pos, quat / np.linalg.norm(quat), ma["jpos"][t + 1]))
                emb = encode_clip(enc, np.stack(frames), 2).float().cpu().half()                     # (1 + n, tokens, dim)
                E[(t, int(s))] = emb
            print(f"  rendered step {t}", flush=True)
        torch.save(E, cache)
        del enc
        torch.cuda.empty_cache()

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m = Models(os.path.join(ROOT, path), "b1", cands[0]["actions"].shape[1], dev)
        top1, rank, gain = [], [], []
        for (t, s), emb in E.items():
            e = emb.float().to(dev)
            if m.offset is not None:
                e = e - m.offset.float().reshape(e.shape[1:]).to(dev)
            e_s, truth = e[:1], e[1:]                                             # truth: (n, tok, dim)
            a = torch.as_tensor(np.stack([c["actions"][t + m.action_lag] for c in cands]), device=dev)
            pred = m.ftm_step(e_s.expand(n, -1, -1), m.proj(a, "b1"))
            D = torch.cdist(pred.flatten(1), truth.flatten(1))                     # (pred a, outcome a')
            diag = D.diag()
            top1.append((D.argmin(1) == torch.arange(n, device=dev)).float().mean().item())
            rank.append(((D < diag[:, None]).sum(1).float() + 1).mean().item())
            spread = torch.cdist(truth.flatten(1), truth.flatten(1)).sum() / (n * (n - 1))
            off = (D.sum(1) - diag) / (n - 1)
            gain.append(((off - diag) / spread).mean().item())
        print(f"{name:10s} top1 {np.mean(top1):.3f} (chance {1 / n:.3f})  mean rank {np.mean(rank):.2f} "
              f"(chance {(n + 1) / 2:.1f})  gain {np.mean(gain):+.4f}  [{len(top1)} state-steps]")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
