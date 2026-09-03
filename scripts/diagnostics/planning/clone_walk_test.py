"""Does the egocentric clone walk further than the allocentric one? The behavioural row that is empty.

    .venv/bin/python3 scripts/diagnostics/planning/clone_walk_test.py \\
        --student wm/runs/students/insect_bc_ego.pt \\
        --goal_clip data/egocentric/beh12_c08f09t09_ego_flat/hexapod_ep100.npz \\
        --scene medauroidea_c08f09t09.ttt --ego --ego_seed 0

**Every number egocentric improved is internal to the model; every behavioural number is unchanged
or worse.** `null/real` 1.03 to 1.16, yaw readability 0.07 to 0.64 -- against ranking flat at chance,
coarse ranking slightly down, and the command 14% harder to read back. **The one row that would
settle it was never filled in**: F144 measured the *allocentric* clone walking 36% of the reference
distance, and no egocentric clone has ever been run in physics.

**The bar is re-measured here rather than borrowed, and that matters.** F142's `D_real = 0.6566 m`
was the reference clip replayed through the **base body** in the allocentric scene. This clone is
`c08f09t09` -- shorter coxa, femur and tibia -- in its own scene. **Reusing 0.6566 would divide this
body's distance by another body's reference**, which is the kind of borrowed denominator this project
has had to withdraw numbers for. So the recorded actions are replayed first, in the same body, the
same scene and the same camera configuration, and *that* is the denominator.

**Read it against F144's 36%, and read the failure modes separately.** Upright the whole window and
short is a gait that does not travel; falling is a different failure and the pre-registered rule
(F142) requires both to pass.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "sim", "collect"))
sys.path.insert(0, os.path.join(ROOT, "sim", "control"))
from vjepa2_encoder import VJEPA2FrameEncoder  # noqa: E402

from teacher_student_insect import Student, body_goal, pooled, verdict  # noqa: E402
from wm.data.embodiment import REGISTRY, load  # noqa: E402
from wm.evaluate import encode_clip  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_label_quality import EGO_CAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--goal_clip", required=True)
    ap.add_argument("--scene", default="medauroidea_c08f09t09.ttt")
    ap.add_argument("--ego", action="store_true")
    ap.add_argument("--ego_seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=23000)
    ap.add_argument("--steps", type=int, default=66)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sck = torch.load(os.path.join(ROOT, args.student), map_location="cpu", weights_only=False)
    student = Student(sck["token_dim"], sck["goal_dim"], sck["action_dim"]).to(device).eval()
    student.load_state_dict(sck["student"])
    channels = list(sck["channels"])
    goal = body_goal(os.path.join(ROOT, args.goal_clip), "hexapod", channels)
    g = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)

    clip = load(os.path.join(ROOT, args.goal_clip), REGISTRY["hexapod"])
    seed = np.asarray(clip["actions"], np.float32)
    encoder = VJEPA2FrameEncoder(dtype=torch.float32)

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    from collect_ik import drive_and_record
    sim = RemoteAPIClient("localhost", port=args.port).getObject("sim")
    ego = dict(EGO_CAM, ego_seed=args.ego_seed) if args.ego else {}

    def run(policy):
        return drive_and_record(sim, args.scene, seed[:args.steps], 0.0, 20,
                                cam_dx=-0.6, cam_dy=0.0, spawn=(0.0, 0.0),
                                policy=policy, **ego)

    print(f"student {args.student}\nbody/scene {args.scene}   "
          f"view {'egocentric' if args.ego else 'allocentric'}   "
          f"goal {np.round(goal, 4)} from {os.path.basename(args.goal_clip)}\n")

    # 1. the denominator, measured in THIS body, scene and camera
    _f, _a, _fo, heads_ref, _o = run(lambda obs, t: seed[min(t, len(seed) - 1)])
    d_real = float(np.linalg.norm(np.asarray(heads_ref)[-1, :2] - np.asarray(heads_ref)[0, :2]))
    print(f"  replayed reference in this body: D_real = {d_real:.4f} m   "
          f"(F142's base body read 0.6566)")

    # 2. the clone, driven from its own camera
    def policy(observation, t):
        with torch.no_grad():
            e = encode_clip(encoder, np.asarray(observation)[None], 1).float().to(device)
            return student.act(pooled(e), g)[0].cpu().numpy()

    _f, _a, _fo, heads, _o = run(policy)
    v = verdict(heads, d_real)
    print(f"  clone travelled {v['distance']:.4f} m = **{v['fraction']:.0%} of D_real**")
    print(f"  upright the whole window: {v['upright']}   "
          f"(min head z {v['min_z']:.4f} against {v['z0']:.4f} settled)")
    print(f"  **{'PASS' if v['pass'] else 'FAIL'}** -- both upright and >= 50% are required (F142)")

    print(f"\n  F144's allocentric clone, base body: 0.2349 m = 36%, upright, FAIL.")
    print("  **Above 36% is the first behavioural evidence that the viewpoint helps.** At or below")
    print("  it, the honest slide reads: the camera change made the model use the action and did")
    print("  not make the robot walk further -- which is reportable and stronger than not measuring.")
    print("\n  **The denominators differ by body**, so the metres are not comparable across the two")
    print("  runs and only the percentages are. That is why the reference was replayed here rather")
    print("  than taken from F142.")


if __name__ == "__main__":
    main()
