"""Per-channel accuracy of reading a goal from video, against the same goal read from privileged
telemetry -- direct, not routed through candidate-selection/family-classification.

**Why this exists.** The 2x2x2 test's family-accuracy metric classifies each goal by whichever
Froude channel has the LARGEST magnitude, which buries yaw: a turn-family clip's forward component
(~0.13) is bigger than its yaw component (~0.005-0.08), so it gets classified and scored as a
FORWARD goal, and "turn goals: 32/32 forward-family, yaw goals: none in this set" told us nothing
about whether yaw itself is read well. This compares mode-A (physics, privileged) and mode-D
(vision) goal vectors DIRECTLY, per channel, with no classification step in between -- the question
this answers is exactly "reads forward from pixels well, yaw poorly?", not "which family wins."

    .venv/bin/python3 scripts/diagnostics/objective_experiments/per_channel_goal_read.py
"""
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics", "objective_experiments"))
from final_2x2x2_test import build_planner, load_goals  # noqa: E402

CKPT = "wm/runs/beh12_hinge_multistep_anchor_v2/b1_adapt_hinge/body_head_b1_hex_v2.pt"
CANDIDATES = "data/egocentric/beh12_b1_ego_flat"
GOAL_DIR = "data/egocentric/beh12_c10f10t10_ego_flat"
CHANNELS = ["fwd", "lat", "yaw"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner = build_planner(os.path.join(ROOT, CKPT), os.path.join(ROOT, CANDIDATES), "b1",
                            horizon=5, free_offset=False, device=device, per_condition=1)

    goals_true = {g["condition"]: g["froude"] for g in
                 load_goals(GOAL_DIR, "hexapod", planner, "physics", 1, device)}
    goals_vision = {g["condition"]: g["froude"] for g in
                    load_goals(GOAL_DIR, "hexapod", planner, "vision", 1, device)}

    by_behaviour = {}
    for p in sorted(glob.glob(os.path.join(ROOT, GOAL_DIR, "*.npz"))):
        with np.load(p, allow_pickle=True) as d:
            cond, beh = str(d["condition"]), str(d["behaviour"])
        by_behaviour.setdefault(cond, beh)

    print(f"{'condition':>16}{'behaviour':>10}  " +
         "".join(f"{'true_' + c:>10}{'vision_' + c:>10}{'|err|_' + c:>10}" for c in CHANNELS))
    errs = {c: [] for c in CHANNELS}
    errs_by_beh = {}
    for cond in sorted(goals_true):
        t, v = goals_true[cond], goals_vision[cond]
        beh = by_behaviour.get(cond, "?")
        row = f"{cond:>16}{beh:>10}  "
        for i, c in enumerate(CHANNELS):
            e = abs(v[i] - t[i])
            row += f"{t[i]:>10.4f}{v[i]:>10.4f}{e:>10.4f}"
            errs[c].append(e)
            errs_by_beh.setdefault((beh, c), []).append(e)
        print(row)

    print(f"\n{'channel':>10}{'mean |err|':>12}{'median |err|':>14}{'mean |true| (scale)':>22}"
         f"{'relative error':>16}")
    for c in CHANNELS:
        e = np.asarray(errs[c])
        scale = np.mean([abs(goals_true[cond][CHANNELS.index(c)]) for cond in goals_true])
        print(f"{c:>10}{e.mean():>12.4f}{np.median(e):>14.4f}{scale:>22.4f}{e.mean() / max(scale, 1e-9):>16.2f}x")

    print(f"\n{'behaviour':>10}{'channel':>10}{'mean |err|':>12}{'n':>4}")
    for (beh, c), e in sorted(errs_by_beh.items()):
        e = np.asarray(e)
        print(f"{beh:>10}{c:>10}{e.mean():>12.4f}{len(e):>4}")

    print("\nrelative error = mean|err| / mean|true value| for that channel -- >1x means the "
         "read error is bigger than the signal itself; this is the number that decides whether "
         "'reads Froude from pixels' is really 'reads forward from pixels, yaw not established'.")


if __name__ == "__main__":
    main()
