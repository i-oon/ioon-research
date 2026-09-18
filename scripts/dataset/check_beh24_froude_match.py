"""Does each hexapod condition's ACHIEVED Froude match its intended B1 counterpart's? Command
values are body-specific units (hex: gait cycle `c*`/turn `s*`; B1: `vx`/`w`) chosen BY
CALIBRATION to hit a shared Froude target, not to share a literal command value -- so this checks
the physics outcome, not the labels, and pairs conditions by rank within each behaviour/direction
group rather than assuming the command values line up.

    .venv/bin/python3 scripts/dataset/check_beh24_froude_match.py
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.data.embodiment import HEXAPOD, B1, load  # noqa: E402

SOURCES = {
    "hexapod": (HEXAPOD, "data/egocentric/beh24_c10f10t10_ego_flat"),
    "b1": (B1, "data/egocentric/beh24_b1_ego_flat"),
}
CHANNELS = ("forward", "lateral", "yaw")

# behaviour groups -> which Froude channel is the one that should rank-match, and whether the
# group is signed (so + and - variants pair separately) or unsigned (side levels share a name).
GROUPS = {
    "speed_fwd": ("forward", 0),
    "speed_bwd": ("forward", 0),
    "turn_pos": ("yaw", 2),
    "turn_neg": ("yaw", 2),
    "side_L": ("lateral", 1),
    "side_R": ("lateral", 1),
}


def condition_group(cond):
    if cond.startswith("speed_"):
        # hex marks backward with a `_bwd` suffix; B1 marks it with a `-` sign on the vx value
        # itself (`speed_vx-0.30`) -- two different conventions for the same distinction.
        is_bwd = cond.endswith("_bwd") or "vx-" in cond
        return "speed_bwd" if is_bwd else "speed_fwd"
    if cond.startswith("turn_") and cond.endswith("_neg"):
        return "turn_neg"
    if cond.startswith("turn_"):
        return "turn_pos"
    if cond.startswith("side_L"):
        return "side_L"
    if cond.startswith("side_R"):
        return "side_R"
    raise ValueError(cond)


def mean_froude_per_condition(spec, data_dir):
    by_cond = {}
    for f in sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz"))):
        clip = load(f, spec)
        motion = np.asarray(clip["body_motion"])
        with np.load(f, allow_pickle=True) as d:
            cond = str(d["condition"])
        by_cond.setdefault(cond, []).append(motion.mean(0))
    return {c: np.mean(vs, axis=0) for c, vs in by_cond.items()}


def main():
    per_body = {}
    for body, (spec, data_dir) in SOURCES.items():
        per_body[body] = mean_froude_per_condition(spec, data_dir)
        print(f"{body}: {len(per_body[body])} conditions loaded from {data_dir}")

    print(f"\n{'group':<10}{'hex cond':<16}{'hex froude':>12}{'b1 cond':<16}{'b1 froude':>12}"
         f"{'abs diff':>10}{'rel diff':>10}")
    flagged = []
    for group, (chan_name, chan_i) in GROUPS.items():
        hex_pool = [c for c in per_body["hexapod"] if condition_group(c) == group]
        b1_pool = [c for c in per_body["b1"] if condition_group(c) == group]
        if len(hex_pool) != len(b1_pool):
            print(f"  {group}: MISMATCHED COUNT hex={len(hex_pool)} b1={len(b1_pool)}")
            continue
        if group.startswith("side"):
            # side levels share a literal name across bodies ("side_L_lvl2" on both) -- match by
            # name directly, not by achieved-magnitude rank, so a level that came out mis-ordered
            # on one body (see F-note below) shows up as a mismatch instead of being silently
            # paired away by rank.
            pairs = [(c, c) for c in sorted(hex_pool)]
        else:
            hex_conds = sorted(hex_pool, key=lambda c: abs(per_body["hexapod"][c][chan_i]))
            b1_conds = sorted(b1_pool, key=lambda c: abs(per_body["b1"][c][chan_i]))
            pairs = list(zip(hex_conds, b1_conds))
        for hc, bc in pairs:
            hv = per_body["hexapod"][hc][chan_i]
            bv = per_body["b1"][bc][chan_i]
            diff = abs(hv - bv)
            rel = diff / max(abs(hv), abs(bv), 1e-6)
            flag = "  <-- >20% off" if rel > 0.20 else ""
            print(f"{group:<10}{hc:<16}{hv:>12.4f}{bc:<16}{bv:>12.4f}{diff:>10.4f}{rel:>9.1%}{flag}")
            if rel > 0.20:
                flagged.append((group, hc, bc, rel))

    print(f"\n{len(flagged)} condition pairs off by more than 20% (rank-matched within group, "
         "by achieved Froude on the group's own dominant channel).")
    for group, hc, bc, rel in flagged:
        print(f"  {group}: {hc} vs {bc}  ({rel:.1%} relative difference)")

    print("\n--- per-body monotonicity check: does achieved |Froude| increase with level index, "
         "within each body's own side conditions? (a naming/calibration sanity check, separate "
         "from the cross-body comparison above) ---")
    for body in per_body:
        for direction in ("side_L", "side_R"):
            chan_i = GROUPS[direction][1]
            conds = sorted((c for c in per_body[body] if condition_group(c) == direction),
                           key=lambda c: int(c.rsplit("lvl", 1)[1]))
            mags = [abs(per_body[body][c][chan_i]) for c in conds]
            ok = all(mags[i] < mags[i + 1] for i in range(len(mags) - 1))
            print(f"  {body:8s} {direction}: " +
                 "  ".join(f"{c}={m:.4f}" for c, m in zip(conds, mags)) +
                 ("" if ok else "   <-- NOT MONOTONIC"))


if __name__ == "__main__":
    main()
