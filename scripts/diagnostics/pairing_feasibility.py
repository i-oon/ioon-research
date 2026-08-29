"""Can a cross-embodiment `lambda_cross` be defined at all, before any GPU is spent on it?

Stage 1's cross-body loss decodes body A's latent against body B's frame and requires B's
command. It is well posed only because every insect body walks the *same expert episodes*, so at
a given timestep two bodies share the intent exactly and differ only in geometry. The hexapod and
the B1 share no episodes, so the pairing has to be reconstructed from something both robots
record. Per-leg contact is the candidate: it needs no shared gait period, and the four corner legs
correspond anatomically (F41b).

This decides whether that works **before** a training run, the same way the encoder probe on
slide 10 predicts a split's outcome for a few minutes of CPU rather than four GPU-hours.

Three conditions have to hold, in order:

  1. the label exists on both robots        -- a 4-bit corner pattern is computable for both
  2. it is not degenerate                   -- a robot parked in one pattern makes pairing a
                                               coin flip inside one huge bucket
  3. it pins down the command WITHIN one robot

**Condition 3 is the one that decides it, and it is the trap this script exists for.** It is
tempting to check only that both robots visit the same labels and call the pairing defined. But
`L_cross` supervises the decoder with the *partner's command*, so if a shared label does not
imply a similar command even on a single robot, a cross-embodiment pair built from it is a
**wrong label, not a noisy one** -- and wrong labels do not average out with more data. Measured
as matched-pair command distance over random-pair distance, within one body: 1.0 means the label
carries nothing and the pairing is a coin flip.

The second trap: a label can pass condition 3 and still be unusable, because *coverage* and
*meaning* trade against each other. Coarsening a label until both robots overlap is exactly the
operation that throws away what made it meaningful. Report both columns or the trade is invisible.

  .venv/bin/python3 scripts/diagnostics/pairing_feasibility.py
  .venv/bin/python3 scripts/diagnostics/pairing_feasibility.py --draws 8000
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.bodies import bodies_in, usable_clips  # noqa: E402
from wm.data.embodiment import REGISTRY, load as load_embodiment  # noqa: E402

# Which foot index is which leg, per embodiment: hexapod foot_order is FL ML HL FR MR HR, the
# B1's is FR FL RR RL. Same anatomical correspondence as leg_contact_probe.py's LEGS table --
# corners only, because the insect's middle legs have no counterpart on a quadruped. Kept in the
# order FL HL FR HR so a pattern code reads the same way for both robots.
HEX_CORNERS = [0, 2, 3, 5]
B1_CORNERS = [1, 3, 0, 2]


def labels_for(contact, corners):
    """Three candidate pairing labels on the same frames, coarse to fine."""
    bits = contact[:, corners]                       # (T, 4), columns FL HL FR HR
    n_down = bits.sum(1)                             # 0-4, coarsest
    # Which diagonal carries the load. A trot alternates between the two diagonals and a wave
    # passes through both, so this is the one description of a gait that assumes no shared period.
    diagonal = np.sign((bits[:, 0] + bits[:, 3]) - (bits[:, 1] + bits[:, 2])) + 1
    full = (bits * (1 << np.arange(4))).sum(1)       # 0-15, finest
    return {"n_feet_down": n_down, "diagonal": diagonal, "corner_pattern": full}


def gather(spec_name, data_dir, keep=None):
    """Labels, commands and body names for every usable clip of one embodiment."""
    spec = REGISTRY[spec_name]
    paths = usable_clips(sorted(glob.glob(os.path.join(ROOT, data_dir, "*.npz"))))
    if keep is not None:
        paths = [p for p in paths if os.path.basename(p).split("_")[0] in keep]
    if not paths:
        raise SystemExit(f"no usable clips in {data_dir}")
    corners = HEX_CORNERS if spec_name == "hexapod" else B1_CORNERS
    labels, actions, bodies = {}, [], []
    for path in paths:
        clip = load_embodiment(path, spec)
        for name, value in labels_for(clip["contact"], corners).items():
            labels.setdefault(name, []).append(value)
        actions.append(clip["actions"])
        bodies.extend([clip["body"]] * len(clip["actions"]))
    return ({k: np.concatenate(v) for k, v in labels.items()},
            np.concatenate(actions), np.array(bodies), len(paths))


def intent_ratio(label, actions, bodies, rng, draws):
    """Matched-pair command distance over random-pair distance, averaged across bodies.

    Standardised per body first, so one wide-swinging joint cannot dominate the mean, and
    computed within a single body so geometry is held constant and only the label varies.
    """
    ratios = []
    for body in sorted(set(bodies)):
        selected = bodies == body
        codes, commands = label[selected], actions[selected]
        commands = (commands - commands.mean(0)) / np.maximum(commands.std(0), 1e-6)
        matched, random_pairs = [], []
        for _ in range(draws):
            i = rng.integers(len(commands))
            same = np.flatnonzero(codes == codes[i])
            if len(same) < 2:
                continue
            matched.append(np.abs(commands[i] - commands[same[rng.integers(len(same))]]).mean())
            random_pairs.append(
                np.abs(commands[i] - commands[rng.integers(len(commands))]).mean())
        if matched:
            ratios.append(np.mean(matched) / np.mean(random_pairs))
    return float(np.mean(ratios)) if ratios else float("nan")


def print_pattern_table(hexapod_codes, b1_codes):
    """The full 16-way corner-pattern table: where each robot's time actually goes."""
    hexapod = np.bincount(hexapod_codes, minlength=16) / len(hexapod_codes)
    b1 = np.bincount(b1_codes, minlength=16) / len(b1_codes)
    print(f"  {'code':>5} {'FL HL FR HR':>13} {'hexapod':>9} {'b1':>9}  note")
    for code in range(16):
        if not (hexapod[code] or b1[code]):
            continue
        bits = "  ".join(str((code >> i) & 1) for i in range(4))
        if hexapod[code] and not b1[code]:
            note = "hexapod only, these frames get no partner"
        elif b1[code] and not hexapod[code]:
            note = "b1 only, these frames get no partner"
        elif b1[code] > 0.2:
            note = "carries most of the b1's time"
        else:
            note = ""
        print(f"  {code:>5} {bits:>13} {hexapod[code]:9.3f} {b1[code]:9.3f}  {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hexapod_dir", default="data/fwd_hex8body")
    parser.add_argument("--b1_dir", default="data/fwd_b1_50hz")
    parser.add_argument("--draws", type=int, default=4000,
                        help="sampled pairs per body for the intent ratio")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    keep = set(bodies_in(os.path.join(ROOT, args.hexapod_dir)))
    hexapod_labels, hexapod_actions, hexapod_bodies, hexapod_clips = gather(
        "hexapod", args.hexapod_dir, keep)
    b1_labels, b1_actions, b1_bodies, b1_clips = gather("b1", args.b1_dir)
    print(f"hexapod {hexapod_clips} clips over {len(set(hexapod_bodies))} bodies, "
          f"{len(hexapod_actions)} frames | b1 {b1_clips} clips, {len(b1_actions)} frames")

    print("\nWhere each robot's time goes, by corner pattern")
    print_pattern_table(hexapod_labels["corner_pattern"], b1_labels["corner_pattern"])

    print("\nThe trade, per candidate label")
    print(f"  {'label':>16} {'overlap':>9} {'hex paired':>11} {'b1 paired':>10} "
          f"{'intent hex':>11} {'intent b1':>10}")
    for name in ("n_feet_down", "diagonal", "corner_pattern"):
        hexapod_code, b1_code = hexapod_labels[name], b1_labels[name]
        size = max(hexapod_code.max(), b1_code.max()) + 1
        hexapod_share = np.bincount(hexapod_code, minlength=size) / len(hexapod_code)
        b1_share = np.bincount(b1_code, minlength=size) / len(b1_code)
        both = (hexapod_share > 0) & (b1_share > 0)
        print(f"  {name:>16} {np.minimum(hexapod_share, b1_share).sum():9.3f} "
              f"{hexapod_share[both].sum():11.1%} {b1_share[both].sum():10.1%} "
              f"{intent_ratio(hexapod_code, hexapod_actions, hexapod_bodies, rng, args.draws):11.3f} "
              f"{intent_ratio(b1_code, b1_actions, b1_bodies, rng, args.draws):10.3f}")

    print("\n  overlap     1.0 = both robots visit the label equally often, 0.0 = disjoint")
    print("  paired      share of that robot's frames whose label the other robot ever visits")
    print("  intent      matched over random command distance within one body. Lower means the")
    print("              label carries intent; 1.0 means it carries none, so a pair built from")
    print("              it gives the decoder a wrong target rather than a noisy one.")
    print("\n  A label is usable only if it is high on coverage AND low on intent, for BOTH")
    print("  robots. Coarsening a label to raise coverage is the same operation that destroys")
    print("  its meaning, so read the two together.")


if __name__ == "__main__":
    main()
