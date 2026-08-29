"""Can a cross-embodiment pairing be built in task space, where the contact labels failed?

F45 ruled out three pairing labels and **all three were built from foot contact** -- which is
precisely where a quadruped's trot and a hexapod's wave differ most. It looked for a correspondence
in the place the two robots have least in common, and found none: the coarse labels pair every
frame and mean nothing, the fine one means something and pairs a third.

F55 argues that this single missing pairing is what leaves `lambda_cross` undefinable, which leaves
nothing forcing one `z` to mean the same thing on both robots, which is why the trunk partitions and
why learned dynamics do not travel. So it is worth one more attempt in a space the contact labels
never touched.

**The candidate.** Two quantities, both defined for a body with any number of legs:

    phase     where the robot is within its own gait cycle, normalised to [0, 1)
    Froude    v / sqrt(g * h), forward speed against hip height -- the standard biomechanical
              way to compare gaits across animals of different size

Neither asks the two robots to have the same gait. Phase is measured against each robot's *own*
cycle, so a 19-frame insect stride and a B1 trot are both mapped onto the same [0, 1). Froude
divides out body size, so 0.18 m/s at 0.13 m of hip height and 0.30 m/s at 0.56 m land near each
other, which absolute speed would not.

**Same three conditions as `pairing_feasibility.py`, in the same order, so the two are comparable:**

  1. the label exists on both robots
  2. it is not degenerate -- both robots have to actually visit the shared cells
  3. **it pins down the command WITHIN one robot**

Condition 3 is what killed the coarse contact labels and it is the one to read first. `L_cross`
supervises the decoder with the *partner's* command, so a label that does not imply a similar
command even on a single robot yields a **wrong** target rather than a noisy one, and wrong targets
do not average out with more data.

**The trap this shares with F45.** Coverage and meaning trade against each other: widen the bins
until both robots overlap and that is the same operation that destroys what the label meant. Both
columns are reported, or the trade is invisible.

  .venv/bin/python3 scripts/diagnostics/pairing_taskspace.py
  .venv/bin/python3 scripts/diagnostics/pairing_taskspace.py --phase_bins 8 --froude_bins 4
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from wm.bodies import CONTACT_THRESHOLD, usable_clips  # noqa: E402

G = 9.81

# Frame rates differ and the label depends on a velocity, so they cannot be left implicit.
# The insect's comes from `sim_time` in the expert CSV (20 Hz); the B1's from DECIMATION 4 on a
# 5 ms MuJoCo step, which is the 50 Hz its collector documents.
DT = {"hexapod": 0.05, "b1": 0.02}


# Which contact column is the front-left foot. The hexapod records six legs in the order
# FL ML HL FR MR HR; the B1 records four as FR FL RR RL. Same anatomical correspondence as
# pairing_feasibility.py and leg_contact_probe.py.
FRONT_LEFT = {"hexapod": 0, "b1": 1}


def phase_of(contact, kind):
    """Gait phase in [0, 1), anchored at front-left touchdown and interpolated between strides.

    **This replaced a Hilbert phase of the joint commands' first principal component**, which is
    the textbook way to get an instantaneous phase and was wrong here. Checked against the frame a
    reference foot actually lands, that estimate put the B1's touchdown at a tight 0.25-0.31 and
    the insect's at 0.41, 0.45, 0.61, 0.76, 0.87 -- five successive strides of the *same* leg in
    one clip, which should all read alike. The insect's stride is stable at 19 frames on every clip
    and body (F53), so the gait was periodic and the estimator was not.

    Anchoring on a gait event fixes both problems at once: phase 0 is the same anatomical instant
    on either robot, so "same phase" means the same thing across them, which is the whole point of
    a pairing label.

    Contact is used **only to place the origin**, never as the label itself -- that is what
    separates this from the three labels F45 ruled out. Where the stride begins is a question
    inside one robot; it does not ask the two gaits to resemble each other.
    """
    down = contact[:, FRONT_LEFT[kind]] > 0.5
    touchdown = np.flatnonzero(down[1:] & ~down[:-1]) + 1
    n = len(contact)
    if len(touchdown) < 2:
        return np.full(n, np.nan)
    phase = np.full(n, np.nan)
    for start, stop in zip(touchdown[:-1], touchdown[1:]):
        phase[start:stop] = np.linspace(0, 1, stop - start, endpoint=False)
    # Frames before the first stride and after the last get the median stride's pace, so a clip
    # is not silently trimmed to whole strides.
    period = float(np.median(np.diff(touchdown)))
    head = np.arange(touchdown[0]) - touchdown[0]
    phase[:touchdown[0]] = (head / period) % 1.0
    tail = np.arange(touchdown[-1], n) - touchdown[-1]
    phase[touchdown[-1]:] = (tail / period) % 1.0
    return phase


def froude_of(position, dt, height):
    """v / sqrt(g * h) per frame, forward speed only, smoothed over five frames.

    Raw frame-to-frame displacement is dominated by the body rocking with each step, so it is
    smoothed before the ratio is taken; the label is meant to say how fast the robot is travelling,
    not where in the stride the body happens to be rising.
    """
    forward = np.gradient(position[:, 0], dt)
    kernel = np.ones(5) / 5
    smooth = np.convolve(forward, kernel, mode="same")
    return smooth / np.sqrt(G * height)


def gather(data_dir, kind, glob_pat="*.npz"):
    """Phase, Froude, commands and per-clip identity for one embodiment."""
    paths = sorted(glob.glob(os.path.join(ROOT, data_dir, glob_pat)))
    if kind == "hexapod":
        paths = usable_clips(paths)
    if not paths:
        raise SystemExit(f"no usable clips in {data_dir}")
    phase, froude, actions, groups = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as clip:
            command = clip["actions"] if "actions" in clip.files else clip["action"]
            position = clip["head"] if "head" in clip.files else clip["base_pos"]
            contact = (clip["forces"] > CONTACT_THRESHOLD if "forces" in clip.files
                       else clip["foot_contact"] > 0.5).astype(float)
            group = (str(clip["morph"]) if "morph" in clip.files
                     else os.path.basename(path).split("_")[0])
        height = float(np.median(position[:, 2]))
        phase.append(phase_of(contact, kind))
        froude.append(froude_of(position, DT[kind], height))
        actions.append(command)
        groups.extend([group] * len(command))
    phase, froude = np.concatenate(phase), np.concatenate(froude)
    actions, groups = np.concatenate(actions), np.array(groups)
    keep = ~np.isnan(phase)
    if not keep.all():
        print(f"  dropped {int((~keep).sum())} frames from clips with fewer than two strides")
    return phase[keep], froude[keep], actions[keep], groups[keep], len(paths)


def cells(phase, froude, edges, phase_bins):
    """One integer per frame: which (phase, Froude) cell it falls in."""
    p = np.clip((phase * phase_bins).astype(int), 0, phase_bins - 1)
    f = np.clip(np.digitize(froude, edges) - 1, 0, len(edges) - 2)
    return p * (len(edges) - 1) + f


def intent_ratio(label, actions, groups, rng, draws):
    """Matched-pair command distance over random-pair distance, within one body.

    Identical to `pairing_feasibility.py` so the two labels can be read against each other.
    1.0 means the label carries nothing; lower is better.
    """
    ratios = []
    for group in sorted(set(groups)):
        selected = groups == group
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hexapod_dir", default="data/fwd_m3d")
    ap.add_argument("--b1_dir", default="data/fwd_b1_50hz")
    ap.add_argument("--phase_bins", type=int, default=8)
    ap.add_argument("--froude_bins", type=int, default=3)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    hex_phase, hex_fr, hex_act, hex_grp, hex_n = gather(args.hexapod_dir, "hexapod")
    b1_phase, b1_fr, b1_act, b1_grp, b1_n = gather(args.b1_dir, "b1")
    print(f"hexapod {hex_n} clips, {len(hex_phase)} frames, {len(set(hex_grp))} bodies")
    print(f"b1      {b1_n} clips, {len(b1_phase)} frames\n")

    print(f"{'':<10}{'Froude mean':>13}{'5th-95th pct':>18}{'hip height m':>14}")
    for name, fr in (("hexapod", hex_fr), ("b1", b1_fr)):
        lo, hi = np.percentile(fr, [5, 95])
        print(f"{name:<10}{fr.mean():>13.3f}{f'{lo:.3f} - {hi:.3f}':>18}", end="")
        print(f"{'0.13' if name == 'hexapod' else '0.56':>14}")

    # Bin edges span both robots, so a cell means the same thing on each side.
    both = np.concatenate([hex_fr, b1_fr])
    edges = np.quantile(both, np.linspace(0, 1, args.froude_bins + 1))
    edges[0], edges[-1] = both.min() - 1e-9, both.max() + 1e-9

    hex_cells = cells(hex_phase, hex_fr, edges, args.phase_bins)
    b1_cells = cells(b1_phase, b1_fr, edges, args.phase_bins)
    n_cells = args.phase_bins * (len(edges) - 1)
    hex_hist = np.bincount(hex_cells, minlength=n_cells) / len(hex_cells)
    b1_hist = np.bincount(b1_cells, minlength=n_cells) / len(b1_cells)

    overlap = float(np.minimum(hex_hist, b1_hist).sum())
    shared = (hex_hist > 0) & (b1_hist > 0)
    hex_pairable = float(hex_hist[shared].sum())
    b1_pairable = float(b1_hist[shared].sum())

    print(f"\n{args.phase_bins} phase bins x {len(edges) - 1} Froude bins = {n_cells} cells")
    print(f"  cells visited by both robots      {int(shared.sum())} of {n_cells}")
    print(f"  **overlap**                       {overlap:.3f}   (1.0 = identical distributions)")
    print(f"  hexapod frames that get a partner {hex_pairable:.1%}")
    print(f"  b1 frames that get a partner      {b1_pairable:.1%}")

    print("\ncondition 3 -- does the label pin down the command within one robot?")
    print(f"  {'label':<28}{'hexapod':>10}{'b1':>10}   (1.0 = carries nothing)")
    hex_r = intent_ratio(hex_cells, hex_act, hex_grp, rng, args.draws)
    b1_r = intent_ratio(b1_cells, b1_act, b1_grp, rng, args.draws)
    print(f"  {'phase x Froude':<28}{hex_r:>10.3f}{b1_r:>10.3f}")
    phase_only = np.clip((hex_phase * args.phase_bins).astype(int), 0, args.phase_bins - 1)
    b1_phase_only = np.clip((b1_phase * args.phase_bins).astype(int), 0, args.phase_bins - 1)
    print(f"  {'phase alone':<28}"
          f"{intent_ratio(phase_only, hex_act, hex_grp, rng, args.draws):>10.3f}"
          f"{intent_ratio(b1_phase_only, b1_act, b1_grp, rng, args.draws):>10.3f}")

    print("\nCompare with F45's contact labels, on the same three conditions:")
    print("  feet down 0-4       overlap 0.572  pairable 98.9%  intent 0.998 on the b1")
    print("  diagonal loaded     overlap 0.711  pairable  100%  intent 0.918 on the hexapod")
    print("  corner pattern      overlap 0.240  pairable 33.8%  intent 0.63 / 0.52")
    print("\nA usable label needs high overlap AND a low intent ratio on BOTH robots. The contact")
    print("labels never managed both at once; read the two numbers above together, not separately.")


if __name__ == "__main__":
    main()
