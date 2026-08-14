"""Build the symlink directories the five Stage 1 retrains read from.

Three constraints have to hold at once, and no directory already on disk satisfies them:

  1. Every clip walks. A clip counts as walking if the body's head travelled at least
     WALK_FORWARD_M forward (signed, so a body that reverses fails) and drifted less than
     WALK_LATERAL_M sideways. The original runs trained on bodies that failed 30/30 -- see
     FINDINGS.md F42 -- and even the sound bodies scatter a few clips each.

  2. tib_cross and bracket_cross see the same number of clips. Slide 9 claims wider
     femur/tibia coverage helps *at matched data volume*; without matching, a better score is
     just more data. Four bodies and six bodies can only be matched by giving each body a
     different clip count, which is why this takes two directories rather than one.

  3. Both are scored on identical held-out frames. The same 20 clips of c10f10t08 are linked
     into both, so the two runs differ in exactly one thing.

Matching at good clips only caps the volume: c06f10t10 has 25 usable clips, so 4 x 24 = 96 is
the largest matched pair reachable, against 4 x 30 = 120 if bad clips were kept. The 20% of
data given up buys a training set with nothing in it that veers.

Clips are spread evenly across episode numbers rather than taken as a prefix, and the six-body
directory draws its shared bodies from the four-body directory's selection, so the smaller
sample is a subset of the larger one rather than an independent draw that might land luckier.

  .venv/bin/python3 scripts/build_stage1_dirs.py
"""
import glob
import os
import sys
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
from wm.bodies import WALK_FORWARD_M, WALK_LATERAL_M, evenly, walks  # noqa: E402
DATA = os.path.join(ROOT, "data")

SOURCE = {
    "c10f10t10": "ik_walk_8body",
    "c06f10t10": "ik_walk_8body",
    "c10f06t06": "ik_walk_8body",
    "c08f09t09": "ik_walk_8body",
    "c06f06t06": "ik_walk_8body",
    "c10f09t07": "ik_walk_decoupled",
    "c10f08t06": "ik_walk_decoupled",
    "c10f10t08": "ik_walk_decoupled",
}

M3D_TRAIN = ("c10f10t10", "c06f10t10", "c10f06t06", "c06f06t06")
M3D_HELDOUT = "c08f09t09"

NARROW_TRAIN = ("c10f10t10", "c06f10t10", "c10f06t06", "c08f09t09")
WIDE_TRAIN = NARROW_TRAIN + ("c10f09t07", "c10f08t06")
COV_HELDOUT = "c10f10t08"

PER_BODY_NARROW = 24   # 4 x 24 = 96
PER_BODY_WIDE = 16     # 6 x 16 = 96
PER_BODY_HELDOUT = 20


def good_clips(body):
    directory = os.path.join(DATA, SOURCE[body])
    return [p for p in sorted(glob.glob(os.path.join(directory, f"{body}_ep*.npz")))
            if walks(p)]


def episode_of(path):
    return os.path.basename(path).split("_ep")[1].split(".")[0]


def shared_episodes(pool, bodies):
    """Episodes every one of `bodies` has a walking clip for.

    `lambda_cross` decodes one body's latent against another body's frame at the same episode and
    timestep, so it is defined only where a partner body exists. `MorphologyPairs.__getitem__`
    emits `cross_x_t` only when one does, and `default_collate` then dies with a bare
    `KeyError: 'cross_x_t'` the moment a batch mixes samples with and without it.

    Capping each body against its *own* usable list is what breaks this: the bodies fail the walk
    check on different episodes, so evenly-spaced picks from different lists leave episodes held
    by a single body. Selecting from the intersection restores the invariant this module's
    docstring already claims -- every body walks the same expert episodes.
    """
    return set.intersection(*[{episode_of(p) for p in pool[b]} for b in bodies])


def restrict(pool, bodies, episodes):
    """Each body's clips, filtered to a shared episode set."""
    return {b: [p for p in pool[b] if episode_of(p) in episodes] for b in bodies}


def build(name, bodies, per_body):
    directory = os.path.join(DATA, name)
    if os.path.islink(directory):
        os.unlink(directory)
    elif os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    total = 0
    for body, paths in bodies.items():
        chosen = evenly(paths, per_body.get(body, per_body["_"]))
        for path in chosen:
            os.symlink(os.path.relpath(path, directory),
                       os.path.join(directory, os.path.basename(path)))
        total += len(chosen)
        print(f"  {body:11} {len(chosen):3d} clips")
    print(f"  {'total':11} {total:3d} clips -> data/{name}")
    return total


def main():
    pool = {body: good_clips(body) for body in SOURCE}
    print("usable clips at source (walked forward >= "
          f"{WALK_FORWARD_M} m, drifted < {WALK_LATERAL_M} m):")
    for body, paths in pool.items():
        available = len(glob.glob(os.path.join(DATA, SOURCE[body], f"{body}_ep*.npz")))
        print(f"  {body:11} {len(paths):3d} / {available}")

    print("\nik_walk_m3d_clean  (m3d_cross, m3d_bracketed)")
    # no matched-volume claim rests on this pair -- they share one directory -- so it keeps
    # every usable clip rather than levelling down to the scarcest body
    build("ik_walk_m3d_clean",
          {b: pool[b] for b in M3D_TRAIN + (M3D_HELDOUT,)},
          {"_": 10 ** 6})

    heldout = evenly(pool[COV_HELDOUT], PER_BODY_HELDOUT)

    # Both coverage directories draw from episodes shared by all of their training bodies, so no
    # episode is left with a single body and `lambda_cross` is defined for every sample. The
    # held-out body is exempt: it is never trained on, so it needs no partner.
    narrow_shared = shared_episodes(pool, NARROW_TRAIN)
    wide_shared = shared_episodes(pool, WIDE_TRAIN)
    print(f"\nepisodes shared by all training bodies: narrow {len(narrow_shared)} "
          f"(need {PER_BODY_NARROW}), wide {len(wide_shared)} (need {PER_BODY_WIDE})")
    for label, shared, need in (("narrow", narrow_shared, PER_BODY_NARROW),
                                ("wide", wide_shared, PER_BODY_WIDE)):
        if len(shared) < need:
            raise SystemExit(f"{label}: only {len(shared)} shared episodes, need {need}. "
                             "Collect more clips or lower the per-body count.")

    print("\nik_walk_cov_narrow  (tib_cross, tib_ctrl)")
    narrow = restrict(pool, NARROW_TRAIN, narrow_shared)
    narrow[COV_HELDOUT] = heldout
    build("ik_walk_cov_narrow", narrow,
          {"_": PER_BODY_NARROW, COV_HELDOUT: PER_BODY_HELDOUT})

    print("\nik_walk_cov_wide  (bracket_cross)")
    # the four shared bodies subsample the narrow run's own selection, so wide sees a subset of
    # what narrow saw rather than an independent draw of the same size
    narrow_choice = {b: evenly(clips, PER_BODY_NARROW)
                     for b, clips in restrict(pool, NARROW_TRAIN, wide_shared).items()}
    wide = {b: narrow_choice.get(b) or restrict(pool, [b], wide_shared)[b] for b in WIDE_TRAIN}
    wide[COV_HELDOUT] = heldout
    build("ik_walk_cov_wide", wide,
          {"_": PER_BODY_WIDE, COV_HELDOUT: PER_BODY_HELDOUT})


if __name__ == "__main__":
    main()
