"""Single source of truth for which bodies and clips count as valid, and what "contact" means.

Everything here was previously copy-pasted. Nine scripts carried their own hardcoded body list,
four defined their own contact threshold, and the walk check lived only in the collector, so
nothing that consumed a dataset could re-check it. That is not a tidiness problem: a hardcoded
list goes stale silently the moment a split changes, and `z_identity_ablation` once read 15.99 deg
where the same checkpoint on its actual training bodies read 1.45 (FINDINGS.md F39).

Import from here rather than redefining. This module deliberately depends on nothing heavier than
numpy and `wm.config`, so the collector in `sim/` and the diagnostics in `scripts/` can both use it
without pulling in torch.
"""
import glob
import os

import numpy as np

# --------------------------------------------------------------------------------------
# Contact
# --------------------------------------------------------------------------------------

# Newtons. The foot force histogram is bimodal and this sits in the empty valley between the
# modes, with 1.8% of samples within +/-0.07 N of it. Verify with plot_gait_quality.py before
# trusting any stance-derived label on a new dataset -- the threshold is a property of the scene's
# contact solver, not a universal constant.
CONTACT_THRESHOLD = 0.27


def contact_labels(forces):
    """Per-foot binary stance, one row per timestep."""
    return (forces > CONTACT_THRESHOLD).astype(np.int64)


# --------------------------------------------------------------------------------------
# Does it walk
# --------------------------------------------------------------------------------------

# Metres, per episode. Every sound body in ik_walk_8body clears both; the two that collapse fail
# forward, the two that veer fail lateral.
WALK_FORWARD_M = 0.30
WALK_LATERAL_M = 0.20


def walk_check(head):
    """Signed forward travel and lateral drift, reported separately.

    Never collapse these into one distance. The earlier check used
    `norm(head[-1,:2] - head[0,:2])`, which is unsigned, and a body that tipped over and rotated on
    the spot read a healthy 0.46 m -- two such bodies reached the dataset and trained into every
    Stage 2 run before anyone looked at the frames (FINDINGS.md F42). A body can also travel a long
    way forward while crabbing just as far sideways, which one number hides.

    A pass here still is not proof. No statistic distinguishes "walks oddly" from "fell over and is
    now spinning"; watch the clip.
    """
    if not len(head):
        return 0.0, 0.0, "EMPTY"
    forward = float(head[-1, 0] - head[0, 0])
    lateral = float(abs(head[-1, 1] - head[0, 1]))
    if forward < WALK_FORWARD_M:
        verdict = "FAILS forward" + (" (BACKWARDS)" if forward < 0 else "")
    elif lateral > WALK_LATERAL_M:
        verdict = "FAILS lateral, veering"
    else:
        verdict = "ok"
    return forward, lateral, verdict


def walks(path):
    """Whether one recorded clip shows the body walking. Reads only `head`, so it is cheap."""
    with np.load(path) as data:
        if "head" not in data:
            return True   # not a locomotion recording; nothing to check
        head = data["head"]
    return walk_check(head)[2] == "ok"


# --------------------------------------------------------------------------------------
# Bodies that never walk
# --------------------------------------------------------------------------------------

# Geometry, not behaviour: each of these asks the IK for a foot target its legs cannot reach, so
# the failure is a property of the body and holds in every dataset it appears in. Excluded
# everywhere by default rather than by remembering to name the good ones -- the convention that
# broke, since naming works only until something globs the directory instead (FINDINGS.md F42).
EXCLUDED_BODIES = {
    "c06f06t10": "collapses and rotates on the spot; 208 mm dead zone, 29/30 clips fail",
    "c10f06t10": "collapses and travels backwards; 208 mm dead zone, 30/30 clips fail",
    "c06f10t06": "veers 0.36 m off course; 94.6 mm dead zone, 30/30 clips fail",
    "c10f10t06": "veers 0.43 m off course; 94.6 mm dead zone, 30/30 clips fail",
}


def usable_clips(paths, excluded=EXCLUDED_BODIES):
    """Drop clips belonging to bodies that do not walk, reporting what went."""
    kept = [p for p in paths if body_of(p) not in excluded]
    dropped = len(paths) - len(kept)
    if dropped:
        names = sorted({body_of(p) for p in paths} & set(excluded))
        print(f"excluding {dropped} clips from non-walking bodies {names} "
              f"({len(kept)} clips remain)")
    return kept


# --------------------------------------------------------------------------------------
# Finding bodies
# --------------------------------------------------------------------------------------

def body_of(path):
    """The body name a clip belongs to. Clips are named `<body>_ep<episode>[_r<repeat>].npz`."""
    return os.path.basename(path).split("_")[0]


def bodies_in(data_dir, exclude=True):
    """Every body with clips in a directory, sorted, non-walking ones dropped by default.

    Use this instead of writing a body list into a script. A literal list is correct only until
    the next dataset changes, and nothing warns you when it stops being.
    """
    found = {body_of(p) for p in glob.glob(os.path.join(data_dir, "*.npz"))}
    return sorted(found - set(EXCLUDED_BODIES) if exclude else found)


def training_bodies(cfg, embodiment="hexapod", root=None):
    """The bodies a run actually trained on, read off its own config rather than assumed.

    Three diagnostics once scored a model on bodies it had never seen, because each carried its
    own literal list and two of the five in it were held out and veer.

    Single-morphology runs name their bodies. Cross-embodiment runs name a directory and glob it,
    so the set is what was globbed minus the non-walking bodies and minus the deliberately
    held-out ones.
    """
    if not cfg.sources:
        return sorted(cfg.train_morphs)
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bodies = set()
    for spec in cfg.sources:
        name, _, data_dir = spec.partition("=")
        if name != embodiment:
            continue
        directory = data_dir if os.path.isabs(data_dir) else os.path.join(root, data_dir)
        bodies |= set(bodies_in(directory, exclude=False))
    return sorted(bodies - set(cfg.heldout_bodies or ()) - set(EXCLUDED_BODIES))


# --------------------------------------------------------------------------------------
# Sampling clips
# --------------------------------------------------------------------------------------

def evenly(items, keep):
    """`keep` items spread across the list, not the first `keep`.

    Clips are sorted by episode, so taking a prefix would take consecutive expert episodes and the
    behavioural range would narrow along with the count.
    """
    if keep >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, keep).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx)]
