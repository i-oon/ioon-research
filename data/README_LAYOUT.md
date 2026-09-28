# data/ layout

    allocentric/   every set collected before 2026-09-01: the fixed third-person camera
    egocentric/    head-mounted camera, randomised room

**The split is the camera, not the behaviour.** The same twelve conditions exist on both sides, and
a number measured under one view cannot be compared with a number measured under the other -- that
difference is the whole subject of F155.

**Moving these directories breaks symlinked sets.** `beh12_b1_flat_9clips` (no longer present as of
2026-09-23; presumably retired deliberately, not re-derivable from disk) and
`beh10_c10f10t10_intent2_flat` are/were symlink farms into their parents, with absolute targets; the
move on 2026-09-01 broke 34 of them and they were repaired by rewriting the targets, not by
re-collecting. **A broken symlink reads as a missing file, several scripts deep.** The same pattern
recurred on 2026-09-23 -- 49 more absolute symlinks (pointing at `/home/aria/...`) found broken
across `cf_confirm/`, `beh10_c10f10t10_intent2_flat/`, and the two `*_ego_flat_cleanval/` sets, fixed
the same way (relative symlinks to the local sibling that already had the file).

`data/README.md` carries the naming rule and what each set is.

**2026-09-24: the working tree lost 621 real files under `data/egocentric/` that git still had.**
All 621 showed as `D` (deleted, tracked) in `git status`, not `??` -- the files were tracked and
their blobs were intact in `HEAD`, they had just been deleted from disk at some point without the
deletion being committed. Restored with a plain `git checkout -- data/egocentric/`; nothing was
actually lost. If a set under `data/egocentric/` looks empty or its clean-split symlinks are broken,
check `git status --porcelain <path> | awk '$1=="D"'` before assuming the raw data is gone -- it
may just need a checkout, not a re-collection.

**That checkout then exposed a second, unrelated bug**: 213 more symlinks, this time broken by a
wrong relative-path depth (some pointed one directory too deep, duplicating the parent dir's own
name in the target; some still had stale absolute `/home/aria/...` targets, same pattern as the
2026-09-23 fix above). Fixed by indexing every real (non-symlink) `.npz` by basename and relinking
each broken symlink to the one real file matching both its basename and its own family name (to
disambiguate e.g. `beh12_b1_ego_flat` vs `beh24_b1_ego_flat` vs `beh12_b1_more_ego_flat`, which can
share basenames). `scripts/diagnostics/objective_experiments/family_z_ceiling.py` and
`froude_ceiling.py` also had a related bug in unrelated code -- their `FAMILY` grouping used a
plain `rsplit("_", 1)`, which mis-parses beh24's two-suffix condition names (`speed_c5.8_bwd`,
`turn_s0.05_neg`) and silently dropped every `_bwd`/`_neg` condition into its own singleton family,
excluding exactly the new beh24 conditions from every ceiling check that used them. Both scripts
now parse the numeric magnitude/level token specifically and keep the mode suffix.
